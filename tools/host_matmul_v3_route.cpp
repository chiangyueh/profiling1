#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "base/context_builder/op_tiling_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"
#include "platform/platform_infos_def.h"

extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char *soPath);

namespace {

struct Workload {
    std::string dtype;
    std::string layout;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
};

struct RouteResult {
    uint32_t rc = 0;
    uint64_t key = 0;
    uint32_t blockDim = 0;
};

ge::DataType ParseDtype(const std::string &value)
{
    if (value == "fp16" || value == "fp16_fp16") {
        return ge::DT_FLOAT16;
    }
    if (value == "bf16" || value == "bf16_bf16") {
        return ge::DT_BF16;
    }
    if (value == "fp32" || value == "fp32_fp32") {
        return ge::DT_FLOAT;
    }
    return ge::DT_UNDEFINED;
}

bool ParsePositive(const std::string &value, int64_t &out)
{
    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed <= 0) {
        return false;
    }
    out = static_cast<int64_t>(parsed);
    return true;
}

bool ParseLine(const std::string &line, Workload &workload)
{
    std::istringstream stream(line);
    std::string m;
    std::string n;
    std::string k;
    if (!(stream >> workload.dtype >> workload.layout >> m >> n >> k)) {
        return false;
    }
    std::string extra;
    if (stream >> extra) {
        return false;
    }
    return ParseDtype(workload.dtype) != ge::DT_UNDEFINED &&
           (workload.layout == "NN" || workload.layout == "NT" ||
            workload.layout == "TN" || workload.layout == "TT") &&
           ParsePositive(m, workload.m) && ParsePositive(n, workload.n) && ParsePositive(k, workload.k);
}

const gert::OpImplKernelRegistry::OpImplFunctionsV2 *GetMatmulV3Impl()
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (registry == nullptr) {
        return nullptr;
    }
    return registry->GetOpImpl("MatMulV3");
}

optiling::MatmulV3CompileInfo MakeCompileInfo()
{
    optiling::MatmulV3CompileInfo info;
    info.aicNum = 20;
    info.aivNum = 40;
    info.ubSize = 196608;
    info.l1Size = 524288;
    info.l2Size = 201326592;
    info.l0CSize = 131072;
    info.l0ASize = 65536;
    info.l0BSize = 65536;
    info.btSize = 1024;
    info.cubeFreq = 1800.0F;
    info.socVersion = platform_ascendc::SocVersion::ASCEND910B;
    info.socVersionStr = "Ascend910B3";
    info.supportL0c2out = true;
    info.supportL12BtBf16 = false;
    return info;
}

const char *Family(uint64_t key)
{
    const uint64_t load = key & 0xFULL;
    const uint64_t split = (key >> 4U) & 0xFFULL;
    const uint64_t fix = (key >> 12U) & 0xFULL;
    const uint64_t nd2nz = (key >> 16U) & 0xFULL;
    const uint64_t special = (key >> 20U) & 0xFULL;
    if (load == 0 && split == 0 && fix == 0 && nd2nz == 1 && special == 0) {
        return "BASE";
    }
    return "NON_BASE";
}

void ClearObserved()
{
    static const char *names[] = {
        "MATMUL_OBSERVED_KEY", "MATMUL_OBSERVED_CORES", "MATMUL_OBSERVED_SINGLE_M",
        "MATMUL_OBSERVED_SINGLE_N", "MATMUL_OBSERVED_SINGLE_K", "MATMUL_OBSERVED_BASE_M",
        "MATMUL_OBSERVED_BASE_N", "MATMUL_OBSERVED_BASE_K"
    };
    for (const char *name : names) {
        unsetenv(name);
    }
}

const char *Observed(const char *name)
{
    const char *value = std::getenv(name);
    return value == nullptr ? "" : value;
}

RouteResult RunOne(const gert::OpImplKernelRegistry::OpImplFunctionsV2 *impl, const Workload &workload,
                   bool printResult)
{
    const bool transA = workload.layout[0] == 'T';
    const bool transB = workload.layout[1] == 'T';
    const ge::DataType dtype = ParseDtype(workload.dtype);

    const gert::StorageShape aShape = transA ?
        gert::StorageShape({workload.k, workload.m}, {workload.k, workload.m}) :
        gert::StorageShape({workload.m, workload.k}, {workload.m, workload.k});
    const gert::StorageShape bShape = transB ?
        gert::StorageShape({workload.n, workload.k}, {workload.n, workload.k}) :
        gert::StorageShape({workload.k, workload.n}, {workload.k, workload.n});
    const gert::StorageShape cShape({workload.m, workload.n}, {workload.m, workload.n});
    const gert::StorageFormat ndFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor a(aShape, ndFormat, dtype);
    gert::Tensor b(bShape, ndFormat, dtype);
    gert::Tensor c(cShape, ndFormat, dtype);
    std::vector<gert::Tensor *> inputs{&a, &b};
    std::vector<gert::Tensor *> outputs{&c};

    auto compileInfo = MakeCompileInfo();
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socInfo = {
        {"ai_core_cnt", "20"}, {"cube_core_cnt", "20"}, {"vector_core_cnt", "40"},
        {"core_type_list", "AICore"}, {"l2_size", "201326592"}
    };
    std::map<std::string, std::string> aiCoreSpec = {
        {"ub_size", "196608"}, {"l0_a_size", "65536"}, {"l0_b_size", "65536"},
        {"l0_c_size", "131072"}, {"l1_size", "524288"}, {"bt_size", "1024"},
        {"cube_freq", "1800"}
    };
    std::map<std::string, std::string> intrinsics = {{"Intrinsic_fix_pipe_l0c2out", "float16"}};
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend910B"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfo);
    platformInfo.SetPlatformRes("AICoreSpec", aiCoreSpec);
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo.SetPlatformRes("version", version);
    platformInfo.SetCoreNumByCoreType("AICore");
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto *workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    gert::OpTilingContextBuilder builder;
    auto holder = builder.OpType("MatMulV3")
        .OpName("MatMulV3_host_route")
        .IONum(2, 1)
        .AppendAttr(transA)
        .AppendAttr(transB)
        .AppendAttr(static_cast<int64_t>(0))
        .AppendAttr(static_cast<int64_t>(0))
        .CompileInfo(&compileInfo)
        .PlatformInfo(&platformInfo)
        .Deterministic(0)
        .TilingDataSize(4096)
        .Workspace(workspace)
        .InputTensors(inputs)
        .OutputTensors(outputs)
        .Build();
    auto *context = holder.GetContext();
    ClearObserved();
    RouteResult result;
    result.rc = impl->tiling(context);
    result.key = result.rc == ge::GRAPH_SUCCESS ? context->GetTilingKey() : 0;
    result.blockDim = result.rc == ge::GRAPH_SUCCESS ? context->GetBlockDim() : 0;

    if (printResult) {
        std::cout << workload.dtype << ',' << workload.layout << ',' << workload.m << ',' << workload.n << ','
                  << workload.k << ',' << result.rc << ',' << result.key << ','
                  << (result.rc == ge::GRAPH_SUCCESS ? Family(result.key) : "FAILED")
                  << ',' << result.blockDim << ',' << Observed("MATMUL_OBSERVED_CORES")
                  << ',' << Observed("MATMUL_OBSERVED_SINGLE_M") << ',' << Observed("MATMUL_OBSERVED_SINGLE_N")
                  << ',' << Observed("MATMUL_OBSERVED_SINGLE_K") << ',' << Observed("MATMUL_OBSERVED_BASE_M")
                  << ',' << Observed("MATMUL_OBSERVED_BASE_N") << ',' << Observed("MATMUL_OBSERVED_BASE_K") << '\n';
    }
    return result;
}

}  // namespace

int main(int argc, char **argv)
{
    const bool stratifiedMode = argc == 5 && std::strcmp(argv[1], "--select-base-stratified") == 0;
    const bool selectMode = argc == 5 &&
        (std::strcmp(argv[1], "--select-base") == 0 || stratifiedMode);
    if (argc != 2 && !selectMode) {
        std::cerr << "usage: host_matmul_v3_route LIBOPHOST_NN_SO\n"
                  << "       host_matmul_v3_route --select-base CAMPAIGN PER_M_RESERVE LIBOPHOST_NN_SO\n"
                  << "       host_matmul_v3_route --select-base-stratified CAMPAIGN PER_CELL_RESERVE LIBOPHOST_NN_SO\n";
        return 2;
    }
    const char *campaign = selectMode ? argv[2] : nullptr;
    const size_t perMReserve = selectMode ? std::strtoull(argv[3], nullptr, 10) : 0;
    const char *hostLibrary = selectMode ? argv[4] : argv[1];
    if (selectMode && perMReserve == 0) {
        std::cerr << "PER_M_RESERVE must be positive\n";
        return 2;
    }
    setenv("MATMUL_DISABLE_REPO", "1", 1);
    unsetenv("MATMUL_BASE_MODE");
    unsetenv("MATMUL_BASE_FULL_M_SWEEP");
    unsetenv("MATMUL_SPLITK_MODE");
    unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE");
    if (TbeLoadSoAndSaveToRegistry(hostLibrary) != 0U) {
        std::cerr << "host registration failed\n";
        return 2;
    }
    const auto *impl = GetMatmulV3Impl();
    if (impl == nullptr || impl->tiling == nullptr) {
        std::cerr << "MatMulV3 tiling callback is not registered\n";
        return 2;
    }
    if (!selectMode) {
        std::cout << "dtype,layout,m,n,k,rc,tiling_key,family,block_dim,used_core,single_m,single_n,single_k,base_m,base_n,base_k\n";
    }
    std::string line;
    size_t invalid = 0;
    size_t failed = 0;
    size_t officialBase = 0;
    size_t candidateSelected = 0;
    std::map<int64_t, size_t> selectedPerM;
    std::map<int64_t, size_t> seenPerM;
    std::vector<Workload> eligibleForM;
    int64_t currentM = 0;
    size_t cellsMet = 0;
    size_t cellsMissing = 0;
    auto nBucket = [](int64_t n) -> size_t {
        return n <= 64 ? 0 : (n <= 160 ? 1 : 2);
    };
    auto dtypeBucket = [](const std::string &dtype) -> size_t {
        return dtype == "bf16" || dtype == "bf16_bf16" ? 1 : 0;
    };
    auto flushStratified = [&]() {
        if (!stratifiedMode || eligibleForM.empty()) {
            eligibleForM.clear();
            return;
        }
        std::set<int64_t> kValues[2][3];
        for (const Workload &workload : eligibleForM) {
            kValues[dtypeBucket(workload.dtype)][nBucket(workload.n)].insert(workload.k);
        }
        std::vector<int64_t> orderedK[2][3];
        for (size_t dtype = 0; dtype < 2; ++dtype) {
            for (size_t n = 0; n < 3; ++n) {
                orderedK[dtype][n].assign(kValues[dtype][n].begin(), kValues[dtype][n].end());
            }
        }
        size_t selected[18] = {};
        for (const Workload &workload : eligibleForM) {
            const size_t dtype = dtypeBucket(workload.dtype);
            const size_t n = nBucket(workload.n);
            const auto &values = orderedK[dtype][n];
            const auto position = std::lower_bound(values.begin(), values.end(), workload.k);
            const size_t rank = static_cast<size_t>(position - values.begin());
            const size_t k = std::min<size_t>(2, rank * 3 / values.size());
            const size_t cell = dtype * 9 + n * 3 + k;
            if (selected[cell] >= perMReserve) {
                continue;
            }
            ++selected[cell];
            std::cout << workload.dtype << '\t' << workload.layout << '\t' << workload.m << '\t'
                      << workload.n << '\t' << workload.k << '\t' << cell << '\n';
        }
        for (size_t cell = 0; cell < 18; ++cell) {
            if (selected[cell] >= perMReserve) {
                ++cellsMet;
            } else {
                ++cellsMissing;
            }
        }
        eligibleForM.clear();
    };
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        Workload workload;
        if (!ParseLine(line, workload)) {
            ++invalid;
            continue;
        }
        if (stratifiedMode && currentM != 0 && workload.m != currentM) {
            flushStratified();
        }
        currentM = workload.m;
        ++seenPerM[workload.m];
        if (selectMode && !stratifiedMode && selectedPerM[workload.m] >= perMReserve) {
            continue;
        }
        (void)unsetenv("MATMUL_BASE_MODE");
        (void)unsetenv("MATMUL_BASE_FULL_M_SWEEP");
        (void)unsetenv("MATMUL_BASE_EXPERIMENT_SELECTED");
        (void)unsetenv("MATMUL_EXPERIMENT_BRANCH");
        const RouteResult official = RunOne(impl, workload, !selectMode);
        if (official.rc != ge::GRAPH_SUCCESS) {
            ++failed;
            continue;
        }
        if (!selectMode) {
            continue;
        }
        if (std::strcmp(Family(official.key), "BASE") != 0) {
            continue;
        }
        ++officialBase;
        (void)setenv("MATMUL_BASE_MODE", campaign, 1);
        (void)setenv("MATMUL_BASE_FULL_M_SWEEP", "1", 1);
        (void)unsetenv("MATMUL_BASE_EXPERIMENT_SELECTED");
        (void)unsetenv("MATMUL_EXPERIMENT_BRANCH");
        const RouteResult candidate = RunOne(impl, workload, false);
        const char *branch = std::getenv("MATMUL_EXPERIMENT_BRANCH");
        if (candidate.rc != ge::GRAPH_SUCCESS || branch == nullptr || std::strcmp(branch, campaign) != 0) {
            continue;
        }
        ++candidateSelected;
        if (stratifiedMode) {
            eligibleForM.push_back(workload);
            continue;
        }
        ++selectedPerM[workload.m];
        std::cout << workload.dtype << '\t' << workload.layout << '\t' << workload.m << '\t'
                  << workload.n << '\t' << workload.k << '\n';
    }
    flushStratified();
    if (selectMode) {
        size_t met = 0;
        size_t missing = 0;
        for (const auto &entry : seenPerM) {
            if (selectedPerM[entry.first] >= perMReserve) {
                ++met;
            } else {
                ++missing;
            }
        }
        if (stratifiedMode) {
            std::cerr << "# host_selection campaign=" << campaign << " m_seen=" << seenPerM.size()
                      << " strata=18 reserve_per_stratum=" << perMReserve
                      << " strata_reserve_met=" << cellsMet << " strata_reserve_missing=" << cellsMissing
                      << " official_base=" << officialBase << " candidate_selected=" << candidateSelected << '\n';
        } else {
            std::cerr << "# host_selection campaign=" << campaign << " m_seen=" << seenPerM.size()
                      << " reserve_per_m=" << perMReserve << " m_reserve_met=" << met
                      << " m_reserve_missing=" << missing << " official_base=" << officialBase
                      << " candidate_selected=" << candidateSelected << '\n';
        }
    }
    return invalid == 0 && (selectMode || failed == 0) ? 0 : 1;
}
