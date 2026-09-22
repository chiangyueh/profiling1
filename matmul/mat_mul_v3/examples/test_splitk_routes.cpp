// NEW BEGIN
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h"
#include "opdev/common_types.h"

extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char *soPath);

constexpr int WARMUP = 2;
constexpr int REPEATS = 12;

struct Tensor {
    aclTensor *tensor = nullptr;
    void *device = nullptr;
};

struct Candidate {
    const char *name;
    const char *binaryEnv;
    const char *functionName;
    uint64_t key;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
};

void ReleaseTensor(Tensor &value)
{
    if (value.tensor != nullptr) (void)aclDestroyTensor(value.tensor);
    if (value.device != nullptr) (void)aclrtFree(value.device);
}

int CreateTensor(const std::vector<float> &host, const std::vector<int64_t> &shape,
                 const std::vector<int64_t> &storage, const std::vector<int64_t> &strides, Tensor &value)
{
    const size_t bytes = host.size() * sizeof(float);
    int rc = aclrtMalloc(&value.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc == ACL_SUCCESS) {
        rc = aclrtMemcpy(value.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (rc == ACL_SUCCESS) {
        value.tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                                       ACL_FORMAT_ND, storage.data(), storage.size(), value.device);
        if (value.tensor == nullptr) rc = 1;
    }
    return rc;
}

template <typename Launch>
int Measure(Launch launch, aclrtStream stream, float &latency)
{
    int rc = ACL_SUCCESS;
    for (int index = 0; rc == ACL_SUCCESS && index < WARMUP; ++index) rc = launch();
    if (rc == ACL_SUCCESS) rc = aclrtSynchronizeStream(stream);
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    if (rc == ACL_SUCCESS) rc = aclrtCreateEvent(&start);
    if (rc == ACL_SUCCESS) rc = aclrtCreateEvent(&end);
    if (rc == ACL_SUCCESS) rc = aclrtRecordEvent(start, stream);
    for (int index = 0; rc == ACL_SUCCESS && index < REPEATS; ++index) rc = launch();
    if (rc == ACL_SUCCESS) rc = aclrtRecordEvent(end, stream);
    if (rc == ACL_SUCCESS) rc = aclrtSynchronizeEvent(end);
    float total = 0.0f;
    if (rc == ACL_SUCCESS) rc = aclrtEventElapsedTime(&total, start, end);
    if (end != nullptr) (void)aclrtDestroyEvent(end);
    if (start != nullptr) (void)aclrtDestroyEvent(start);
    if (rc == ACL_SUCCESS) latency = total / REPEATS;
    return rc;
}

std::vector<uint8_t> DecodeHex(const char *text)
{
    if (text == nullptr || (std::strlen(text) & 1U) != 0U) return {};
    const size_t size = std::strlen(text) / 2;
    std::vector<uint8_t> bytes(size);
    for (size_t index = 0; index < size; ++index) {
        auto decode = [](char value) -> int {
            if (value >= '0' && value <= '9') return value - '0';
            if (value >= 'a' && value <= 'f') return value - 'a' + 10;
            if (value >= 'A' && value <= 'F') return value - 'A' + 10;
            return -1;
        };
        const int high = decode(text[index * 2]);
        const int low = decode(text[index * 2 + 1]);
        if (high < 0 || low < 0) return {};
        bytes[index] = static_cast<uint8_t>((high << 4) | low);
    }
    return bytes;
}

const char *BranchFromKey(uint64_t key)
{
    const uint64_t load = key & 0x0fU;
    const uint64_t split = (key >> 4U) & 0xffU;
    if (split == 2) return "SINGLE_CORE_SPLIT_K";
    if (split == 3) return "DETERMINISTIC_SPLIT_K";
    if (split == 4) return "MULTI_CORE_SPLIT_K";
    if (split == 5) return "SINGLE_CORE_NKM_SPLIT_K";
    if (split == 6) return "SINGLE_CORE_SPLIT_K_GM_TO_L1";
    if (load == 1) return "AL1_FULL_LOAD";
    if (load == 2) return "BL1_FULL_LOAD";
    return "BASE";
}

bool Validate(void *output, int64_t m, int64_t n, int64_t k)
{
    std::vector<float> host(static_cast<size_t>(m * n));
    if (aclrtMemcpy(host.data(), host.size() * sizeof(float), output,
                    host.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) return false;
    const int64_t periods = k / 4;
    const float dot = static_cast<float>(periods * 9);
    for (int64_t row = 0; row < m; ++row) {
        const float rowScale = static_cast<float>(row % 3 + 1);
        for (int64_t column = 0; column < n; ++column) {
            const float expected = rowScale * static_cast<float>(column % 5 + 1) * dot;
            const float actual = host[static_cast<size_t>(row * n + column)];
            if (!std::isfinite(actual) || std::fabs(actual - expected) > std::max(1.0f, expected * 1.0e-5f)) {
                return false;
            }
        }
    }
    return true;
}

int DirectLaunch(const Candidate &candidate, uint32_t cores, const std::vector<uint8_t> &tiling,
                 void *a, void *b, void *c, void *workspace, aclrtStream stream, float &latency)
{
    void *tilingDevice = nullptr;
    int rc = aclrtMalloc(&tilingDevice, tiling.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc == ACL_SUCCESS) {
        rc = aclrtMemcpy(tilingDevice, tiling.size(), tiling.data(), tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    aclrtArgsHandle arguments = nullptr;
    if (rc == ACL_SUCCESS) rc = aclrtKernelArgsInit(candidate.function, &arguments);
    void *bias = nullptr;
    void *offset = nullptr;
    void *values[] = {a, b, bias, offset, c, workspace, tilingDevice};
    for (void *&value : values) {
        aclrtParamHandle parameter = nullptr;
        if (rc == ACL_SUCCESS) rc = aclrtKernelArgsAppend(arguments, &value, sizeof(value), &parameter);
    }
    if (rc == ACL_SUCCESS) rc = aclrtKernelArgsFinalize(arguments);
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclrtLaunchKernelWithConfig(candidate.function, cores, stream, nullptr, arguments, nullptr);
        }, stream, latency);
    }
    if (tilingDevice != nullptr) (void)aclrtFree(tilingDevice);
    return rc;
}

int RunShape(int64_t m, int64_t n, int64_t k, aclrtStream stream, const std::vector<Candidate> &candidates)
{
    std::vector<float> a(static_cast<size_t>(m * k));
    std::vector<float> b(static_cast<size_t>(n * k));
    std::vector<float> c(static_cast<size_t>(m * n), 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t index = 0; index < k; ++index) {
            a[static_cast<size_t>(row * k + index)] =
                static_cast<float>((row % 3 + 1) * (index % 2 + 1));
        }
    }
    for (int64_t column = 0; column < n; ++column) {
        for (int64_t index = 0; index < k; ++index) {
            b[static_cast<size_t>(column * k + index)] =
                static_cast<float>((column % 5 + 1) * ((index / 2) % 2 + 1));
        }
    }
    Tensor aTensor;
    Tensor bTensor;
    Tensor cTensor;
    int rc = CreateTensor(a, {m, k}, {m, k}, {k, 1}, aTensor);
    if (rc == ACL_SUCCESS) rc = CreateTensor(b, {k, n}, {n, k}, {1, k}, bTensor);
    if (rc == ACL_SUCCESS) rc = CreateTensor(c, {m, n}, {m, n}, {n, 1}, cTensor);
    if (rc != ACL_SUCCESS) {
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc;
    }

    (void)::unsetenv("MATMUL_SPLITK_MODE");
    (void)::unsetenv("MATMUL_EXPERIMENT_SELECTED");
    (void)::setenv("MATMUL_VECTOR_ENABLE", "0", 1);
    (void)::unsetenv("MATMUL_OBSERVED_KEY");
    (void)::unsetenv("MATMUL_OBSERVED_CORES");
    uint64_t officialWorkspaceSize = 0;
    aclOpExecutor *officialExecutor = nullptr;
    rc = aclnnMatmulGetWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, 1,
                                     &officialWorkspaceSize, &officialExecutor);
    void *officialWorkspace = nullptr;
    if (rc == ACL_SUCCESS && officialWorkspaceSize != 0) {
        rc = aclrtMalloc(&officialWorkspace, officialWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (rc == ACL_SUCCESS) rc = aclSetAclOpExecutorRepeatable(officialExecutor);
    float officialLatency = 0.0f;
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclnnMatmul(officialWorkspace, officialWorkspaceSize, officialExecutor, stream);
        }, stream, officialLatency);
    }
    const bool officialCorrect = rc == ACL_SUCCESS && Validate(cTensor.device, m, n, k);
    const uint64_t officialKey = std::strtoull(std::getenv("MATMUL_OBSERVED_KEY") == nullptr ? "0" :
                                               std::getenv("MATMUL_OBSERVED_KEY"), nullptr, 10);
    const uint32_t officialCore = static_cast<uint32_t>(std::strtoul(
        std::getenv("MATMUL_OBSERVED_CORES") == nullptr ? "0" : std::getenv("MATMUL_OBSERVED_CORES"), nullptr, 10));
    if (officialExecutor != nullptr) (void)aclDestroyAclOpExecutor(officialExecutor);
    if (officialWorkspace != nullptr) (void)aclrtFree(officialWorkspace);
    if (!officialCorrect) {
        std::printf(
            "{\"shape\":\"M%ld_N%ld_K%ld_NT\",\"official_branch\":\"%s\","
            "\"official_core\":%u,\"official_latency_ms\":null,"
            "\"candidate_branch\":\"ADAPTIVE_DETERMINISTIC_SPLIT_K\",\"candidate_core\":null,"
            "\"split_factor_min\":null,\"split_factor_max\":null,\"candidate_latency_ms\":null,"
            "\"delta_pct\":null,\"correctness\":\"NOT_CHECKED\",\"result_code\":%d,"
            "\"failure_stage\":\"official_measurement\"}\n",
            static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), BranchFromKey(officialKey),
            officialCore, rc == ACL_SUCCESS ? 3 : rc);
        std::fflush(stdout);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc == ACL_SUCCESS ? 3 : rc;
    }

    for (const Candidate &candidate : candidates) {
        (void)::unsetenv("MATMUL_EXPERIMENT_TILING");
        (void)::unsetenv("MATMUL_EXPERIMENT_CORES");
        (void)::unsetenv("MATMUL_EXPERIMENT_KEY");
        (void)::unsetenv("MATMUL_EXPERIMENT_BRANCH");
        (void)::unsetenv("MATMUL_EXPERIMENT_SELECTED");
        (void)::setenv("MATMUL_SPLITK_MODE", candidate.name, 1);
        uint64_t candidateWorkspaceSize = 0;
        aclOpExecutor *candidateExecutor = nullptr;
        rc = aclnnMatmulGetWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, 1,
                                         &candidateWorkspaceSize, &candidateExecutor);
        const char *tilingText = std::getenv("MATMUL_EXPERIMENT_TILING");
        const char *coreText = std::getenv("MATMUL_EXPERIMENT_CORES");
        const char *keyText = std::getenv("MATMUL_EXPERIMENT_KEY");
        if (rc != ACL_SUCCESS || tilingText == nullptr || coreText == nullptr || keyText == nullptr) {
            std::printf(
                "{\"shape\":\"M%ld_N%ld_K%ld_NT\",\"official_branch\":\"%s\","
                "\"official_core\":%u,\"official_latency_ms\":%.9f,\"candidate_branch\":\"%s\","
                "\"candidate_core\":null,\"split_factor_min\":null,\"split_factor_max\":null,"
                "\"candidate_latency_ms\":null,\"delta_pct\":null,\"correctness\":\"NOT_CHECKED\","
                "\"result_code\":%d,\"failure_stage\":\"candidate_tiling\"}\n",
                static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), BranchFromKey(officialKey),
                officialCore, officialLatency, candidate.name, rc == ACL_SUCCESS ? 4 : rc);
            std::fflush(stdout);
            if (candidateExecutor != nullptr) (void)aclDestroyAclOpExecutor(candidateExecutor);
            continue;
        }
        const std::vector<uint8_t> tiling = DecodeHex(tilingText);
        const uint32_t cores = static_cast<uint32_t>(std::strtoul(coreText, nullptr, 10));
        const uint64_t key = std::strtoull(keyText, nullptr, 10);
        if (tiling.empty() || cores == 0 || key != candidate.key) {
            std::printf(
                "{\"shape\":\"M%ld_N%ld_K%ld_NT\",\"official_branch\":\"%s\","
                "\"official_core\":%u,\"official_latency_ms\":%.9f,\"candidate_branch\":\"%s\","
                "\"candidate_core\":%u,\"split_factor_min\":null,\"split_factor_max\":null,"
                "\"candidate_latency_ms\":null,\"delta_pct\":null,\"correctness\":\"NOT_CHECKED\","
                "\"result_code\":4,\"failure_stage\":\"candidate_packet\"}\n",
                static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), BranchFromKey(officialKey),
                officialCore, officialLatency, candidate.name, cores);
            std::fflush(stdout);
            if (candidateExecutor != nullptr) (void)aclDestroyAclOpExecutor(candidateExecutor);
            continue;
        }
        void *candidateWorkspace = nullptr;
        if (candidateWorkspaceSize != 0) {
            rc = aclrtMalloc(&candidateWorkspace, candidateWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (rc == ACL_SUCCESS) {
            rc = aclrtMemset(cTensor.device, static_cast<size_t>(m * n * sizeof(float)), 0,
                             static_cast<size_t>(m * n * sizeof(float)));
        }
        float candidateLatency = 0.0f;
        if (rc == ACL_SUCCESS && candidate.function != nullptr) {
            rc = DirectLaunch(candidate, cores, tiling, aTensor.device, bTensor.device, cTensor.device,
                              candidateWorkspace, stream, candidateLatency);
        } else if (rc == ACL_SUCCESS) {
            rc = aclSetAclOpExecutorRepeatable(candidateExecutor);
            if (rc == ACL_SUCCESS) {
                rc = Measure([&]() {
                    return aclnnMatmul(candidateWorkspace, candidateWorkspaceSize, candidateExecutor, stream);
                }, stream, candidateLatency);
            }
        }
        const bool correct = rc == ACL_SUCCESS && Validate(cTensor.device, m, n, k);
        const double delta = correct ? (candidateLatency / officialLatency - 1.0) * 100.0 : 0.0;
        uint64_t splitMin = cores;
        uint64_t splitMax = cores;
        if (std::strcmp(candidate.name, "TAIL_STREAM_K") == 0) {
            const uint64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
            const uint64_t tail = cores == 0 ? 0 : tiles % cores;
            splitMin = tail == 0 ? cores : cores / tail;
            splitMax = tail == 0 ? cores : (cores + tail - 1) / tail;
        }
        std::printf(
            "{\"shape\":\"M%ld_N%ld_K%ld_NT\",\"official_branch\":\"%s\","
            "\"official_core\":%u,\"official_latency_ms\":%.9f,\"candidate_branch\":\"%s\","
            "\"candidate_core\":%u,\"split_factor_min\":%lu,\"split_factor_max\":%lu,"
            "\"candidate_latency_ms\":%s,\"delta_pct\":%s,\"correctness\":\"%s\","
            "\"result_code\":%d}\n",
            static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), BranchFromKey(officialKey),
            officialCore, officialLatency, candidate.name, cores,
            static_cast<unsigned long>(splitMin), static_cast<unsigned long>(splitMax),
            correct ? std::to_string(candidateLatency).c_str() : "null",
            correct ? std::to_string(delta).c_str() : "null", correct ? "PASS" : "FAIL", rc);
        std::fflush(stdout);
        if (candidateExecutor != nullptr) (void)aclDestroyAclOpExecutor(candidateExecutor);
        if (candidateWorkspace != nullptr) (void)aclrtFree(candidateWorkspace);
    }
    (void)::unsetenv("MATMUL_SPLITK_MODE");
    ReleaseTensor(cTensor);
    ReleaseTensor(bTensor);
    ReleaseTensor(aTensor);
    return ACL_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc < 4 || (argc - 1) % 3 != 0) return 2;
    const char *hostLibrary = std::getenv("MATMUL_HOST_LIBRARY");
    if (hostLibrary == nullptr) return 4;
    int rc = aclInit(nullptr);
    if (rc == ACL_SUCCESS) rc = aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    if (rc == ACL_SUCCESS) rc = aclrtCreateStream(&stream);
    if (rc != ACL_SUCCESS || TbeLoadSoAndSaveToRegistry(hostLibrary) != 0U) return 4;
    std::vector<Candidate> candidates = {
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", "MATMUL_ADAPTIVE_BINARY", "MatMulV3_Adaptive_65648", 65648},
    };
    for (Candidate &candidate : candidates) {
        const char *path = std::getenv(candidate.binaryEnv);
        if (path == nullptr || aclrtBinaryLoadFromFile(path, nullptr, &candidate.binary) != ACL_SUCCESS ||
            aclrtBinaryGetFunction(candidate.binary, candidate.functionName, &candidate.function) != ACL_SUCCESS) {
            return 4;
        }
    }
    for (int index = 1; index < argc; index += 3) {
        (void)RunShape(std::strtoll(argv[index], nullptr, 10),
                       std::strtoll(argv[index + 1], nullptr, 10),
                       std::strtoll(argv[index + 2], nullptr, 10), stream, candidates);
    }
    for (Candidate &candidate : candidates) {
        if (candidate.binary != nullptr) (void)aclrtBinaryUnLoad(candidate.binary);
    }
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return 0;
}
// NEW END
