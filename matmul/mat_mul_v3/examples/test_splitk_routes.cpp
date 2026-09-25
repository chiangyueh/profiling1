// NEW BEGIN
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h"

extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char *soPath);

constexpr int WARMUP = 1;
constexpr int SAMPLES = 3;
constexpr int RUNS_PER_SAMPLE = 2;

struct ScalarType {
    aclDataType aclType;
    size_t bytes;
};

struct DTypeSpec {
    const char *name;
    const char *inputName;
    const char *outputName;
    ScalarType input;
    ScalarType output;
};

struct LayoutSpec {
    const char *name;
    bool transA;
    bool transB;
};

struct Tensor {
    aclTensor *tensor = nullptr;
    void *device = nullptr;
    size_t bytes = 0;
};

struct TilingSnapshot {
    uint64_t key = 0;
    uint32_t cores = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t singleK = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t stepM = 0;
    uint32_t stepN = 0;
    uint32_t stepKa = 0;
    uint32_t stepKb = 0;
    uint32_t depthA1 = 0;
    uint32_t depthB1 = 0;
    uint32_t l2MTile = 0;
    uint32_t l2NTile = 0;
    uint32_t l2MBlock = 0;
    uint32_t l2NBlock = 0;
    uint32_t l2Order = 0;
};

struct RunCounts {
    uint64_t inputs = 0;
    uint64_t nonDeterministic = 0;
    uint64_t deterministic = 0;
    uint64_t adaptiveSelected = 0;
    uint64_t officialPreserved = 0;
    uint64_t passed = 0;
    uint64_t failed = 0;
    uint64_t officialFailed = 0;
    uint64_t skippedNonV3 = 0;
    uint64_t wideNShallowKPassed = 0;
    uint64_t clearCandidateWins = 0;
    uint64_t clearOfficialWins = 0;
    uint64_t overlap = 0;
    uint64_t stableCandidateWins = 0;
    uint64_t stableOfficialWins = 0;
    uint64_t mixedOrder = 0;
    uint64_t mCoverage[5] = {};
    uint64_t nCoverage[5] = {};
    uint64_t kCoverage[5] = {};
    uint64_t jointInputs[200] = {};
    uint64_t jointOfficial[200] = {};
    uint64_t jointPassed[200] = {};
    uint64_t jointQuotaSkipped = 0;
};

uint64_t ReadEnvUnsigned(const char *name)
{
    const char *text = std::getenv(name);
    return text == nullptr ? 0 : std::strtoull(text, nullptr, 10);
}

void CountPassCoverage(RunCounts &counts, int64_t m, int64_t n, int64_t k)
{
    const size_t mBucket = m <= 9 ? 0 : (m <= 31 ? 1 : (m <= 63 ? 2 : (m <= 95 ? 3 : 4)));
    const size_t nBucket = n <= 24576 ? 0 : (n <= 26624 ? 1 : (n <= 28160 ? 2 : (n <= 30464 ? 3 : 4)));
    const size_t kBucket = k <= 8192 ? 0 : (k <= 15872 ? 1 : (k <= 27264 ? 2 : 3));
    ++counts.mCoverage[mBucket];
    ++counts.nCoverage[nBucket];
    ++counts.kCoverage[kBucket];
}

size_t JointCoverageBucket(const DTypeSpec &dtype, int64_t m, int64_t n, int64_t k)
{
    const size_t dtypeBucket = std::strcmp(dtype.inputName, "bf16") == 0 ? 1 : 0;
    const size_t mBucket = m <= 9 ? 0 : (m <= 31 ? 1 : (m <= 63 ? 2 : (m <= 95 ? 3 : 4)));
    const size_t nBucket = n <= 24576 ? 0 : (n <= 26624 ? 1 : (n <= 28160 ? 2 : (n <= 30464 ? 3 : 4)));
    const size_t kBucket = k <= 8192 ? 0 : (k <= 15872 ? 1 : (k <= 27264 ? 2 : 3));
    return (((dtypeBucket * 5 + mBucket) * 5 + nBucket) * 4 + kBucket);
}

TilingSnapshot ReadTilingSnapshot()
{
    TilingSnapshot value;
    value.key = ReadEnvUnsigned("MATMUL_OBSERVED_KEY");
    value.cores = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_CORES"));
    value.singleM = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_SINGLE_M"));
    value.singleN = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_SINGLE_N"));
    value.singleK = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_SINGLE_K"));
    value.baseM = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_BASE_M"));
    value.baseN = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_BASE_N"));
    value.baseK = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_BASE_K"));
    value.stepM = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_STEP_M"));
    value.stepN = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_STEP_N"));
    value.stepKa = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_STEP_KA"));
    value.stepKb = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_STEP_KB"));
    value.depthA1 = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_DEPTH_A1"));
    value.depthB1 = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_DEPTH_B1"));
    value.l2MTile = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_L2_M_TILE"));
    value.l2NTile = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_L2_N_TILE"));
    value.l2MBlock = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_L2_M_BLOCK"));
    value.l2NBlock = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_L2_N_BLOCK"));
    value.l2Order = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_L2_ORDER"));
    return value;
}

void ClearObservedTiling()
{
    const char *names[] = {
        "MATMUL_OBSERVED_KEY", "MATMUL_OBSERVED_CORES", "MATMUL_OBSERVED_SINGLE_M",
        "MATMUL_OBSERVED_SINGLE_N", "MATMUL_OBSERVED_SINGLE_K", "MATMUL_OBSERVED_BASE_M",
        "MATMUL_OBSERVED_BASE_N", "MATMUL_OBSERVED_BASE_K", "MATMUL_OBSERVED_STEP_M",
        "MATMUL_OBSERVED_STEP_N", "MATMUL_OBSERVED_STEP_KA",
        "MATMUL_OBSERVED_STEP_KB", "MATMUL_OBSERVED_DEPTH_A1", "MATMUL_OBSERVED_DEPTH_B1",
        "MATMUL_OBSERVED_L2_M_TILE", "MATMUL_OBSERVED_L2_N_TILE", "MATMUL_OBSERVED_L2_M_BLOCK",
        "MATMUL_OBSERVED_L2_N_BLOCK", "MATMUL_OBSERVED_L2_ORDER"
    };
    for (const char *name : names) (void)::unsetenv(name);
}

std::vector<uint8_t> DecodeHex(const char *text)
{
    if (text == nullptr || (std::strlen(text) & 1U) != 0U) return {};
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    std::vector<uint8_t> bytes(std::strlen(text) / 2);
    for (size_t index = 0; index < bytes.size(); ++index) {
        const int high = nibble(text[index * 2]);
        const int low = nibble(text[index * 2 + 1]);
        if (high < 0 || low < 0) return {};
        bytes[index] = static_cast<uint8_t>((high << 4) | low);
    }
    return bytes;
}

bool IsDeterministicSplitK(uint64_t key)
{
    return ((key >> 4U) & 0xffU) == 3U;
}

bool IsPlainBase(uint64_t key)
{
    return (key & 0x0fU) == 0U && ((key >> 4U) & 0xffU) == 0U &&
        ((key >> 12U) & 0x0fU) == 0U && ((key >> 16U) & 0x0fU) == 1U &&
        ((key >> 20U) & 0x0fU) == 0U;
}

bool SameTilingExceptSingleM(const TilingSnapshot &left, const TilingSnapshot &right)
{
    return left.key == right.key && left.cores == right.cores && left.singleN == right.singleN &&
        left.singleK == right.singleK && left.baseM == right.baseM && left.baseN == right.baseN &&
        left.baseK == right.baseK && left.stepKa == right.stepKa && left.stepKb == right.stepKb &&
        left.depthA1 == right.depthA1 && left.depthB1 == right.depthB1 &&
        left.l2MTile == right.l2MTile && left.l2NTile == right.l2NTile &&
        left.l2MBlock == right.l2MBlock && left.l2NBlock == right.l2NBlock &&
        left.l2Order == right.l2Order;
}

bool IsRectangularCampaign()
{
    const char *campaign = std::getenv("MATMUL_CAMPAIGN");
    return campaign != nullptr && std::strcmp(campaign, "RECTANGULAR_CUBE") == 0;
}

bool IsAdaptiveCampaign()
{
    const char *campaign = std::getenv("MATMUL_CAMPAIGN");
    return campaign != nullptr && std::strcmp(campaign, "ADAPTIVE_DETERMINISTIC_SPLIT_K") == 0;
}

bool IsBaseCampaign()
{
    const char *campaign = std::getenv("MATMUL_CAMPAIGN");
    return campaign != nullptr &&
        (std::strcmp(campaign, "RECTANGULAR_CUBE") == 0 ||
         std::strcmp(campaign, "REUSE_DIRECTED") == 0 ||
         std::strcmp(campaign, "CUBE_VECTOR_EDGE") == 0 ||
         std::strcmp(campaign, "WIDE_N_PANEL_ONLY") == 0 ||
         std::strcmp(campaign, "WIDE_N_WINDOW_ONLY") == 0 ||
         std::strcmp(campaign, "WIDE_N_SHALLOW_K_BASE") == 0);
}

const char *CampaignName()
{
    if (IsRectangularCampaign()) return "RECTANGULAR_CUBE";
    const char *campaign = std::getenv("MATMUL_CAMPAIGN");
    if (campaign != nullptr && std::strcmp(campaign, "REUSE_DIRECTED") == 0) return "REUSE_DIRECTED";
    if (campaign != nullptr && std::strcmp(campaign, "CUBE_VECTOR_EDGE") == 0) return "CUBE_VECTOR_EDGE";
    if (campaign != nullptr && std::strcmp(campaign, "WIDE_N_PANEL_ONLY") == 0) {
        return "WIDE_N_PANEL_ONLY";
    }
    if (campaign != nullptr && std::strcmp(campaign, "WIDE_N_WINDOW_ONLY") == 0) {
        return "WIDE_N_WINDOW_ONLY";
    }
    if (campaign != nullptr && std::strcmp(campaign, "WIDE_N_SHALLOW_K_BASE") == 0) {
        return "WIDE_N_SHALLOW_K_BASE";
    }
    if (IsAdaptiveCampaign()) return "ADAPTIVE_DETERMINISTIC_SPLIT_K";
    return "INVALID";
}

const DTypeSpec *FindDType(const char *name)
{
    static const DTypeSpec values[] = {
        {"fp16_fp16", "fp16", "fp16", {ACL_FLOAT16, sizeof(uint16_t)}, {ACL_FLOAT16, sizeof(uint16_t)}},
        {"fp16_fp32", "fp16", "fp32", {ACL_FLOAT16, sizeof(uint16_t)}, {ACL_FLOAT, sizeof(float)}},
        {"bf16_bf16", "bf16", "bf16", {ACL_BF16, sizeof(uint16_t)}, {ACL_BF16, sizeof(uint16_t)}},
        {"bf16_fp32", "bf16", "fp32", {ACL_BF16, sizeof(uint16_t)}, {ACL_FLOAT, sizeof(float)}},
        {"fp32_fp32", "fp32", "fp32", {ACL_FLOAT, sizeof(float)}, {ACL_FLOAT, sizeof(float)}},
    };
    for (const DTypeSpec &value : values) {
        if (std::strcmp(value.name, name) == 0) return &value;
    }
    return nullptr;
}

const LayoutSpec *FindLayout(const char *name)
{
    static const LayoutSpec values[] = {
        {"NN", false, false}, {"NT", false, true},
        {"TN", true, false}, {"TT", true, true},
    };
    for (const LayoutSpec &value : values) {
        if (std::strcmp(value.name, name) == 0) return &value;
    }
    return nullptr;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void StoreValue(std::vector<uint8_t> &data, size_t index, const ScalarType &dtype, float value)
{
    uint8_t *destination = data.data() + index * dtype.bytes;
    if (dtype.aclType == ACL_FLOAT) {
        std::memcpy(destination, &value, sizeof(value));
    } else if (dtype.aclType == ACL_FLOAT16) {
        const aclFloat16 converted = aclFloatToFloat16(value);
        std::memcpy(destination, &converted, sizeof(converted));
    } else {
        const uint16_t converted = FloatToBf16(value);
        std::memcpy(destination, &converted, sizeof(converted));
    }
}

void FillScalarRange(std::vector<uint8_t> &data, size_t first, size_t count,
                     const ScalarType &dtype, float value)
{
    if (count == 0) return;
    StoreValue(data, first, dtype, value);
    size_t filled = dtype.bytes;
    const size_t begin = first * dtype.bytes;
    const size_t total = count * dtype.bytes;
    while (filled < total) {
        const size_t copied = std::min(filled, total - filled);
        std::memcpy(data.data() + begin + filled, data.data() + begin, copied);
        filled += copied;
    }
}

void RepeatPrefix(std::vector<uint8_t> &data, size_t prefixBytes)
{
    size_t filled = prefixBytes;
    while (filled < data.size()) {
        const size_t copied = std::min(filled, data.size() - filled);
        std::memcpy(data.data() + filled, data.data(), copied);
        filled += copied;
    }
}

float LoadValue(const std::vector<uint8_t> &data, size_t index, const ScalarType &dtype)
{
    const uint8_t *source = data.data() + index * dtype.bytes;
    if (dtype.aclType == ACL_FLOAT) {
        float value = 0.0f;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }
    uint16_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    if (dtype.aclType == ACL_FLOAT16) return aclFloat16ToFloat(static_cast<aclFloat16>(value));
    return Bf16ToFloat(value);
}

void ReleaseTensor(Tensor &value)
{
    if (value.tensor != nullptr) (void)aclDestroyTensor(value.tensor);
    if (value.device != nullptr) (void)aclrtFree(value.device);
    value = {};
}

int CreateEmptyTensor(size_t bytes, const std::vector<int64_t> &shape,
                      const std::vector<int64_t> &storage, const std::vector<int64_t> &strides,
                      const ScalarType &dtype, Tensor &value)
{
    value.bytes = bytes;
    int rc = aclrtMalloc(&value.device, value.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc == ACL_SUCCESS) {
        value.tensor = aclCreateTensor(shape.data(), shape.size(), dtype.aclType, strides.data(), 0,
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
    std::vector<float> samples;
    for (int sample = 0; rc == ACL_SUCCESS && sample < SAMPLES; ++sample) {
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        rc = aclrtCreateEvent(&start);
        if (rc == ACL_SUCCESS) rc = aclrtCreateEvent(&end);
        if (rc == ACL_SUCCESS) rc = aclrtRecordEvent(start, stream);
        for (int index = 0; rc == ACL_SUCCESS && index < RUNS_PER_SAMPLE; ++index) rc = launch();
        if (rc == ACL_SUCCESS) rc = aclrtRecordEvent(end, stream);
        if (rc == ACL_SUCCESS) rc = aclrtSynchronizeEvent(end);
        float total = 0.0f;
        if (rc == ACL_SUCCESS) rc = aclrtEventElapsedTime(&total, start, end);
        if (end != nullptr) (void)aclrtDestroyEvent(end);
        if (start != nullptr) (void)aclrtDestroyEvent(start);
        if (rc == ACL_SUCCESS) samples.push_back(total / RUNS_PER_SAMPLE);
    }
    if (rc == ACL_SUCCESS) {
        std::sort(samples.begin(), samples.end());
        latency = samples[samples.size() / 2];
    }
    return rc;
}

std::vector<uint8_t> CopyDeviceOutput(const Tensor &tensor)
{
    std::vector<uint8_t> host(tensor.bytes);
    if (aclrtMemcpy(host.data(), host.size(), tensor.device, tensor.bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) return {};
    return host;
}

bool CompareOutputs(const std::vector<uint8_t> &reference, const std::vector<uint8_t> &candidate,
                    const ScalarType &dtype, double &maxAbs, double &maxRel)
{
    if (reference.size() != candidate.size() || reference.empty()) return false;
    const size_t count = reference.size() / dtype.bytes;
    bool correct = true;
    maxAbs = 0.0;
    maxRel = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const float expected = LoadValue(reference, index, dtype);
        const float actual = LoadValue(candidate, index, dtype);
        if (!std::isfinite(expected) || !std::isfinite(actual)) {
            correct = expected == actual && correct;
            continue;
        }
        const double absolute = std::fabs(static_cast<double>(actual) - expected);
        const double relative = absolute / std::max(1.0, std::fabs(static_cast<double>(expected)));
        maxAbs = std::max(maxAbs, absolute);
        maxRel = std::max(maxRel, relative);
        const double tolerance = dtype.aclType == ACL_FLOAT ? 2.0e-3 :
            (dtype.aclType == ACL_FLOAT16 ? 2.0e-2 : 5.0e-2);
        if (relative > tolerance && absolute > 1.0) correct = false;
    }
    return correct;
}

void PrintTiling(const char *name, const TilingSnapshot &value)
{
    std::printf("\"%s\":{\"key\":%lu,\"core\":%u,\"single_m\":%u,\"single_n\":%u,"
                "\"single_k\":%u,\"base_m\":%u,\"base_n\":%u,\"base_k\":%u,"
                "\"step_m\":%u,\"step_n\":%u,\"step_ka\":%u,\"step_kb\":%u,"
                "\"depth_a1\":%u,\"depth_b1\":%u,"
                "\"l2_m_tile\":%u,\"l2_n_tile\":%u,\"l2_m_block\":%u,\"l2_n_block\":%u,"
                "\"l2_order\":%u}",
                name, static_cast<unsigned long>(value.key), value.cores, value.singleM, value.singleN,
                value.singleK, value.baseM, value.baseN, value.baseK, value.stepM, value.stepN,
                value.stepKa, value.stepKb,
                value.depthA1, value.depthB1, value.l2MTile, value.l2NTile, value.l2MBlock,
                value.l2NBlock, value.l2Order);
}

int RunWorkload(const DTypeSpec &dtype, const LayoutSpec &layout, int64_t m, int64_t n, int64_t k,
                aclrtStream stream, aclrtFuncHandle edgeFunction, RunCounts &counts,
                uint64_t manifestIndex)
{
    const bool baseCampaign = IsBaseCampaign();
    const bool edgeCampaign = std::strcmp(CampaignName(), "CUBE_VECTOR_EDGE") == 0;
    const bool adaptiveCampaign = IsAdaptiveCampaign();
    ++counts.inputs;
    Tensor aTensor;
    Tensor bTensor;
    Tensor cTensor;
    int rc = CreateEmptyTensor(static_cast<size_t>(m * k) * dtype.input.bytes, {m, k},
                               layout.transA ? std::vector<int64_t>{k, m} : std::vector<int64_t>{m, k},
                               layout.transA ? std::vector<int64_t>{1, m} : std::vector<int64_t>{k, 1},
                               dtype.input, aTensor);
    if (rc == ACL_SUCCESS) {
        rc = CreateEmptyTensor(static_cast<size_t>(n * k) * dtype.input.bytes, {k, n},
                               layout.transB ? std::vector<int64_t>{n, k} : std::vector<int64_t>{k, n},
                               layout.transB ? std::vector<int64_t>{1, k} : std::vector<int64_t>{n, 1},
                               dtype.input, bTensor);
    }
    if (rc == ACL_SUCCESS) {
        rc = CreateEmptyTensor(static_cast<size_t>(m * n) * dtype.output.bytes,
                               {m, n}, {m, n}, {n, 1}, dtype.output, cTensor);
    }
    if (rc != ACL_SUCCESS) {
        ++counts.officialFailed;
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc;
    }

    (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE");
    (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED");
    (void)::unsetenv("MATMUL_SPLITK_MODE");
    (void)::unsetenv("MATMUL_BASE_MODE");
    (void)::unsetenv("MATMUL_BASE_EXPERIMENT_SELECTED");
    (void)::unsetenv("MATMUL_EXPERIMENT_BRANCH");
    (void)::setenv("MATMUL_VECTOR_ENABLE", "0", 1);
    ClearObservedTiling();
    uint64_t officialWorkspaceSize = 0;
    aclOpExecutor *officialExecutor = nullptr;
    rc = aclnnMatmulGetWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, 1,
                                     &officialWorkspaceSize, &officialExecutor);
    const TilingSnapshot official = ReadTilingSnapshot();
    if (rc != ACL_SUCCESS || officialExecutor == nullptr) {
        ++counts.officialFailed;
        if (rc != 561103) {
            std::printf("{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"input_dtype\":\"%s\","
                        "\"output_dtype\":\"%s\",\"status\":\"OFFICIAL_GET_WORKSPACE_FAILED\",\"result_code\":%d}\n",
                        static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), layout.name,
                        dtype.inputName, dtype.outputName, rc == ACL_SUCCESS ? 4 : rc);
            std::fflush(stdout);
        }
        if (officialExecutor != nullptr) (void)aclDestroyAclOpExecutor(officialExecutor);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc == ACL_SUCCESS ? 4 : rc;
    }
    if (official.key == 0) {
        ++counts.skippedNonV3;
        (void)aclDestroyAclOpExecutor(officialExecutor);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return ACL_SUCCESS;
    }
    const bool officialRouteMatched = baseCampaign ? IsPlainBase(official.key) :
        (adaptiveCampaign ? IsDeterministicSplitK(official.key) : false);
    if (!officialRouteMatched) {
        ++counts.nonDeterministic;
        (void)aclDestroyAclOpExecutor(officialExecutor);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return ACL_SUCCESS;
    }
    ++counts.deterministic;

    if (baseCampaign) {
        (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE");
        (void)::unsetenv("MATMUL_SPLITK_MODE");
        (void)::setenv("MATMUL_VECTOR_ENABLE", "0", 1);
        (void)::setenv("MATMUL_BASE_MODE", CampaignName(), 1);
        (void)::unsetenv("MATMUL_BASE_EXPERIMENT_SELECTED");
        (void)::unsetenv("MATMUL_EXPERIMENT_BRANCH");
    } else if (adaptiveCampaign) {
        (void)::setenv("MATMUL_DETERMINISTIC_ADAPTIVE", "1", 1);
        (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED");
        (void)::setenv("MATMUL_VECTOR_ENABLE", "0", 1);
        (void)::unsetenv("MATMUL_BASE_MODE");
        (void)::unsetenv("MATMUL_SPLITK_MODE");
        (void)::unsetenv("MATMUL_EXPERIMENT_SELECTED");
        (void)::unsetenv("MATMUL_EXPERIMENT_BRANCH");
        const char *modelNames[] = {
            "MATMUL_ADAPTIVE_OLD_PARTIAL_BYTES", "MATMUL_ADAPTIVE_NEW_PARTIAL_BYTES",
            "MATMUL_ADAPTIVE_K_UNITS", "MATMUL_ADAPTIVE_K_QUOTIENT",
            "MATMUL_ADAPTIVE_K_REMAINDER", "MATMUL_ADAPTIVE_K_MIN", "MATMUL_ADAPTIVE_K_MAX"
        };
        for (const char *name : modelNames) (void)::unsetenv(name);
    } else {
        (void)::setenv("MATMUL_DETERMINISTIC_ADAPTIVE", "1", 1);
        (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED");
    }
    ClearObservedTiling();
    uint64_t adaptiveWorkspaceSize = 0;
    aclOpExecutor *adaptiveExecutor = nullptr;
    rc = aclnnMatmulGetWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, 1,
                                     &adaptiveWorkspaceSize, &adaptiveExecutor);
    const TilingSnapshot adaptive = ReadTilingSnapshot();
    const char *variantText = std::getenv("MATMUL_BASE_EXPERIMENT_VARIANT");
    const std::string candidateVariant = variantText == nullptr ? "" : variantText;
    const char *experimentBranch = std::getenv("MATMUL_EXPERIMENT_BRANCH");
    const bool changed = baseCampaign ?
        experimentBranch != nullptr && std::strcmp(experimentBranch, CampaignName()) == 0 :
        ReadEnvUnsigned("MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED") == 1;
    const bool candidateSelected = adaptiveCampaign || changed;
    if (candidateSelected) ++counts.adaptiveSelected;
    if (!changed && rc == ACL_SUCCESS) ++counts.officialPreserved;
    void *adaptiveWorkspace = nullptr;
    if (rc == ACL_SUCCESS && adaptiveExecutor == nullptr) rc = 4;
    if (baseCampaign && !changed) {
        if (rc != ACL_SUCCESS) {
            ++counts.failed;
            std::printf("{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"input_dtype\":\"%s\","
                        "\"output_dtype\":\"%s\",\"candidate_branch\":\"%s\","
                        "\"status\":\"CANDIDATE_TILING_FAILED\",\"result_code\":%d}\n",
                        static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), layout.name,
                        dtype.inputName, dtype.outputName,
                        CampaignName(), rc);
            std::fflush(stdout);
        }
        if (adaptiveExecutor != nullptr) (void)aclDestroyAclOpExecutor(adaptiveExecutor);
        (void)aclDestroyAclOpExecutor(officialExecutor);
        (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE");
        (void)::unsetenv("MATMUL_SPLITK_MODE");
        (void)::unsetenv("MATMUL_BASE_MODE");
        (void)::unsetenv("MATMUL_VECTOR_ENABLE");
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc == ACL_SUCCESS ? ACL_SUCCESS : rc;
    }
    const bool invariantPassed = !adaptiveCampaign || SameTilingExceptSingleM(official, adaptive);
    const bool candidateRouteMatched = baseCampaign ? changed :
        (IsDeterministicSplitK(adaptive.key) && invariantPassed);
    if (rc == ACL_SUCCESS && !candidateRouteMatched) rc = 4;
    if (rc != ACL_SUCCESS) {
        ++counts.failed;
        std::printf("{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"input_dtype\":\"%s\","
                    "\"output_dtype\":\"%s\",\"candidate_branch\":\"%s\","
                    "\"status\":\"%s\",\"result_code\":%d}\n",
                    static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), layout.name,
                    dtype.inputName, dtype.outputName,
                    CampaignName(),
                    invariantPassed ? "CANDIDATE_TILING_FAILED" : "CANDIDATE_INVARIANT_FAILED", rc);
        std::fflush(stdout);
        if (adaptiveExecutor != nullptr) (void)aclDestroyAclOpExecutor(adaptiveExecutor);
        (void)aclDestroyAclOpExecutor(officialExecutor);
        (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE");
        (void)::unsetenv("MATMUL_SPLITK_MODE");
        (void)::unsetenv("MATMUL_BASE_MODE");
        (void)::unsetenv("MATMUL_VECTOR_ENABLE");
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc;
    }
    std::vector<uint8_t> a(static_cast<size_t>(m * k) * dtype.input.bytes);
    std::vector<uint8_t> b(static_cast<size_t>(n * k) * dtype.input.bytes);
    if (layout.transA) {
        for (int64_t row = 0; row < m; ++row) {
            StoreValue(a, static_cast<size_t>(row), dtype.input,
                       static_cast<float>(row % 3 + 1) / 16.0f);
        }
        RepeatPrefix(a, static_cast<size_t>(m) * dtype.input.bytes);
    } else {
        for (int64_t row = 0; row < m; ++row) {
            FillScalarRange(a, static_cast<size_t>(row * k), static_cast<size_t>(k), dtype.input,
                            static_cast<float>(row % 3 + 1) / 16.0f);
        }
    }
    if (layout.transB) {
        for (int64_t column = 0; column < n; ++column) {
            FillScalarRange(b, static_cast<size_t>(column * k), static_cast<size_t>(k), dtype.input,
                            static_cast<float>(column % 5 + 1) / 16.0f);
        }
    } else {
        for (int64_t column = 0; column < n; ++column) {
            StoreValue(b, static_cast<size_t>(column), dtype.input,
                       static_cast<float>(column % 5 + 1) / 16.0f);
        }
        RepeatPrefix(b, static_cast<size_t>(n) * dtype.input.bytes);
    }
    if (rc == ACL_SUCCESS) {
        rc = aclrtMemcpy(aTensor.device, aTensor.bytes, a.data(), a.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (rc == ACL_SUCCESS) {
        rc = aclrtMemcpy(bTensor.device, bTensor.bytes, b.data(), b.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (rc == ACL_SUCCESS) rc = aclrtMemset(cTensor.device, cTensor.bytes, 0, cTensor.bytes);
    void *officialWorkspace = nullptr;
    const uint64_t sharedWorkspaceSize = std::max(officialWorkspaceSize, adaptiveWorkspaceSize);
    if (rc == ACL_SUCCESS && sharedWorkspaceSize != 0) {
        rc = aclrtMalloc(&officialWorkspace, sharedWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    adaptiveWorkspace = officialWorkspace;
    if (rc == ACL_SUCCESS) rc = aclSetAclOpExecutorRepeatable(officialExecutor);
    float officialLatencyForward = 0.0f;
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclnnMatmul(officialWorkspace, officialWorkspaceSize, officialExecutor, stream);
        }, stream, officialLatencyForward);
    }
    const std::vector<uint8_t> officialOutput = rc == ACL_SUCCESS ? CopyDeviceOutput(cTensor) : std::vector<uint8_t>{};
    if (rc != ACL_SUCCESS || officialOutput.empty()) {
        ++counts.officialFailed;
        std::printf("{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"input_dtype\":\"%s\","
                    "\"output_dtype\":\"%s\",\"status\":\"OFFICIAL_MEASUREMENT_FAILED\",\"result_code\":%d}\n",
                    static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), layout.name,
                    dtype.inputName, dtype.outputName, rc == ACL_SUCCESS ? 4 : rc);
        std::fflush(stdout);
        if (adaptiveExecutor != nullptr) (void)aclDestroyAclOpExecutor(adaptiveExecutor);
        if (officialWorkspace != nullptr) (void)aclrtFree(officialWorkspace);
        (void)aclDestroyAclOpExecutor(officialExecutor);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc == ACL_SUCCESS ? 4 : rc;
    }
    if (rc == ACL_SUCCESS) rc = aclrtMemset(cTensor.device, cTensor.bytes, 0, cTensor.bytes);
    if (rc == ACL_SUCCESS && !edgeCampaign) rc = aclSetAclOpExecutorRepeatable(adaptiveExecutor);
    float adaptiveLatencyForward = 0.0f;
    void *tilingDevice = nullptr;
    aclrtArgsHandle edgeArguments = nullptr;
    if (rc == ACL_SUCCESS && edgeCampaign) {
        const std::vector<uint8_t> tiling = DecodeHex(std::getenv("MATMUL_EXPERIMENT_TILING"));
        if (edgeFunction == nullptr || tiling.empty() || adaptive.cores == 0) rc = 4;
        if (rc == ACL_SUCCESS) rc = aclrtMalloc(&tilingDevice, tiling.size(), ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc == ACL_SUCCESS) {
            rc = aclrtMemcpy(tilingDevice, tiling.size(), tiling.data(), tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (rc == ACL_SUCCESS) rc = aclrtKernelArgsInit(edgeFunction, &edgeArguments);
        void *bias = nullptr;
        void *offset = nullptr;
        void *values[] = {
            aTensor.device, bTensor.device, bias, offset, cTensor.device, adaptiveWorkspace, tilingDevice
        };
        for (void *&value : values) {
            aclrtParamHandle parameter = nullptr;
            if (rc == ACL_SUCCESS) rc = aclrtKernelArgsAppend(edgeArguments, &value, sizeof(value), &parameter);
        }
        if (rc == ACL_SUCCESS) rc = aclrtKernelArgsFinalize(edgeArguments);
    }
    if (rc == ACL_SUCCESS && edgeCampaign) {
        rc = Measure([&]() {
            return aclrtLaunchKernelWithConfig(edgeFunction, adaptive.cores, stream, nullptr,
                                               edgeArguments, nullptr);
        }, stream, adaptiveLatencyForward);
    } else if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclnnMatmul(adaptiveWorkspace, adaptiveWorkspaceSize, adaptiveExecutor, stream);
        }, stream, adaptiveLatencyForward);
    }
    const std::vector<uint8_t> adaptiveOutput = rc == ACL_SUCCESS ? CopyDeviceOutput(cTensor) : std::vector<uint8_t>{};
    float adaptiveLatencyReverse = adaptiveLatencyForward;
    float officialLatencyReverse = officialLatencyForward;
    const bool repeatInReverseOrder = adaptiveCampaign || (baseCampaign && !edgeCampaign);
    if (rc == ACL_SUCCESS && repeatInReverseOrder) {
        rc = Measure([&]() {
            return aclnnMatmul(adaptiveWorkspace, adaptiveWorkspaceSize, adaptiveExecutor, stream);
        }, stream, adaptiveLatencyReverse);
    }
    if (rc == ACL_SUCCESS && repeatInReverseOrder) {
        rc = Measure([&]() {
            return aclnnMatmul(officialWorkspace, officialWorkspaceSize, officialExecutor, stream);
        }, stream, officialLatencyReverse);
    }
    const float officialLatency = (officialLatencyForward + officialLatencyReverse) * 0.5f;
    const float adaptiveLatency = (adaptiveLatencyForward + adaptiveLatencyReverse) * 0.5f;
    double maxAbs = 0.0;
    double maxRel = 0.0;
    const bool correct = rc == ACL_SUCCESS && CompareOutputs(officialOutput, adaptiveOutput, dtype.output, maxAbs, maxRel);
    const double delta = correct ? (adaptiveLatency / officialLatency - 1.0) * 100.0 : 0.0;
    const uint64_t oldPartialBytes = ReadEnvUnsigned("MATMUL_ADAPTIVE_OLD_PARTIAL_BYTES");
    const uint64_t newPartialBytes = ReadEnvUnsigned("MATMUL_ADAPTIVE_NEW_PARTIAL_BYTES");
    const int64_t partialDelta = static_cast<int64_t>(newPartialBytes) - static_cast<int64_t>(oldPartialBytes);
    const double partialSaved = oldPartialBytes > 0 && newPartialBytes <= oldPartialBytes ?
        static_cast<double>(oldPartialBytes - newPartialBytes) * 100.0 / oldPartialBytes : 0.0;
    std::printf("{\"manifest_index\":%lu,\"shape\":\"M%ld_N%ld_K%ld_%s\",\"input_dtype\":\"%s\","
                "\"output_dtype\":\"%s\","
                "\"candidate_branch\":\"%s\",\"candidate_variant\":\"%s\","
                "\"candidate_selected\":%s,\"tiling_changed\":%s,",
                static_cast<unsigned long>(manifestIndex),
                static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), layout.name,
                dtype.inputName, dtype.outputName,
                CampaignName(), candidateVariant.c_str(),
                candidateSelected ? "true" : "false", changed ? "true" : "false");
    PrintTiling("official_tiling", official);
    std::printf(",");
    PrintTiling("candidate_tiling", adaptive);
    std::printf(",\"official_core\":%u,\"candidate_core\":%u", official.cores, adaptive.cores);
    if (adaptiveCampaign) {
        std::printf(",\"old_partial_bytes\":%lu,\"new_partial_bytes\":%lu,"
                    "\"partial_delta_bytes\":%ld,\"partial_saved_bytes\":%lu,"
                    "\"partial_saved_pct\":%.6f,\"k_balance_applied\":false,"
                    "\"k_units\":%lu,\"k_quotient\":%lu,\"k_remainder\":%lu,"
                    "\"balanced_k_min\":%lu,\"balanced_k_max\":%lu",
                    static_cast<unsigned long>(oldPartialBytes),
                    static_cast<unsigned long>(newPartialBytes),
                    static_cast<long>(partialDelta),
                    static_cast<unsigned long>(oldPartialBytes - newPartialBytes), partialSaved,
                    static_cast<unsigned long>(ReadEnvUnsigned("MATMUL_ADAPTIVE_K_UNITS")),
                    static_cast<unsigned long>(ReadEnvUnsigned("MATMUL_ADAPTIVE_K_QUOTIENT")),
                    static_cast<unsigned long>(ReadEnvUnsigned("MATMUL_ADAPTIVE_K_REMAINDER")),
                    static_cast<unsigned long>(ReadEnvUnsigned("MATMUL_ADAPTIVE_K_MIN")),
                    static_cast<unsigned long>(ReadEnvUnsigned("MATMUL_ADAPTIVE_K_MAX")));
    }
    std::printf(",\"measurement_order\":\"OCCO\","
                "\"official_forward_ms\":%.9f,\"candidate_forward_ms\":%.9f,"
                "\"candidate_reverse_ms\":%.9f,\"official_reverse_ms\":%.9f,"
                "\"official_workspace\":%lu,\"candidate_workspace\":%lu,"
                "\"official_latency_ms\":%.9f,\"candidate_latency_ms\":%s,"
                "\"delta_pct\":%s,\"max_abs_diff\":%.9g,\"max_rel_diff\":%.9g,"
                "\"correctness\":\"%s\",\"result_code\":%d}\n",
                officialLatencyForward, adaptiveLatencyForward,
                adaptiveLatencyReverse, officialLatencyReverse,
                static_cast<unsigned long>(officialWorkspaceSize),
                static_cast<unsigned long>(adaptiveWorkspaceSize), officialLatency,
                correct ? std::to_string(adaptiveLatency).c_str() : "null",
                correct ? std::to_string(delta).c_str() : "null", maxAbs, maxRel,
                correct ? "PASS" : "FAIL", rc);
    std::fflush(stdout);

    if (correct) {
        ++counts.passed;
        if (delta < -2.0) {
            ++counts.clearCandidateWins;
        } else if (delta > 2.0) {
            ++counts.clearOfficialWins;
        } else {
            ++counts.overlap;
        }
        if (adaptiveLatencyForward < officialLatencyForward && adaptiveLatencyReverse < officialLatencyReverse) {
            ++counts.stableCandidateWins;
        } else if (adaptiveLatencyForward > officialLatencyForward &&
                   adaptiveLatencyReverse > officialLatencyReverse) {
            ++counts.stableOfficialWins;
        } else {
            ++counts.mixedOrder;
        }
        CountPassCoverage(counts, m, n, k);
        if (candidateVariant == "WIDE_N_PANEL_ONLY" || candidateVariant == "WIDE_N_WINDOW_ONLY" ||
            candidateVariant == "WIDE_N_SHALLOW_K_BASE") {
            ++counts.wideNShallowKPassed;
        }
    } else {
        ++counts.failed;
    }
    if (tilingDevice != nullptr) (void)aclrtFree(tilingDevice);
    if (adaptiveExecutor != nullptr) (void)aclDestroyAclOpExecutor(adaptiveExecutor);
    if (officialWorkspace != nullptr) (void)aclrtFree(officialWorkspace);
    (void)aclDestroyAclOpExecutor(officialExecutor);
    (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE");
    (void)::unsetenv("MATMUL_BASE_MODE");
    (void)::unsetenv("MATMUL_VECTOR_ENABLE");
    (void)::unsetenv("MATMUL_SPLITK_MODE");
    ReleaseTensor(cTensor);
    ReleaseTensor(bTensor);
    ReleaseTensor(aTensor);
    return correct ? ACL_SUCCESS : (rc == ACL_SUCCESS ? 3 : rc);
}

int main(int argc, char **argv)
{
    const bool manifestMode = argc == 3 && std::strcmp(argv[1], "--manifest") == 0;
    if (!manifestMode && (argc < 6 || (argc - 1) % 5 != 0)) return 2;
    if (!IsBaseCampaign() && !IsAdaptiveCampaign()) return 4;
    const uint64_t targetPasses = ReadEnvUnsigned("MATMUL_TARGET_PASSES");
    const uint64_t cellQuota = ReadEnvUnsigned("MATMUL_CELL_QUOTA");
    std::printf("{\"campaign_start\":\"%s\",\"target_passes\":%lu,"
                "\"joint_cell_quota\":%lu,\"runner\":\"wide_n_mechanism_ablation_v2\"}\n",
                CampaignName(), static_cast<unsigned long>(targetPasses), static_cast<unsigned long>(cellQuota));
    std::fflush(stdout);
    const char *hostLibrary = std::getenv("MATMUL_HOST_LIBRARY");
    if (hostLibrary == nullptr) return 4;
    int rc = aclInit(nullptr);
    if (rc == ACL_SUCCESS) rc = aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    if (rc == ACL_SUCCESS) rc = aclrtCreateStream(&stream);
    if (rc != ACL_SUCCESS || TbeLoadSoAndSaveToRegistry(hostLibrary) != 0U) return 4;
    aclrtBinHandle edgeBinary = nullptr;
    aclrtFuncHandle edgeFunction = nullptr;
    if (std::strcmp(CampaignName(), "CUBE_VECTOR_EDGE") == 0) {
        const char *path = std::getenv("MATMUL_EDGE_BINARY");
        if (path == nullptr) return 4;
        rc = aclrtBinaryLoadFromFile(path, nullptr, &edgeBinary);
        if (rc == ACL_SUCCESS) {
            rc = aclrtBinaryGetFunction(edgeBinary, "MatMulV3_CubeVectorEdge_3211264", &edgeFunction);
        }
        if (rc != ACL_SUCCESS) return rc;
    }

    RunCounts counts;
    const uint64_t startIndex = ReadEnvUnsigned("MATMUL_START_INDEX");
    uint64_t manifestIndex = 0;
    auto runOne = [&](const char *dtypeName, const char *layoutName, int64_t m, int64_t n, int64_t k) {
        const DTypeSpec *dtype = FindDType(dtypeName);
        const LayoutSpec *layout = FindLayout(layoutName);
        if (dtype == nullptr || layout == nullptr) return;
        const size_t jointBucket = JointCoverageBucket(*dtype, m, n, k);
        ++counts.jointInputs[jointBucket];
        if (cellQuota != 0 && counts.jointPassed[jointBucket] >= cellQuota) {
            ++counts.jointQuotaSkipped;
            return;
        }
        const uint64_t officialBefore = counts.deterministic;
        const uint64_t passedBefore = counts.passed;
        (void)RunWorkload(*dtype, *layout, m, n, k, stream, edgeFunction, counts, manifestIndex);
        if (counts.deterministic > officialBefore) ++counts.jointOfficial[jointBucket];
        if (counts.passed > passedBefore) ++counts.jointPassed[jointBucket];
        if (counts.inputs != 0 && counts.inputs % 1000 == 0) {
            std::printf("{\"progress\":true,\"manifest_index\":%lu,\"inputs\":%lu,"
                        "\"passed\":%lu,\"official_failed\":%lu,\"non_target_route\":%lu}\n",
                        static_cast<unsigned long>(manifestIndex),
                        static_cast<unsigned long>(counts.inputs),
                        static_cast<unsigned long>(counts.passed),
                        static_cast<unsigned long>(counts.officialFailed),
                        static_cast<unsigned long>(counts.nonDeterministic));
            std::fflush(stdout);
        }
    };
    if (manifestMode) {
        std::ifstream input(argv[2]);
        if (!input) return 4;
        std::string dtypeName;
        std::string layoutName;
        int64_t m = 0;
        int64_t n = 0;
        int64_t k = 0;
        while (input >> dtypeName >> layoutName >> m >> n >> k) {
            ++manifestIndex;
            if (manifestIndex <= startIndex) continue;
            runOne(dtypeName.c_str(), layoutName.c_str(), m, n, k);
            if (targetPasses != 0 && counts.passed >= targetPasses) break;
        }
    } else {
        for (int index = 1; index < argc; index += 5) {
            ++manifestIndex;
            if (manifestIndex <= startIndex) continue;
            runOne(argv[index], argv[index + 1],
                   std::strtoll(argv[index + 2], nullptr, 10),
                   std::strtoll(argv[index + 3], nullptr, 10),
                   std::strtoll(argv[index + 4], nullptr, 10));
            if (targetPasses != 0 && counts.passed >= targetPasses) break;
        }
    }
    std::printf("{\"summary\":true,\"campaign\":\"%s\",\"inputs\":%lu,\"skipped_non_v3\":%lu,"
                "\"non_target_route\":%lu,"
                "\"official_target\":%lu,\"candidate_selected\":%lu,\"official_preserved\":%lu,"
                "\"target_passes\":%lu,\"quota_met\":%s,\"passed\":%lu,\"failed\":%lu,"
                "\"wide_n_shallow_k_passed\":%lu,"
                "\"clear_candidate_wins\":%lu,\"false_positive_intercepts\":%lu,\"overlap\":%lu,"
                "\"stable_candidate_wins\":%lu,\"stable_official_wins\":%lu,\"mixed_order\":%lu,"
                "\"official_failed\":%lu,\"manifest_start_index\":%lu,"
                "\"manifest_end_index\":%lu}\n",
                CampaignName(),
                static_cast<unsigned long>(counts.inputs),
                static_cast<unsigned long>(counts.skippedNonV3),
                static_cast<unsigned long>(counts.nonDeterministic),
                static_cast<unsigned long>(counts.deterministic),
                static_cast<unsigned long>(counts.adaptiveSelected),
                static_cast<unsigned long>(counts.officialPreserved),
                static_cast<unsigned long>(targetPasses),
                (targetPasses == 0 || counts.passed >= targetPasses) ? "true" : "false",
                static_cast<unsigned long>(counts.passed),
                static_cast<unsigned long>(counts.failed),
                static_cast<unsigned long>(counts.wideNShallowKPassed),
                static_cast<unsigned long>(counts.clearCandidateWins),
                static_cast<unsigned long>(counts.clearOfficialWins),
                static_cast<unsigned long>(counts.overlap),
                static_cast<unsigned long>(counts.stableCandidateWins),
                static_cast<unsigned long>(counts.stableOfficialWins),
                static_cast<unsigned long>(counts.mixedOrder),
                static_cast<unsigned long>(counts.officialFailed),
                static_cast<unsigned long>(startIndex),
                static_cast<unsigned long>(manifestIndex));
    std::printf("{\"measured_coverage\":true,"
                "\"m\":{\"1_9\":%lu,\"10_31\":%lu,\"32_63\":%lu,\"64_95\":%lu,\"96_128\":%lu},"
                "\"n\":{\"23296_24576\":%lu,\"24832_26624\":%lu,\"26880_28160\":%lu,\"28416_30464\":%lu,\"30720_32768\":%lu},"
                "\"k\":{\"2048_8192\":%lu,\"8320_15872\":%lu,\"16000_27264\":%lu,\"27392_65536\":%lu}}\n",
                static_cast<unsigned long>(counts.mCoverage[0]),
                static_cast<unsigned long>(counts.mCoverage[1]),
                static_cast<unsigned long>(counts.mCoverage[2]),
                static_cast<unsigned long>(counts.mCoverage[3]),
                static_cast<unsigned long>(counts.mCoverage[4]),
                static_cast<unsigned long>(counts.nCoverage[0]),
                static_cast<unsigned long>(counts.nCoverage[1]),
                static_cast<unsigned long>(counts.nCoverage[2]),
                static_cast<unsigned long>(counts.nCoverage[3]),
                static_cast<unsigned long>(counts.nCoverage[4]),
                static_cast<unsigned long>(counts.kCoverage[0]),
                static_cast<unsigned long>(counts.kCoverage[1]),
                static_cast<unsigned long>(counts.kCoverage[2]),
                static_cast<unsigned long>(counts.kCoverage[3]));
    if (cellQuota != 0) {
        uint64_t inputCells = 0;
        uint64_t officialCells = 0;
        uint64_t metCells = 0;
        uint64_t partialCells = 0;
        uint64_t unreachedCells = 0;
        uint64_t pairedPasses = 0;
        for (size_t index = 0; index < 200; ++index) {
            pairedPasses += counts.jointPassed[index];
            if (counts.jointInputs[index] != 0) ++inputCells;
            if (counts.jointOfficial[index] != 0) {
                ++officialCells;
                if (counts.jointPassed[index] >= cellQuota) {
                    ++metCells;
                } else {
                    ++partialCells;
                }
            } else if (counts.jointInputs[index] != 0) {
                ++unreachedCells;
            }
        }
        std::printf("{\"joint_coverage\":true,\"quota_per_cell\":%lu,\"input_cells\":%lu,"
                    "\"official_base_cells\":%lu,\"quota_met_cells\":%lu,\"partial_cells\":%lu,"
                    "\"unreached_cells\":%lu,\"unrepresented_cells\":%lu,\"paired_passes\":%lu,"
                    "\"represented_capacity\":%lu,\"quota_skipped_inputs\":%lu}\n",
                    static_cast<unsigned long>(cellQuota), static_cast<unsigned long>(inputCells),
                    static_cast<unsigned long>(officialCells), static_cast<unsigned long>(metCells),
                    static_cast<unsigned long>(partialCells), static_cast<unsigned long>(unreachedCells),
                    static_cast<unsigned long>(200 - inputCells), static_cast<unsigned long>(pairedPasses),
                    static_cast<unsigned long>(inputCells * cellQuota),
                    static_cast<unsigned long>(counts.jointQuotaSkipped));
        const char *dtypeNames[] = {"fp16", "bf16"};
        const char *mNames[] = {"1_9", "10_31", "32_63", "64_95", "96_128"};
        const char *nNames[] = {"23296_24576", "24832_26624", "26880_28160", "28416_30464", "30720_32768"};
        const char *kNames[] = {"2048_8192", "8320_15872", "16000_27264", "27392_65536"};
        bool first = true;
        std::printf("{\"joint_unmet_cells\":[");
        for (size_t index = 0; index < 200; ++index) {
            if (counts.jointPassed[index] >= cellQuota) continue;
            size_t value = index;
            const size_t kBucket = value % 4;
            value /= 4;
            const size_t nBucket = value % 5;
            value /= 5;
            const size_t mBucket = value % 5;
            value /= 5;
            std::printf("%s\"%s:M%s:N%s:K%s:input=%lu:official=%lu:passed=%lu\"", first ? "" : ",",
                        dtypeNames[value], mNames[mBucket], nNames[nBucket], kNames[kBucket],
                        static_cast<unsigned long>(counts.jointInputs[index]),
                        static_cast<unsigned long>(counts.jointOfficial[index]),
                        static_cast<unsigned long>(counts.jointPassed[index]));
            first = false;
        }
        std::printf("]}\n");
    }
    if (edgeBinary != nullptr) (void)aclrtBinaryUnLoad(edgeBinary);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return counts.deterministic == 0 || (targetPasses != 0 && counts.passed < targetPasses) ? 4 : 0;
}
// NEW END
