// NEW BEGIN
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h"

extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char *soPath);

constexpr int WARMUP = 1;
constexpr int REPEATS = 5;

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
    uint32_t stepKa = 0;
    uint32_t stepKb = 0;
    uint32_t depthA1 = 0;
    uint32_t depthB1 = 0;
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
};

uint64_t ReadEnvUnsigned(const char *name)
{
    const char *text = std::getenv(name);
    return text == nullptr ? 0 : std::strtoull(text, nullptr, 10);
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
    value.stepKa = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_STEP_KA"));
    value.stepKb = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_STEP_KB"));
    value.depthA1 = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_DEPTH_A1"));
    value.depthB1 = static_cast<uint32_t>(ReadEnvUnsigned("MATMUL_OBSERVED_DEPTH_B1"));
    return value;
}

void ClearObservedTiling()
{
    const char *names[] = {
        "MATMUL_OBSERVED_KEY", "MATMUL_OBSERVED_CORES", "MATMUL_OBSERVED_SINGLE_M",
        "MATMUL_OBSERVED_SINGLE_N", "MATMUL_OBSERVED_SINGLE_K", "MATMUL_OBSERVED_BASE_M",
        "MATMUL_OBSERVED_BASE_N", "MATMUL_OBSERVED_BASE_K", "MATMUL_OBSERVED_STEP_KA",
        "MATMUL_OBSERVED_STEP_KB", "MATMUL_OBSERVED_DEPTH_A1", "MATMUL_OBSERVED_DEPTH_B1"
    };
    for (const char *name : names) (void)::unsetenv(name);
}

bool IsDeterministicSplitK(uint64_t key)
{
    return ((key >> 4U) & 0xffU) == 3U;
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

int CreateTensor(const std::vector<uint8_t> &host, const std::vector<int64_t> &shape,
                 const std::vector<int64_t> &storage, const std::vector<int64_t> &strides,
                 const ScalarType &dtype, Tensor &value)
{
    value.bytes = host.size();
    int rc = aclrtMalloc(&value.device, value.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc == ACL_SUCCESS) {
        rc = aclrtMemcpy(value.device, value.bytes, host.data(), value.bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
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
                "\"step_ka\":%u,\"step_kb\":%u,\"depth_a1\":%u,\"depth_b1\":%u}",
                name, static_cast<unsigned long>(value.key), value.cores, value.singleM, value.singleN,
                value.singleK, value.baseM, value.baseN, value.baseK, value.stepKa, value.stepKb,
                value.depthA1, value.depthB1);
}

int RunWorkload(const DTypeSpec &dtype, const LayoutSpec &layout, int64_t m, int64_t n, int64_t k,
                aclrtStream stream, RunCounts &counts)
{
    ++counts.inputs;
    std::vector<uint8_t> a(static_cast<size_t>(m * k) * dtype.input.bytes);
    std::vector<uint8_t> b(static_cast<size_t>(n * k) * dtype.input.bytes);
    std::vector<uint8_t> c(static_cast<size_t>(m * n) * dtype.output.bytes, 0);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t index = 0; index < k; ++index) {
            const size_t offset = layout.transA ? static_cast<size_t>(index * m + row) :
                                                  static_cast<size_t>(row * k + index);
            const float value = static_cast<float>((row % 3 + 1) * (index % 2 + 1)) / 16.0f;
            StoreValue(a, offset, dtype.input, value);
        }
    }
    for (int64_t column = 0; column < n; ++column) {
        for (int64_t index = 0; index < k; ++index) {
            const size_t offset = layout.transB ? static_cast<size_t>(column * k + index) :
                                                  static_cast<size_t>(index * n + column);
            const float value = static_cast<float>((column % 5 + 1) * ((index / 2) % 2 + 1)) / 16.0f;
            StoreValue(b, offset, dtype.input, value);
        }
    }

    Tensor aTensor;
    Tensor bTensor;
    Tensor cTensor;
    int rc = CreateTensor(a, {m, k}, layout.transA ? std::vector<int64_t>{k, m} : std::vector<int64_t>{m, k},
                          layout.transA ? std::vector<int64_t>{1, m} : std::vector<int64_t>{k, 1}, dtype.input, aTensor);
    if (rc == ACL_SUCCESS) {
        rc = CreateTensor(b, {k, n}, layout.transB ? std::vector<int64_t>{n, k} : std::vector<int64_t>{k, n},
                          layout.transB ? std::vector<int64_t>{1, k} : std::vector<int64_t>{n, 1}, dtype.input, bTensor);
    }
    if (rc == ACL_SUCCESS) rc = CreateTensor(c, {m, n}, {m, n}, {n, 1}, dtype.output, cTensor);
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
    (void)::setenv("MATMUL_VECTOR_ENABLE", "0", 1);
    ClearObservedTiling();
    uint64_t officialWorkspaceSize = 0;
    aclOpExecutor *officialExecutor = nullptr;
    rc = aclnnMatmulGetWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, 1,
                                     &officialWorkspaceSize, &officialExecutor);
    const TilingSnapshot official = ReadTilingSnapshot();
    if (rc != ACL_SUCCESS || officialExecutor == nullptr) {
        ++counts.officialFailed;
        std::printf("{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"input_dtype\":\"%s\","
                    "\"output_dtype\":\"%s\",\"status\":\"OFFICIAL_TILING_FAILED\",\"result_code\":%d}\n",
                    static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), layout.name,
                    dtype.inputName, dtype.outputName, rc == ACL_SUCCESS ? 4 : rc);
        std::fflush(stdout);
        if (officialExecutor != nullptr) (void)aclDestroyAclOpExecutor(officialExecutor);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc == ACL_SUCCESS ? 4 : rc;
    }
    if (!IsDeterministicSplitK(official.key)) {
        ++counts.nonDeterministic;
        (void)aclDestroyAclOpExecutor(officialExecutor);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return ACL_SUCCESS;
    }
    ++counts.deterministic;

    void *officialWorkspace = nullptr;
    if (officialWorkspaceSize != 0) rc = aclrtMalloc(&officialWorkspace, officialWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc == ACL_SUCCESS) rc = aclSetAclOpExecutorRepeatable(officialExecutor);
    float officialLatency = 0.0f;
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclnnMatmul(officialWorkspace, officialWorkspaceSize, officialExecutor, stream);
        }, stream, officialLatency);
    }
    const std::vector<uint8_t> officialOutput = rc == ACL_SUCCESS ? CopyDeviceOutput(cTensor) : std::vector<uint8_t>{};
    if (officialWorkspace != nullptr) (void)aclrtFree(officialWorkspace);
    (void)aclDestroyAclOpExecutor(officialExecutor);
    if (rc != ACL_SUCCESS || officialOutput.empty()) {
        ++counts.officialFailed;
        std::printf("{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"input_dtype\":\"%s\","
                    "\"output_dtype\":\"%s\",\"status\":\"OFFICIAL_MEASUREMENT_FAILED\",\"result_code\":%d}\n",
                    static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), layout.name,
                    dtype.inputName, dtype.outputName, rc == ACL_SUCCESS ? 4 : rc);
        std::fflush(stdout);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return rc == ACL_SUCCESS ? 4 : rc;
    }

    (void)::setenv("MATMUL_DETERMINISTIC_ADAPTIVE", "1", 1);
    (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED");
    ClearObservedTiling();
    uint64_t adaptiveWorkspaceSize = 0;
    aclOpExecutor *adaptiveExecutor = nullptr;
    rc = aclnnMatmulGetWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, 1,
                                     &adaptiveWorkspaceSize, &adaptiveExecutor);
    const TilingSnapshot adaptive = ReadTilingSnapshot();
    const bool changed = ReadEnvUnsigned("MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED") == 1;
    if (changed) ++counts.adaptiveSelected; else ++counts.officialPreserved;
    void *adaptiveWorkspace = nullptr;
    if (rc == ACL_SUCCESS && adaptiveExecutor == nullptr) rc = 4;
    if (rc == ACL_SUCCESS && !IsDeterministicSplitK(adaptive.key)) rc = 4;
    if (rc == ACL_SUCCESS && adaptiveWorkspaceSize != 0) {
        rc = aclrtMalloc(&adaptiveWorkspace, adaptiveWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (rc == ACL_SUCCESS) rc = aclrtMemset(cTensor.device, cTensor.bytes, 0, cTensor.bytes);
    if (rc == ACL_SUCCESS) rc = aclSetAclOpExecutorRepeatable(adaptiveExecutor);
    float adaptiveLatency = 0.0f;
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclnnMatmul(adaptiveWorkspace, adaptiveWorkspaceSize, adaptiveExecutor, stream);
        }, stream, adaptiveLatency);
    }
    const std::vector<uint8_t> adaptiveOutput = rc == ACL_SUCCESS ? CopyDeviceOutput(cTensor) : std::vector<uint8_t>{};
    double maxAbs = 0.0;
    double maxRel = 0.0;
    const bool correct = rc == ACL_SUCCESS && CompareOutputs(officialOutput, adaptiveOutput, dtype.output, maxAbs, maxRel);
    const double delta = correct ? (adaptiveLatency / officialLatency - 1.0) * 100.0 : 0.0;

    std::printf("{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"input_dtype\":\"%s\","
                "\"output_dtype\":\"%s\","
                "\"adaptive_changed\":%s,",
                static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), layout.name,
                dtype.inputName, dtype.outputName, changed ? "true" : "false");
    PrintTiling("official_tiling", official);
    std::printf(",");
    PrintTiling("adaptive_tiling", adaptive);
    std::printf(",\"official_workspace\":%lu,\"adaptive_workspace\":%lu,"
                "\"official_latency_ms\":%.9f,\"adaptive_latency_ms\":%s,"
                "\"delta_pct\":%s,\"max_abs_diff\":%.9g,\"max_rel_diff\":%.9g,"
                "\"correctness\":\"%s\",\"result_code\":%d}\n",
                static_cast<unsigned long>(officialWorkspaceSize),
                static_cast<unsigned long>(adaptiveWorkspaceSize), officialLatency,
                correct ? std::to_string(adaptiveLatency).c_str() : "null",
                correct ? std::to_string(delta).c_str() : "null", maxAbs, maxRel,
                correct ? "PASS" : "FAIL", rc);
    std::fflush(stdout);

    if (correct) ++counts.passed; else ++counts.failed;
    if (adaptiveWorkspace != nullptr) (void)aclrtFree(adaptiveWorkspace);
    if (adaptiveExecutor != nullptr) (void)aclDestroyAclOpExecutor(adaptiveExecutor);
    (void)::unsetenv("MATMUL_DETERMINISTIC_ADAPTIVE");
    ReleaseTensor(cTensor);
    ReleaseTensor(bTensor);
    ReleaseTensor(aTensor);
    return correct ? ACL_SUCCESS : (rc == ACL_SUCCESS ? 3 : rc);
}

int main(int argc, char **argv)
{
    if (argc < 6 || (argc - 1) % 5 != 0) return 2;
    const char *hostLibrary = std::getenv("MATMUL_HOST_LIBRARY");
    if (hostLibrary == nullptr) return 4;
    int rc = aclInit(nullptr);
    if (rc == ACL_SUCCESS) rc = aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    if (rc == ACL_SUCCESS) rc = aclrtCreateStream(&stream);
    if (rc != ACL_SUCCESS || TbeLoadSoAndSaveToRegistry(hostLibrary) != 0U) return 4;

    RunCounts counts;
    for (int index = 1; index < argc; index += 5) {
        const DTypeSpec *dtype = FindDType(argv[index]);
        const LayoutSpec *layout = FindLayout(argv[index + 1]);
        if (dtype == nullptr || layout == nullptr) continue;
        (void)RunWorkload(*dtype, *layout,
                          std::strtoll(argv[index + 2], nullptr, 10),
                          std::strtoll(argv[index + 3], nullptr, 10),
                          std::strtoll(argv[index + 4], nullptr, 10), stream, counts);
    }
    std::printf("{\"summary\":true,\"inputs\":%lu,\"non_deterministic\":%lu,"
                "\"deterministic\":%lu,\"adaptive_selected\":%lu,\"official_preserved\":%lu,"
                "\"passed\":%lu,\"failed\":%lu,"
                "\"official_failed\":%lu}\n",
                static_cast<unsigned long>(counts.inputs),
                static_cast<unsigned long>(counts.nonDeterministic),
                static_cast<unsigned long>(counts.deterministic),
                static_cast<unsigned long>(counts.adaptiveSelected),
                static_cast<unsigned long>(counts.officialPreserved),
                static_cast<unsigned long>(counts.passed),
                static_cast<unsigned long>(counts.failed),
                static_cast<unsigned long>(counts.officialFailed));
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return counts.deterministic == 0 ? 4 : 0;
}
// NEW END
