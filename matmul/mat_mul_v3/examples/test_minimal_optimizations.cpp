//NEW BEGIN
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h"

extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char *soPath);

struct ScalarType {
    aclDataType type;
    size_t bytes;
    int fill;
};

struct Tensor {
    aclTensor *tensor = nullptr;
    void *device = nullptr;
    size_t bytes = 0;
};

struct Shape {
    const char *branch;
    int64_t m;
    int64_t n;
    int64_t k;
    bool transB;
    ScalarType scalar;
};

void Release(Tensor &value)
{
    if (value.tensor != nullptr) (void)aclDestroyTensor(value.tensor);
    if (value.device != nullptr) (void)aclrtFree(value.device);
    value = {};
}

int CreateTensor(size_t elements, const std::vector<int64_t> &shape,
                 const std::vector<int64_t> &storage, const std::vector<int64_t> &strides,
                 ScalarType scalar, Tensor &value)
{
    value.bytes = elements * scalar.bytes;
    int rc = aclrtMalloc(&value.device, value.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc == ACL_SUCCESS) rc = aclrtMemset(value.device, value.bytes, scalar.fill, value.bytes);
    if (rc == ACL_SUCCESS) {
        value.tensor = aclCreateTensor(shape.data(), shape.size(), scalar.type, strides.data(), 0,
                                       ACL_FORMAT_ND, storage.data(), storage.size(), value.device);
        if (value.tensor == nullptr) rc = 1;
    }
    return rc;
}

template <typename Launch>
int Measure(Launch launch, aclrtStream stream, float &latency)
{
    int rc = ACL_SUCCESS;
    for (int index = 0; rc == ACL_SUCCESS && index < 3; ++index) rc = launch();
    if (rc == ACL_SUCCESS) rc = aclrtSynchronizeStream(stream);
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    if (rc == ACL_SUCCESS) rc = aclrtCreateEvent(&start);
    if (rc == ACL_SUCCESS) rc = aclrtCreateEvent(&end);
    if (rc == ACL_SUCCESS) rc = aclrtRecordEvent(start, stream);
    for (int index = 0; rc == ACL_SUCCESS && index < 30; ++index) rc = launch();
    if (rc == ACL_SUCCESS) rc = aclrtRecordEvent(end, stream);
    if (rc == ACL_SUCCESS) rc = aclrtSynchronizeEvent(end);
    float total = 0.0f;
    if (rc == ACL_SUCCESS) rc = aclrtEventElapsedTime(&total, start, end);
    if (end != nullptr) (void)aclrtDestroyEvent(end);
    if (start != nullptr) (void)aclrtDestroyEvent(start);
    if (rc == ACL_SUCCESS) latency = total / 30.0f;
    return rc;
}

std::vector<uint8_t> CopyOutput(const Tensor &value)
{
    std::vector<uint8_t> host(value.bytes);
    if (aclrtMemcpy(host.data(), host.size(), value.device, value.bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) return {};
    return host;
}

float LoadValue(const std::vector<uint8_t> &data, size_t index, ScalarType scalar)
{
    const uint8_t *source = data.data() + index * scalar.bytes;
    if (scalar.type == ACL_FLOAT) {
        float value = 0.0f;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }
    uint16_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    if (scalar.type == ACL_FLOAT16) return aclFloat16ToFloat(static_cast<aclFloat16>(value));
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool SameOutput(const std::vector<uint8_t> &reference, const std::vector<uint8_t> &candidate,
                ScalarType scalar)
{
    if (reference.empty() || reference.size() != candidate.size()) return false;
    const size_t count = reference.size() / scalar.bytes;
    const double tolerance = scalar.type == ACL_FLOAT ? 2.0e-3 :
        (scalar.type == ACL_FLOAT16 ? 2.0e-2 : 5.0e-2);
    for (size_t index = 0; index < count; ++index) {
        const double expected = LoadValue(reference, index, scalar);
        const double actual = LoadValue(candidate, index, scalar);
        if (!std::isfinite(expected) || !std::isfinite(actual)) return false;
        const double absolute = std::fabs(actual - expected);
        const double relative = absolute / std::max(1.0, std::fabs(expected));
        if (absolute > 1.0 && relative > tolerance) return false;
    }
    return true;
}

std::vector<uint8_t> DecodeHex(const char *text)
{
    if (text == nullptr || (std::strlen(text) & 1U) != 0U) return {};
    auto nibble = [](char value) {
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

int PrepareTensors(const Shape &shape, Tensor &a, Tensor &b, Tensor &c)
{
    int rc = CreateTensor(static_cast<size_t>(shape.m * shape.k), {shape.m, shape.k},
                          {shape.m, shape.k}, {shape.k, 1}, shape.scalar, a);
    if (rc == ACL_SUCCESS && shape.transB) {
        rc = CreateTensor(static_cast<size_t>(shape.n * shape.k), {shape.k, shape.n},
                          {shape.n, shape.k}, {1, shape.k}, shape.scalar, b);
    } else if (rc == ACL_SUCCESS) {
        rc = CreateTensor(static_cast<size_t>(shape.n * shape.k), {shape.k, shape.n},
                          {shape.k, shape.n}, {shape.n, 1}, shape.scalar, b);
    }
    if (rc == ACL_SUCCESS) {
        rc = CreateTensor(static_cast<size_t>(shape.m * shape.n), {shape.m, shape.n},
                          {shape.m, shape.n}, {shape.n, 1}, shape.scalar, c);
    }
    return rc;
}

bool Selected(const char *branch)
{
    const char *selected = std::getenv("MATMUL_OPT_SELECTED");
    return selected != nullptr && std::strcmp(selected, branch) == 0;
}

int RunCube(const Shape &shape, aclrtStream stream)
{
    Tensor a;
    Tensor b;
    Tensor c;
    int rc = PrepareTensors(shape, a, b, c);
    uint64_t originalWorkspaceSize = 0;
    uint64_t optimisedWorkspaceSize = 0;
    aclOpExecutor *originalExecutor = nullptr;
    aclOpExecutor *optimisedExecutor = nullptr;
    if (rc == ACL_SUCCESS) {
        (void)::unsetenv("MATMUL_OPT_BRANCH");
        (void)::unsetenv("MATMUL_OPT_SELECTED");
        rc = aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, c.tensor, 1,
                                         &originalWorkspaceSize, &originalExecutor);
    }
    if (rc == ACL_SUCCESS) {
        (void)::setenv("MATMUL_OPT_BRANCH", shape.branch, 1);
        (void)::unsetenv("MATMUL_OPT_SELECTED");
        rc = aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, c.tensor, 1,
                                         &optimisedWorkspaceSize, &optimisedExecutor);
        if (rc == ACL_SUCCESS && !Selected(shape.branch)) rc = 4;
    }
    void *workspace = nullptr;
    const uint64_t workspaceSize = std::max(originalWorkspaceSize, optimisedWorkspaceSize);
    if (rc == ACL_SUCCESS && workspaceSize != 0) rc = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc == ACL_SUCCESS) rc = aclSetAclOpExecutorRepeatable(originalExecutor);
    if (rc == ACL_SUCCESS) rc = aclSetAclOpExecutorRepeatable(optimisedExecutor);
    float originalLatency = 0.0f;
    float optimisedLatency = 0.0f;
    if (rc == ACL_SUCCESS) rc = aclrtMemset(c.device, c.bytes, 0, c.bytes);
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclnnMatmul(workspace, originalWorkspaceSize, originalExecutor, stream);
        }, stream, originalLatency);
    }
    const std::vector<uint8_t> original = rc == ACL_SUCCESS ? CopyOutput(c) : std::vector<uint8_t>{};
    if (rc == ACL_SUCCESS) rc = aclrtMemset(c.device, c.bytes, 0, c.bytes);
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclnnMatmul(workspace, optimisedWorkspaceSize, optimisedExecutor, stream);
        }, stream, optimisedLatency);
    }
    const std::vector<uint8_t> optimised = rc == ACL_SUCCESS ? CopyOutput(c) : std::vector<uint8_t>{};
    if (rc == ACL_SUCCESS && !SameOutput(original, optimised, shape.scalar)) rc = 3;
    if (rc == ACL_SUCCESS) {
        std::printf("{\"shape\":\"M%lld_N%lld_K%lld_%s\",\"branch\":\"%s\","
                    "\"original_latency\":%.9f,\"optimised_latency\":%.9f}\n",
                    static_cast<long long>(shape.m), static_cast<long long>(shape.n),
                    static_cast<long long>(shape.k), shape.transB ? "NT" : "NN",
                    shape.branch, originalLatency, optimisedLatency);
        std::fflush(stdout);
    }
    if (workspace != nullptr) (void)aclrtFree(workspace);
    if (optimisedExecutor != nullptr) (void)aclDestroyAclOpExecutor(optimisedExecutor);
    if (originalExecutor != nullptr) (void)aclDestroyAclOpExecutor(originalExecutor);
    Release(c);
    Release(b);
    Release(a);
    return rc;
}

int RunVector(const Shape &shape, aclrtStream stream, aclrtFuncHandle function)
{
    Tensor a;
    Tensor b;
    Tensor c;
    int rc = PrepareTensors(shape, a, b, c);
    uint64_t originalWorkspaceSize = 0;
    aclOpExecutor *originalExecutor = nullptr;
    if (rc == ACL_SUCCESS) {
        (void)::unsetenv("MATMUL_OPT_BRANCH");
        (void)::unsetenv("MATMUL_OPT_SELECTED");
        rc = aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, c.tensor, 1,
                                         &originalWorkspaceSize, &originalExecutor);
    }
    void *workspace = nullptr;
    if (rc == ACL_SUCCESS && originalWorkspaceSize != 0) {
        rc = aclrtMalloc(&workspace, originalWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (rc == ACL_SUCCESS) rc = aclSetAclOpExecutorRepeatable(originalExecutor);
    float originalLatency = 0.0f;
    if (rc == ACL_SUCCESS) rc = aclrtMemset(c.device, c.bytes, 0, c.bytes);
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclnnMatmul(workspace, originalWorkspaceSize, originalExecutor, stream);
        }, stream, originalLatency);
    }
    const std::vector<uint8_t> original = rc == ACL_SUCCESS ? CopyOutput(c) : std::vector<uint8_t>{};

    (void)::setenv("MATMUL_OPT_BRANCH", shape.branch, 1);
    (void)::unsetenv("MATMUL_OPT_SELECTED");
    (void)::unsetenv("MATMUL_VECTOR_TILING");
    (void)::unsetenv("MATMUL_VECTOR_CORES");
    uint64_t ignoredWorkspaceSize = 0;
    aclOpExecutor *ignoredExecutor = nullptr;
    if (rc == ACL_SUCCESS) {
        rc = aclnnMatmulGetWorkspaceSize(a.tensor, b.tensor, c.tensor, 1,
                                         &ignoredWorkspaceSize, &ignoredExecutor);
        if (rc == ACL_SUCCESS && !Selected(shape.branch)) rc = 4;
    }
    const std::vector<uint8_t> tiling = DecodeHex(std::getenv("MATMUL_VECTOR_TILING"));
    const char *coreText = std::getenv("MATMUL_VECTOR_CORES");
    const uint32_t cores = coreText == nullptr ? 0 : static_cast<uint32_t>(std::strtoul(coreText, nullptr, 10));
    if (ignoredExecutor != nullptr) (void)aclDestroyAclOpExecutor(ignoredExecutor);
    void *tilingDevice = nullptr;
    aclrtArgsHandle arguments = nullptr;
    if (rc == ACL_SUCCESS && (tiling.empty() || cores == 0)) rc = 4;
    if (rc == ACL_SUCCESS) rc = aclrtMalloc(&tilingDevice, tiling.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc == ACL_SUCCESS) {
        rc = aclrtMemcpy(tilingDevice, tiling.size(), tiling.data(), tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (rc == ACL_SUCCESS) rc = aclrtKernelArgsInit(function, &arguments);
    void *bias = nullptr;
    void *offset = nullptr;
    void *candidateWorkspace = nullptr;
    void *values[] = {a.device, b.device, bias, offset, c.device, candidateWorkspace, tilingDevice};
    for (void *&value : values) {
        aclrtParamHandle parameter = nullptr;
        if (rc == ACL_SUCCESS) rc = aclrtKernelArgsAppend(arguments, &value, sizeof(value), &parameter);
    }
    if (rc == ACL_SUCCESS) rc = aclrtKernelArgsFinalize(arguments);
    float optimisedLatency = 0.0f;
    if (rc == ACL_SUCCESS) rc = aclrtMemset(c.device, c.bytes, 0, c.bytes);
    if (rc == ACL_SUCCESS) {
        rc = Measure([&]() {
            return aclrtLaunchKernelWithConfig(function, cores, stream, nullptr, arguments, nullptr);
        }, stream, optimisedLatency);
    }
    const std::vector<uint8_t> optimised = rc == ACL_SUCCESS ? CopyOutput(c) : std::vector<uint8_t>{};
    if (rc == ACL_SUCCESS && !SameOutput(original, optimised, shape.scalar)) rc = 3;
    if (rc == ACL_SUCCESS) {
        std::printf("{\"shape\":\"M%lld_N%lld_K%lld_%s\",\"branch\":\"%s\","
                    "\"original_latency\":%.9f,\"optimised_latency\":%.9f}\n",
                    static_cast<long long>(shape.m), static_cast<long long>(shape.n),
                    static_cast<long long>(shape.k), shape.transB ? "NT" : "NN",
                    shape.branch, originalLatency, optimisedLatency);
        std::fflush(stdout);
    }
    if (tilingDevice != nullptr) (void)aclrtFree(tilingDevice);
    if (workspace != nullptr) (void)aclrtFree(workspace);
    if (originalExecutor != nullptr) (void)aclDestroyAclOpExecutor(originalExecutor);
    Release(c);
    Release(b);
    Release(a);
    return rc;
}

int main()
{
    int rc = aclInit(nullptr);
    if (rc == ACL_SUCCESS) rc = aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    if (rc == ACL_SUCCESS) rc = aclrtCreateStream(&stream);
    const char *hostLibrary = std::getenv("MATMUL_HOST_LIBRARY");
    const char *vectorBinary = std::getenv("MATMUL_VECTOR_BINARY");
    if (rc != ACL_SUCCESS || hostLibrary == nullptr || vectorBinary == nullptr ||
        TbeLoadSoAndSaveToRegistry(hostLibrary) != 0U) return 4;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    rc = aclrtBinaryLoadFromFile(vectorBinary, nullptr, &binary);
    if (rc == ACL_SUCCESS) rc = aclrtBinaryGetFunction(binary, "MatMulV3_VectorDot_2162688", &function);
    const ScalarType fp32{ACL_FLOAT, sizeof(float), 0x3f};
    const ScalarType fp16{ACL_FLOAT16, sizeof(uint16_t), 0x3c};
    const ScalarType bf16{ACL_BF16, sizeof(uint16_t), 0x3f};
    const Shape shapes[] = {
        {"VECTOR_DOT", 1, 23, 8192, true, fp32},
        {"VECTOR_DOT", 1, 24, 8192, true, fp32},
        {"VECTOR_DOT", 1, 31, 8192, true, fp32},
        {"VECTOR_DOT", 1, 32, 8192, true, fp32},
        {"VECTOR_DOT", 1, 33, 8192, true, fp32},
        {"VECTOR_DOT", 1, 40, 8192, true, fp32},
        {"VECTOR_DOT", 1, 47, 8192, true, fp32},
        {"VECTOR_DOT", 1, 48, 8192, true, fp32},
        {"VECTOR_DOT", 1, 63, 8192, true, fp32},
        {"VECTOR_DOT", 1, 64, 8192, true, fp32},
        {"VECTOR_DOT", 2, 17, 16384, true, fp32},
        {"VECTOR_DOT", 2, 18, 16384, true, fp32},
        {"VECTOR_DOT", 2, 23, 16384, true, fp32},
        {"VECTOR_DOT", 2, 32, 16384, true, fp32},
        {"VECTOR_DOT", 2, 48, 16384, true, fp32},
        {"VECTOR_DOT", 3, 17, 16384, true, fp32},
        {"VECTOR_DOT", 3, 18, 16384, true, fp32},
        {"VECTOR_DOT", 3, 20, 16384, true, fp32},
        {"VECTOR_DOT", 3, 24, 16384, true, fp32},
        {"VECTOR_DOT", 3, 32, 16384, true, fp32},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 30, 5120, 32771, true, bf16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 22, 3840, 36864, false, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 12, 3072, 49157, true, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 23, 3328, 32768, false, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 28, 3072, 45056, true, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 11, 3328, 28672, false, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 29, 3072, 32771, true, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 21, 5120, 49152, false, bf16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 9, 3328, 49152, true, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 48, 3328, 65536, true, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 21, 5120, 28672, false, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 27, 4608, 57344, false, bf16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 32, 5120, 49152, false, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 14, 3840, 40960, true, fp16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 14, 5120, 49152, false, bf16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 27, 3840, 32768, true, fp32},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 64, 3328, 36864, false, bf16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 25, 3328, 65521, true, bf16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 8, 3584, 49152, false, bf16},
        {"ADAPTIVE_DETERMINISTIC_SPLIT_K", 13, 5120, 28672, false, bf16},
        {"SMALL_M_WIDE_N", 14, 26880, 15488, false, bf16},
        {"SMALL_M_WIDE_N", 12, 26368, 20480, false, fp16},
        {"SMALL_M_WIDE_N", 13, 31744, 18304, false, fp16},
        {"SMALL_M_WIDE_N", 16, 26880, 19968, false, fp16},
        {"SMALL_M_WIDE_N", 12, 32512, 18176, false, bf16},
        {"SMALL_M_WIDE_N", 8, 32768, 16128, false, fp16},
        {"SMALL_M_WIDE_N", 12, 31232, 16000, false, bf16},
        {"SMALL_M_WIDE_N", 10, 31744, 15232, false, bf16},
        {"SMALL_M_WIDE_N", 13, 25856, 18432, false, bf16},
        {"SMALL_M_WIDE_N", 15, 26624, 17920, false, fp16},
        {"SMALL_M_WIDE_N", 14, 31232, 19072, false, bf16},
        {"SMALL_M_WIDE_N", 9, 31232, 16512, false, fp16},
        {"SMALL_M_WIDE_N", 13, 27136, 18688, false, fp16},
        {"SMALL_M_WIDE_N", 10, 26624, 17152, false, fp16},
        {"SMALL_M_WIDE_N", 11, 32768, 15488, false, bf16},
        {"SMALL_M_WIDE_N", 8, 32256, 20352, false, bf16},
        {"SMALL_M_WIDE_N", 12, 31488, 15104, false, bf16},
        {"SMALL_M_WIDE_N", 10, 26368, 16768, false, fp16},
        {"SMALL_M_WIDE_N", 10, 27392, 16640, false, fp16},
        {"SMALL_M_WIDE_N", 11, 27136, 18176, false, bf16},
        {"TWO_WAVE_N_PANEL", 3808, 48, 15360, false, fp16},
        {"TWO_WAVE_N_PANEL", 3008, 64, 16384, false, fp16},
        {"TWO_WAVE_N_PANEL", 3104, 32, 17920, false, bf16},
        {"TWO_WAVE_N_PANEL", 2976, 80, 27136, false, fp16},
        {"TWO_WAVE_N_PANEL", 3072, 80, 15872, false, fp16},
        {"TWO_WAVE_N_PANEL", 3616, 96, 18944, false, fp16},
        {"TWO_WAVE_N_PANEL", 3008, 48, 19968, false, bf16},
        {"TWO_WAVE_N_PANEL", 3584, 64, 15360, false, fp16},
        {"TWO_WAVE_N_PANEL", 3200, 80, 16896, false, bf16},
        {"TWO_WAVE_N_PANEL", 3840, 32, 24064, false, fp16},
        {"TWO_WAVE_N_PANEL", 3168, 48, 22528, false, fp16},
        {"TWO_WAVE_N_PANEL", 3104, 32, 23040, false, bf16},
        {"TWO_WAVE_N_PANEL", 3616, 48, 19456, false, fp16},
        {"TWO_WAVE_N_PANEL", 2976, 48, 18432, false, bf16},
        {"TWO_WAVE_N_PANEL", 3616, 96, 25600, false, fp16},
        {"TWO_WAVE_N_PANEL", 3616, 80, 20480, false, bf16},
        {"TWO_WAVE_N_PANEL", 3200, 64, 17920, false, fp16},
        {"TWO_WAVE_N_PANEL", 3712, 80, 20992, false, fp16},
        {"TWO_WAVE_N_PANEL", 3008, 32, 24576, false, bf16},
        {"TWO_WAVE_N_PANEL", 2976, 48, 24064, false, bf16},
    };
    for (const Shape &shape : shapes) {
        if (rc != ACL_SUCCESS) break;
        if (std::strcmp(shape.branch, "VECTOR_DOT") == 0) {
            rc = RunVector(shape, stream, function);
        } else {
            rc = RunCube(shape, stream);
        }
    }
    if (rc != ACL_SUCCESS) std::fprintf(stderr, "fatal: comparison failed rc=%d\n", rc);
    if (binary != nullptr) (void)aclrtBinaryUnLoad(binary);
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return rc == ACL_SUCCESS ? 0 : 4;
}
//NEW END
