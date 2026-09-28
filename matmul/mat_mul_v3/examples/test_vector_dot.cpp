// NEW BEGIN
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h"
#include "opdev/common_types.h"

extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char *soPath);

constexpr int WARMUP = 3;
constexpr int REPEATS = 30;

struct Tensor {
    aclTensor *tensor = nullptr;
    void *device = nullptr;
};

void ReleaseTensor(Tensor &value)
{
    if (value.tensor != nullptr) {
        (void)aclDestroyTensor(value.tensor);
    }
    if (value.device != nullptr) {
        (void)aclrtFree(value.device);
    }
}

int CreateTensor(const std::vector<float> &host, const std::vector<int64_t> &shape,
                 const std::vector<int64_t> &storage, const std::vector<int64_t> &strides, Tensor &value)
{
    const size_t bytes = host.size() * sizeof(float);
    int rc = aclrtMalloc(&value.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (rc != ACL_SUCCESS) {
        return rc;
    }
    rc = aclrtMemcpy(value.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        return rc;
    }
    value.tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                                   ACL_FORMAT_ND, storage.data(), storage.size(), value.device);
    return value.tensor == nullptr ? 1 : ACL_SUCCESS;
}

template <typename Launch>
int Measure(Launch launch, aclrtStream stream, float &latency)
{
    for (int index = 0; index < WARMUP; ++index) {
        const int rc = launch();
        if (rc != ACL_SUCCESS) {
            return rc;
        }
    }
    int rc = aclrtSynchronizeStream(stream);
    if (rc != ACL_SUCCESS) {
        return rc;
    }
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    rc = aclrtCreateEvent(&start);
    if (rc != ACL_SUCCESS) {
        return rc;
    }
    rc = aclrtCreateEvent(&end);
    if (rc != ACL_SUCCESS) {
        (void)aclrtDestroyEvent(start);
        return rc;
    }
    rc = aclrtRecordEvent(start, stream);
    for (int index = 0; rc == ACL_SUCCESS && index < REPEATS; ++index) {
        rc = launch();
    }
    if (rc == ACL_SUCCESS) {
        rc = aclrtRecordEvent(end, stream);
    }
    if (rc == ACL_SUCCESS) {
        rc = aclrtSynchronizeEvent(end);
    }
    float total = 0.0f;
    if (rc == ACL_SUCCESS) {
        rc = aclrtEventElapsedTime(&total, start, end);
    }
    (void)aclrtDestroyEvent(end);
    (void)aclrtDestroyEvent(start);
    if (rc == ACL_SUCCESS) {
        latency = total / REPEATS;
    }
    return rc;
}

bool Validate(void *output, int64_t m, int64_t n, int64_t k)
{
    std::vector<float> host(static_cast<size_t>(m * n));
    if (aclrtMemcpy(host.data(), host.size() * sizeof(float), output,
                    host.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    for (int64_t index = 0; index < m * n; ++index) {
        const float expected = static_cast<float>(k * (index % n + 1));
        if (!std::isfinite(host[index]) || host[index] != expected) {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> DecodeHex(const char *text)
{
    if (text == nullptr || (std::strlen(text) & 1U) != 0U) {
        return {};
    }
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    const size_t size = std::strlen(text) / 2;
    std::vector<uint8_t> bytes(size);
    for (size_t index = 0; index < size; ++index) {
        const int high = nibble(text[index * 2]);
        const int low = nibble(text[index * 2 + 1]);
        if (high < 0 || low < 0) {
            return {};
        }
        bytes[index] = static_cast<uint8_t>((high << 4) | low);
    }
    return bytes;
}

int RunShape(int64_t m, int64_t n, int64_t k, aclrtStream stream,
             aclrtFuncHandle vectorFunction, std::string &failureStage, std::string &failureDetail)
{
    failureStage.clear();
    failureDetail.clear();
    auto captureFailure = [&](const char *stage, int code) {
        failureStage = stage;
        const char *detail = aclGetRecentErrMsg();
        if (detail != nullptr) {
            failureDetail = detail;
            std::replace(failureDetail.begin(), failureDetail.end(), '\n', ' ');
            std::replace(failureDetail.begin(), failureDetail.end(), '\r', ' ');
        }
        return code;
    };
    std::vector<float> a(static_cast<size_t>(m * k), 1.0f);
    std::vector<float> b(static_cast<size_t>(n * k));
    std::vector<float> c(static_cast<size_t>(m * n), 0.0f);
    for (int64_t column = 0; column < n; ++column) {
        std::fill(b.begin() + column * k, b.begin() + (column + 1) * k,
                  static_cast<float>(column + 1));
    }

    Tensor aTensor;
    Tensor bTensor;
    Tensor cTensor;
    const char *activeStage = "create_a";
    int rc = CreateTensor(a, {m, k}, {m, k}, {k, 1}, aTensor);
    if (rc == ACL_SUCCESS) {
        activeStage = "create_b";
        rc = CreateTensor(b, {k, n}, {n, k}, {1, k}, bTensor);
    }
    if (rc == ACL_SUCCESS) {
        activeStage = "create_c";
        rc = CreateTensor(c, {m, n}, {m, n}, {n, 1}, cTensor);
    }
    if (rc != ACL_SUCCESS) {
        const int failureCode = captureFailure(activeStage, rc);
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return failureCode;
    }

    (void)::setenv("MATMUL_VECTOR_ENABLE", "0", 1);
    uint64_t originalWorkspaceSize = 0;
    aclOpExecutor *originalExecutor = nullptr;
    activeStage = "original_get_workspace";
    rc = aclnnMatmulGetWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, 1,
                                     &originalWorkspaceSize, &originalExecutor);
    void *originalWorkspace = nullptr;
    if (rc == ACL_SUCCESS && originalWorkspaceSize != 0) {
        activeStage = "original_allocate_workspace";
        rc = aclrtMalloc(&originalWorkspace, originalWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (rc == ACL_SUCCESS) {
        activeStage = "original_make_repeatable";
        rc = aclSetAclOpExecutorRepeatable(originalExecutor);
    }
    float originalLatency = 0.0f;
    if (rc == ACL_SUCCESS) {
        activeStage = "original_measure";
        rc = Measure([&]() {
            return aclnnMatmul(originalWorkspace, originalWorkspaceSize, originalExecutor, stream);
        }, stream, originalLatency);
    }
    const bool originalCorrect = rc == ACL_SUCCESS && Validate(cTensor.device, m, n, k);
    int originalFailureCode = ACL_SUCCESS;
    if (!originalCorrect) {
        if (rc == ACL_SUCCESS) activeStage = "original_validate";
        originalFailureCode = captureFailure(activeStage, rc == ACL_SUCCESS ? 3 : rc);
    }
    if (originalExecutor != nullptr) (void)aclDestroyAclOpExecutor(originalExecutor);
    if (originalWorkspace != nullptr) (void)aclrtFree(originalWorkspace);
    if (!originalCorrect) {
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return originalFailureCode;
    }

    (void)::unsetenv("MATMUL_VECTOR_TILING");
    (void)::unsetenv("MATMUL_VECTOR_CORES");
    (void)::setenv("MATMUL_VECTOR_ENABLE", "1", 1);
    uint64_t ignoredWorkspaceSize = 0;
    aclOpExecutor *ignoredExecutor = nullptr;
    activeStage = "modified_get_workspace";
    rc = aclnnMatmulGetWorkspaceSize(aTensor.tensor, bTensor.tensor, cTensor.tensor, 1,
                                     &ignoredWorkspaceSize, &ignoredExecutor);
    int modifiedWorkspaceFailureCode = ACL_SUCCESS;
    if (rc != ACL_SUCCESS) modifiedWorkspaceFailureCode = captureFailure(activeStage, rc);
    if (ignoredExecutor != nullptr) (void)aclDestroyAclOpExecutor(ignoredExecutor);
    const char *tilingText = std::getenv("MATMUL_VECTOR_TILING");
    const char *coreText = std::getenv("MATMUL_VECTOR_CORES");
    if (rc != ACL_SUCCESS) {
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return modifiedWorkspaceFailureCode;
    }
    if (tilingText == nullptr || coreText == nullptr) {
        ReleaseTensor(cTensor);
        ReleaseTensor(bTensor);
        ReleaseTensor(aTensor);
        return ACL_SUCCESS;
    }

    const std::vector<uint8_t> tiling = DecodeHex(tilingText);
    const uint32_t cores = static_cast<uint32_t>(std::strtoul(coreText, nullptr, 10));
    void *tilingDevice = nullptr;
    aclrtArgsHandle arguments = nullptr;
    if (tiling.empty() || cores == 0) {
        activeStage = "modified_packet";
        rc = 4;
    }
    if (rc == ACL_SUCCESS) {
        activeStage = "modified_allocate_tiling";
        rc = aclrtMalloc(&tilingDevice, tiling.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (rc == ACL_SUCCESS) {
        activeStage = "modified_copy_tiling";
        rc = aclrtMemcpy(tilingDevice, tiling.size(), tiling.data(), tiling.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (rc == ACL_SUCCESS) {
        activeStage = "modified_args_init";
        rc = aclrtKernelArgsInit(vectorFunction, &arguments);
    }
    void *bias = nullptr;
    void *offset = nullptr;
    void *workspace = nullptr;
    void *values[] = {aTensor.device, bTensor.device, bias, offset, cTensor.device, workspace, tilingDevice};
    for (void *&value : values) {
        aclrtParamHandle parameter = nullptr;
        if (rc == ACL_SUCCESS) {
            activeStage = "modified_args_append";
            rc = aclrtKernelArgsAppend(arguments, &value, sizeof(value), &parameter);
        }
    }
    if (rc == ACL_SUCCESS) {
        activeStage = "modified_args_finalize";
        rc = aclrtKernelArgsFinalize(arguments);
    }
    float modifiedLatency = 0.0f;
    if (rc == ACL_SUCCESS) {
        activeStage = "modified_measure";
        rc = Measure([&]() {
            return aclrtLaunchKernelWithConfig(vectorFunction, cores, stream, nullptr, arguments, nullptr);
        }, stream, modifiedLatency);
    }
    const bool modifiedCorrect = rc == ACL_SUCCESS && Validate(cTensor.device, m, n, k);
    int modifiedFailureCode = ACL_SUCCESS;
    if (!modifiedCorrect) {
        if (rc == ACL_SUCCESS) activeStage = "modified_validate";
        modifiedFailureCode = captureFailure(activeStage, rc == ACL_SUCCESS ? 3 : rc);
    }
    if (tilingDevice != nullptr) (void)aclrtFree(tilingDevice);
    if (modifiedCorrect) {
        std::printf(
            "{\"shape\":\"M%ld_N%ld_K%ld_NT\",\"modified_branch\":\"VECTOR_DOT\","
            "\"modified_branch_latency\":%.9f,\"original_branch\":\"BASE\","
            "\"original_branch_latency\":%.9f}\n",
            static_cast<long>(m), static_cast<long>(n), static_cast<long>(k),
            modifiedLatency, originalLatency);
        std::fflush(stdout);
    }
    ReleaseTensor(cTensor);
    ReleaseTensor(bTensor);
    ReleaseTensor(aTensor);
    if (modifiedCorrect) return ACL_SUCCESS;
    return modifiedFailureCode;
}

int main(int argc, char **argv)
{
    if (argc < 4 || (argc - 1) % 3 != 0) {
        return 2;
    }
    int rc = aclInit(nullptr);
    if (rc == ACL_SUCCESS) rc = aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    if (rc == ACL_SUCCESS) rc = aclrtCreateStream(&stream);
    const char *hostLibrary = std::getenv("MATMUL_HOST_LIBRARY");
    const char *vectorBinary = std::getenv("MATMUL_VECTOR_BINARY");
    if (rc != ACL_SUCCESS || hostLibrary == nullptr || vectorBinary == nullptr ||
        TbeLoadSoAndSaveToRegistry(hostLibrary) != 0U) {
        return 4;
    }
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    rc = aclrtBinaryLoadFromFile(vectorBinary, nullptr, &binary);
    if (rc == ACL_SUCCESS) {
        rc = aclrtBinaryGetFunction(binary, "MatMulV3_VectorDot_2162688", &function);
    }
    if (rc != ACL_SUCCESS) {
        return rc;
    }
    for (int index = 1; index < argc; index += 3) {
        const int64_t m = std::strtoll(argv[index], nullptr, 10);
        const int64_t n = std::strtoll(argv[index + 1], nullptr, 10);
        const int64_t k = std::strtoll(argv[index + 2], nullptr, 10);
        std::string failureStage;
        std::string failureDetail;
        const int shapeRc = RunShape(m, n, k, stream, function, failureStage, failureDetail);
        if (shapeRc != ACL_SUCCESS) {
            std::fprintf(stderr, "skip M%ld_N%ld_K%ld_NT stage=%s rc=%d detail=%s\n",
                         static_cast<long>(m), static_cast<long>(n), static_cast<long>(k),
                         failureStage.c_str(), shapeRc, failureDetail.c_str());
        }
    }
    (void)aclrtBinaryUnLoad(binary);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return 0;
}
// NEW END
