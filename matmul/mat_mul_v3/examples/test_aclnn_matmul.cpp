/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
//NEW
#include <algorithm>
#include <cmath> //NEW
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring> //NEW
#include <dlfcn.h> //NEW
#include <sstream> //NEW
#include <string>
#include "acl/acl.h"
#include "exe_graph/runtime/tensor.h" //NEW
#include "matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h"
#include "opdev/common_types.h" //NEW
#include "opdev/platform.h" //NEW

#ifdef CHECK_RET
#undef CHECK_RET
#endif
#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

//NEW
aclDataType SelectedAclDataType() {
  const char* value = std::getenv("MATMUL_DATA_TYPE");
  if (value != nullptr && std::strcmp(value, "fp16") == 0) {
    return aclDataType::ACL_FLOAT16;
  }
  if (value != nullptr && std::strcmp(value, "bf16") == 0) {
    return aclDataType::ACL_BF16;
  }
  return aclDataType::ACL_FLOAT;
}

//NEW
aclDataType SelectedOutputAclDataType() {
  const char* value = std::getenv("MATMUL_OUTPUT_DATA_TYPE");
  if (value != nullptr && std::strcmp(value, "fp16") == 0) {
    return aclDataType::ACL_FLOAT16;
  }
  if (value != nullptr && std::strcmp(value, "bf16") == 0) {
    return aclDataType::ACL_BF16;
  }
  if (value != nullptr && std::strcmp(value, "fp32") == 0) {
    return aclDataType::ACL_FLOAT;
  }
  return SelectedAclDataType();
}

//NEW
ge::DataType SelectedGeDataType() {
  switch (SelectedAclDataType()) {
    case aclDataType::ACL_FLOAT16:
      return ge::DT_FLOAT16;
    case aclDataType::ACL_BF16:
      return ge::DT_BF16;
    default:
      return ge::DT_FLOAT;
  }
}

//NEW
const char* SelectedDataTypeName() {
  switch (SelectedAclDataType()) {
    case aclDataType::ACL_FLOAT16:
      return "fp16";
    case aclDataType::ACL_BF16:
      return "bf16";
    default:
      return "fp32";
  }
}

//NEW
const char* SelectedOutputDataTypeName() {
  switch (SelectedOutputAclDataType()) {
    case aclDataType::ACL_FLOAT16:
      return "fp16";
    case aclDataType::ACL_BF16:
      return "bf16";
    default:
      return "fp32";
  }
}

//NEW
size_t SelectedElementSize() {
  return SelectedAclDataType() == aclDataType::ACL_FLOAT ? sizeof(float) : sizeof(uint16_t);
}

//NEW
size_t SelectedOutputElementSize() {
  return SelectedOutputAclDataType() == aclDataType::ACL_FLOAT ? sizeof(float) : sizeof(uint16_t);
}

//NEW
void FillOnes(std::vector<uint8_t>* bytes) {
  if (SelectedAclDataType() == aclDataType::ACL_FLOAT) {
    const float one = 1.0F;
    for (size_t offset = 0; offset < bytes->size(); offset += sizeof(one)) {
      std::memcpy(bytes->data() + offset, &one, sizeof(one));
    }
    return;
  }
  const uint16_t one = SelectedAclDataType() == aclDataType::ACL_FLOAT16 ? 0x3c00U : 0x3f80U;
  for (size_t offset = 0; offset < bytes->size(); offset += sizeof(one)) {
    std::memcpy(bytes->data() + offset, &one, sizeof(one));
  }
}

//NEW
float HalfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
  uint32_t exponent = (value >> 10U) & 0x1fU;
  uint32_t mantissa = value & 0x03ffU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        ++shift;
      }
      mantissa &= 0x03ffU;
      bits = sign | static_cast<uint32_t>(127 - 14 - shift) << 23U | mantissa << 13U;
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | mantissa << 13U;
  } else {
    bits = sign | (exponent + 112U) << 23U | mantissa << 13U;
  }
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

//NEW
float DecodeOutputValue(const uint8_t* bytes) {
  if (SelectedOutputAclDataType() == aclDataType::ACL_FLOAT) {
    float value = 0.0F;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
  }
  uint16_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  if (SelectedOutputAclDataType() == aclDataType::ACL_FLOAT16) {
    return HalfToFloat(value);
  }
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, return ACL_ERROR_INVALID_PARAM);
  return 0;
}

//NEW
template <typename T>
int CreateTransposedAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& logicalShape,
                              const std::vector<int64_t>& storageShape, void** deviceAddr,
                              aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(storageShape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<int64_t> strides = {1, logicalShape[0]};
  *tensor = aclCreateTensor(logicalShape.data(), logicalShape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, return ACL_ERROR_INVALID_PARAM);
  return 0;
}

//NEW
std::vector<int64_t> TensorStrides(const std::vector<int64_t>& logicalShape, bool transposed) {
  if (transposed) {
    return {1, logicalShape[0]};
  }
  return {logicalShape[1], 1};
}

//NEW
int CreateRawAclTensor(const std::vector<uint8_t>& hostData, const std::vector<int64_t>& logicalShape,
                       const std::vector<int64_t>& storageShape, bool transposed, bool metadataOnly,
                       aclDataType dataType, size_t elementSize, void** deviceAddr, aclTensor** tensor) {
  *deviceAddr = nullptr;
  if (metadataOnly) {
    //NEW
    // Some ACL releases reject a tensor whose data address is null even when
    // only shape inference and tiling are requested. One element is enough;
    // discovery never launches a kernel or reads tensor contents.
    auto ret = aclrtMalloc(deviceAddr, elementSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  } else {
    auto ret = aclrtMalloc(deviceAddr, hostData.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, hostData.size(), hostData.data(), hostData.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  const auto strides = TensorStrides(logicalShape, transposed);
  *tensor = aclCreateTensor(logicalShape.data(), logicalShape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, storageShape.data(), storageShape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, return ACL_ERROR_INVALID_PARAM);
  return ACL_SUCCESS;
}

//NEW
std::string ReadSelectedBranch() {
  const char* commonBranch = std::getenv("MATMUL_SELECTED_BRANCH");
  if (commonBranch != nullptr && commonBranch[0] != '\0') {
    return commonBranch;
  }
  const char* selectedBranch = std::getenv("MATMUL_V3_SELECTED_BRANCH");
  if (selectedBranch != nullptr && selectedBranch[0] != '\0') {
    return selectedBranch;
  }
  const char* v3Only = std::getenv("MATMUL_V3_ONLY");
  if (v3Only != nullptr && v3Only[0] == '1' && v3Only[1] == '\0') {
    return "";
  }
  const char* shrinkMode = std::getenv("MATMUL_SHRINK_MODE");
  return shrinkMode != nullptr && shrinkMode[0] == '1' && shrinkMode[1] == '\0' ? "" : "OFFICIAL_BASELINE";
}

//NEW
std::string ReadEnvironment(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? "" : value;
}

//NEW
uint32_t ReadEnvironmentUint(const char* name, uint32_t defaultValue = 0) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return defaultValue;
  }
  return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}

//NEW
bool UseTransposedB() {
  const char* value = std::getenv("MATMUL_B_TRANSPOSE");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

//NEW
bool UseTransposedA() {
  const char* value = std::getenv("MATMUL_A_TRANSPOSE");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

//NEW
bool DiscoveryOnly() {
  const char* value = std::getenv("MATMUL_V3_DISCOVERY_ONLY");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

//NEW
void ClearSelectedBranch() {
  (void)::unsetenv("MATMUL_SELECTED_BRANCH");
  (void)::unsetenv("MATMUL_V3_SELECTED_BRANCH");
  (void)::unsetenv("MATMUL_SHRINK_EFFECTIVE");
  (void)::unsetenv("MATMUL_SHRINK_OLD_CORES");
  (void)::unsetenv("MATMUL_SHRINK_NEW_CORES");
  (void)::unsetenv("MATMUL_TILING_STAGE");
  (void)::unsetenv("MATMUL_V3_OFFICIAL_CORE");
  (void)::unsetenv("MATMUL_V3_REQUESTED_CORE");
  (void)::unsetenv("MATMUL_V3_ACTUAL_CORE");
  (void)::unsetenv("MATMUL_V3_TILING_JSON");
}

//NEW
int SelectOfficialMatMulV3Route(int64_t m, int64_t n, int64_t k, bool transposeA, bool transposeB,
                               std::string* failureDetail) {
  const auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == op::SocVersion::ASCEND910_95) {
    return 1;
  }
  if (socVersion != op::SocVersion::ASCEND910B && socVersion != op::SocVersion::ASCEND910_93) {
    return 0;
  }

  using Selector = bool (*)(const gert::Tensor*, const gert::Tensor*, const gert::Tensor*, bool, bool,
                            op::Format, bool, uint32_t, const std::string&);
  static void* selectorHandle = nullptr;
  static Selector selector = nullptr;
  static bool selectorLoadAttempted = false;
  if (!selectorLoadAttempted) {
    selectorLoadAttempted = true;
    const char* selectorLibrary = std::getenv("MATMUL_LEGACY_COMMON_LIBRARY");
    if (selectorLibrary != nullptr && selectorLibrary[0] != '\0') {
      selectorHandle = dlopen(selectorLibrary, RTLD_LAZY | RTLD_LOCAL);
      if (selectorHandle != nullptr) {
        selector = reinterpret_cast<Selector>(dlsym(selectorHandle, "LegacyMmCheckHitV3Shape"));
      }
    }
  }
  if (selector == nullptr) {
    const char* loaderError = dlerror();
    *failureDetail = loaderError == nullptr ? "official MatMul V3 selector is unavailable" : loaderError;
    return -1;
  }

  gert::Tensor selfTensor;
  selfTensor.MutableOriginShape() = transposeA ? gert::Shape({k, m}) : gert::Shape({m, k});
  selfTensor.MutableStorageShape() = transposeA ? gert::Shape({k, m}) : gert::Shape({m, k});
  selfTensor.SetOriginFormat(ge::FORMAT_ND);
  selfTensor.SetStorageFormat(ge::FORMAT_ND);
  selfTensor.SetDataType(SelectedGeDataType());

  gert::Tensor mat2Tensor;
  //NEW
  // ContiguousAndCast swaps an NT {K,N} view to {N,K} before calling the
  // selector. NN remains {K,N}.
  if (transposeB) {
    mat2Tensor.MutableOriginShape() = gert::Shape({n, k});
    mat2Tensor.MutableStorageShape() = gert::Shape({n, k});
  } else {
    mat2Tensor.MutableOriginShape() = gert::Shape({k, n});
    mat2Tensor.MutableStorageShape() = gert::Shape({k, n});
  }
  mat2Tensor.SetOriginFormat(ge::FORMAT_ND);
  mat2Tensor.SetStorageFormat(ge::FORMAT_ND);
  mat2Tensor.SetDataType(SelectedGeDataType());

  //NEW
  return selector(&selfTensor, &mat2Tensor, nullptr, transposeA, transposeB, ge::FORMAT_ND, false,
                  op::GetCurrentPlatformInfo().GetCubeCoreNum(),
                  op::GetCurrentPlatformInfo().GetSocLongVersion()) ? 1 : 0;
}

//NEW
extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char* soPath);

//NEW
int EnableMatMulV3Host() {
  const char* v3LibraryPath = std::getenv("MATMUL_V3_HOST_LIBRARY");
  if (v3LibraryPath == nullptr || v3LibraryPath[0] == '\0') {
    fprintf(stderr, "tiling registration failed: MatMulV3 host library path is missing\n");
    return 4;
  }

  //NEW
  // V2 inputs are rejected by the official dispatcher predicate before an
  // executor is built. Only the matching local MatMulV3 host is registered.
  const uint32_t v3Status = TbeLoadSoAndSaveToRegistry(v3LibraryPath);
  if (v3Status != 0U) {
    fprintf(stderr, "tiling registration failed: cannot register MatMulV3 host library rc=%u\n", v3Status);
    return 4;
  }
  return ACL_SUCCESS;
}

//NEW
int MeasureShape(int64_t m, int64_t n, int64_t k, aclrtStream stream, float* averageMs,
                 std::string* branch, uint64_t* workspaceBytes, std::string* tilingJson,
                 uint32_t* officialCore, uint32_t* requestedCore, uint32_t* actualCore,
                 bool* timingComplete, std::string* failedStage, std::string* failureDetail) {
  constexpr int kSkipNotMatMulV3 = 10001;
  const bool transposeA = UseTransposedA();
  const bool transposeB = UseTransposedB();
  //NEW
  // Reject official MatMulV2 routes before allocating or copying any tensor.
  const char* v3Only = std::getenv("MATMUL_V3_ONLY");
  if (v3Only != nullptr && v3Only[0] == '1' && v3Only[1] == '\0') {
    const int selectedRoute = SelectOfficialMatMulV3Route(m, n, k, transposeA, transposeB, failureDetail);
    if (selectedRoute < 0) {
      *failedStage = "official_v3_route_selection";
      return 4;
    }
    if (selectedRoute == 0) {
      failedStage->clear();
      return kSkipNotMatMulV3;
    }
  }

  auto ret = ACL_SUCCESS;
  std::vector<int64_t> selfShape = {m, k};
  //NEW
  std::vector<int64_t> selfStorageShape = transposeA ? std::vector<int64_t>{k, m} : selfShape;
  std::vector<int64_t> mat2Shape = {k, n};
  //NEW
  std::vector<int64_t> mat2StorageShape = transposeB ? std::vector<int64_t>{n, k} : mat2Shape;
  std::vector<int64_t> outShape = {m, n};
  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  //NEW
  const bool metadataOnly = DiscoveryOnly();
  const size_t inputElementSize = SelectedElementSize();
  const size_t outputElementSize = SelectedOutputElementSize();
  std::vector<uint8_t> selfHostData;
  std::vector<uint8_t> mat2HostData;
  std::vector<uint8_t> outHostData;
  if (!metadataOnly) {
    selfHostData.resize(static_cast<size_t>(GetShapeSize(selfStorageShape)) * inputElementSize);
    mat2HostData.resize(static_cast<size_t>(GetShapeSize(mat2StorageShape)) * inputElementSize);
    outHostData.resize(static_cast<size_t>(GetShapeSize(outShape)) * outputElementSize, 0);
    FillOnes(&selfHostData);
    FillOnes(&mat2HostData);
  }
  // 创建self aclTensor
  *failedStage = "create_self";
  ret = CreateRawAclTensor(selfHostData, selfShape, selfStorageShape, transposeA, metadataOnly,
                           SelectedAclDataType(), inputElementSize, &selfDeviceAddr, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat2 aclTensor
  *failedStage = "create_mat2";
  //NEW
  ret = CreateRawAclTensor(mat2HostData, mat2Shape, mat2StorageShape, transposeB, metadataOnly,
                           SelectedAclDataType(), inputElementSize, &mat2DeviceAddr, &mat2);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  *failedStage = "create_out";
  ret = CreateRawAclTensor(outHostData, outShape, outShape, false, metadataOnly,
                           SelectedOutputAclDataType(), outputElementSize, &outDeviceAddr, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::unique_ptr<void, aclError (*)(void*)> executorAddrPtr(nullptr, aclrtFree);
  // 调用aclnnMatmul第一段接口
  *failedStage = "get_workspace";
  ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    const char* recentError = aclGetRecentErrMsg();
    if (recentError != nullptr) {
      *failureDetail = recentError;
      std::replace(failureDetail->begin(), failureDetail->end(), '\n', ' ');
      std::replace(failureDetail->begin(), failureDetail->end(), '\r', ' ');
    }
    return ret;
  }
  //NEW
  *branch = ReadSelectedBranch();
  *workspaceBytes = workspaceSize;
  *tilingJson = ReadEnvironment("MATMUL_V3_TILING_JSON");
  *officialCore = ReadEnvironmentUint("MATMUL_V3_OFFICIAL_CORE");
  *requestedCore = ReadEnvironmentUint("MATMUL_V3_REQUESTED_CORE");
  *actualCore = ReadEnvironmentUint("MATMUL_V3_ACTUAL_CORE");
  if (v3Only != nullptr && v3Only[0] == '1' && v3Only[1] == '\0' && branch->empty()) {
    (void)aclDestroyAclOpExecutor(executor);
    executor = nullptr;
    *failedStage = "v3_tiling_callback_invariant";
    *failureDetail = "official dispatcher selected MatMulV3 but the local V3 tiler was not entered";
    return 4;
  }
  //NEW
  // Discovery performs the real official dispatcher and local V3 tiling
  // callback, but deliberately stops before allocating workspace or launching
  // a kernel. This makes large candidate pools cheap and memory-safe.
  if (metadataOnly) {
    (void)aclDestroyAclOpExecutor(executor);
    failedStage->clear();
    return ACL_SUCCESS;
  }
  //NEW
  std::unique_ptr<aclOpExecutor, aclnnStatus (*)(aclOpExecutor*)> executorPtr(
      executor, aclDestroyAclOpExecutor);
  *failedStage = "make_executor_repeatable";
  ret = aclSetAclOpExecutorRepeatable(executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclSetAclOpExecutorRepeatable failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    *failedStage = "allocate_workspace";
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    executorAddrPtr.reset(workspaceAddr);
  }
  //NEW
  // // 调用aclnnMatmul第二段接口
  // ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
  //
  // // 4. （固定写法）同步等待任务执行结束
  // ret = aclrtSynchronizeStream(stream);
  // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  //NEW
  //NEW: Keep the sampling depth configurable so a broad 30-combination
  // campaign does not spend 110 launches on every one of 25,500 points.
  // The defaults preserve the original standalone runner behaviour.
  const int warmup = static_cast<int>(ReadEnvironmentUint("MATMUL_V3_WARMUP", 10));
  const int repeat = static_cast<int>(ReadEnvironmentUint("MATMUL_V3_REPEATS", 100));
  CHECK_RET(warmup >= 0 && repeat > 0, return 2);

  *failedStage = "warmup_submit";
  for (int i = 0; i < warmup; ++i) {
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *failedStage = "warmup_sync";
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclrtEvent startEvent = nullptr;
  aclrtEvent endEvent = nullptr;
  *failedStage = "create_start_event";
  ret = aclrtCreateEvent(&startEvent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  *failedStage = "create_end_event";
  ret = aclrtCreateEvent(&endEvent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  *failedStage = "record_start_event";
  ret = aclrtRecordEvent(startEvent, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  *failedStage = "timed_submit";
  for (int i = 0; i < repeat; ++i) {
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  *failedStage = "record_end_event";
  ret = aclrtRecordEvent(endEvent, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  *failedStage = "timed_sync";
  ret = aclrtSynchronizeEvent(endEvent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  float totalMs = 0.0F;
  *failedStage = "elapsed_time";
  ret = aclrtEventElapsedTime(&totalMs, startEvent, endEvent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  *averageMs = totalMs / repeat;
  *timingComplete = true;

  aclrtDestroyEvent(endEvent);
  aclrtDestroyEvent(startEvent);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  //NEW
  std::vector<uint8_t> resultData(static_cast<size_t>(size) * outputElementSize, 0);
  *failedStage = "copy_output";
  ret = aclrtMemcpy(resultData.data(), resultData.size(), outDeviceAddr,
                    resultData.size(), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  //NEW
  *failedStage = "validate_output";
  for (int64_t i = 0; i < size; i++) {
    const float value = DecodeOutputValue(resultData.data() + static_cast<size_t>(i) * outputElementSize);
    const float expected = static_cast<float>(k);
    const float tolerance = SelectedOutputAclDataType() == aclDataType::ACL_FLOAT ? 0.0F :
        std::max(2.0F, std::fabs(expected) * 0.02F);
    CHECK_RET(std::isfinite(value) && std::fabs(value - expected) <= tolerance, return 3);
  }

  failedStage->clear();
  return 0;
}

//NEW
struct MeasurementResult {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  float averageMs = 0.0F;
  std::string branch;
  uint64_t workspaceBytes = 0;
  std::string tilingJson;
  uint32_t officialCore = 0;
  uint32_t requestedCore = 0;
  uint32_t actualCore = 0;
  bool timingComplete = false;
  int resultCode = 0;
  std::string failedStage;
  bool complete = false;
};

//NEW
void PrintMeasurementResult(const MeasurementResult& result) {
  const bool transposeA = UseTransposedA();
  const bool transposeB = UseTransposedB();
  const std::string layout = std::string(transposeA ? "T" : "N") + (transposeB ? "T" : "N");
  const std::string mode = ReadEnvironment("MATMUL_V3_MEASUREMENT_MODE").empty() ?
      "official" : ReadEnvironment("MATMUL_V3_MEASUREMENT_MODE");
  const std::string soc = op::GetCurrentPlatformInfo().GetSocLongVersion();
  const bool discovery = DiscoveryOnly();
  const char* status = result.resultCode == ACL_SUCCESS ? (discovery ? "DISCOVERED" : "OK") :
      (result.resultCode == 3 ? "INVALID_OUTPUT" : "ERROR");
  const char* correctness = result.resultCode == ACL_SUCCESS ? (discovery ? "NOT_RUN" : "PASS") :
      (result.resultCode == 3 ? "FAIL" : "NOT_CHECKED");
  const std::string requestedCore = result.requestedCore == 0 ? "null" : std::to_string(result.requestedCore);
  char latencyText[64] = {};
  if (result.timingComplete) {
    (void)snprintf(latencyText, sizeof(latencyText), "%.9f", result.averageMs);
  }
  const std::string latency = result.timingComplete ? latencyText : "null";
  const std::string tiling = result.tilingJson.empty() ? "null" : result.tilingJson;
  const std::string failedStage = result.failedStage.empty() ? "null" : "\"" + result.failedStage + "\"";
  const std::string branch = result.branch.empty() ? "UNKNOWN" : result.branch;
  LOG_PRINT(
      "{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"dtype\":\"%s\","
      "\"input_dtype\":\"%s\",\"output_dtype\":\"%s\",\"layout\":\"%s\","
      "\"soc\":\"%s\",\"branch\":\"%s\",\"mode\":\"%s\","
      "\"experiment\":\"fixed_tiling_core_sweep\","
      "\"override_scope\":\"used_core_num_block_dim_workspace\",\"requested_core\":%s,"
      "\"official_core\":%u,\"actual_core\":%u,\"latency_ms\":%s,"
      "\"workspace_bytes\":%llu,\"warmup\":%d,\"repeats\":%d,"
      "\"status\":\"%s\",\"correctness\":\"%s\",\"failure_stage\":%s,\"tiling\":%s}\n",
      static_cast<long>(result.m), static_cast<long>(result.n), static_cast<long>(result.k),
      layout.c_str(), SelectedDataTypeName(), SelectedDataTypeName(), SelectedOutputDataTypeName(),
      layout.c_str(), soc.c_str(), branch.c_str(), mode.c_str(),
      requestedCore.c_str(), result.officialCore, result.actualCore, latency.c_str(),
      static_cast<unsigned long long>(result.workspaceBytes),
      discovery ? 0 : static_cast<int>(ReadEnvironmentUint("MATMUL_V3_WARMUP", 10)),
      discovery ? 0 : static_cast<int>(ReadEnvironmentUint("MATMUL_V3_REPEATS", 100)),
      status, correctness,
      failedStage.c_str(), tiling.c_str());
}

//NEW
int main(int argc, char** argv) {
  constexpr int kSkipNotMatMulV3 = 10001;
  CHECK_RET(argc >= 4 && (argc - 1) % 3 == 0, return 2);

  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  //NEW
  std::vector<MeasurementResult> results;
  for (int arg = 1; arg < argc; arg += 3) {
    const int64_t m = std::strtoll(argv[arg], nullptr, 10);
    const int64_t n = std::strtoll(argv[arg + 1], nullptr, 10);
    const int64_t k = std::strtoll(argv[arg + 2], nullptr, 10);
    if (m <= 0 || n <= 0 || k <= 0) {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 2;
    }
    MeasurementResult result;
    result.m = m;
    result.n = n;
    result.k = k;
    results.push_back(result);
  }

  //NEW
  const char* v3Only = std::getenv("MATMUL_V3_ONLY");
  const bool v3OnlyEnabled = v3Only != nullptr && v3Only[0] == '1' && v3Only[1] == '\0';
  if (v3OnlyEnabled) {
    ret = EnableMatMulV3Host();
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("MatMulV3 host setup failed. ERROR: %d\n", ret); return ret);
  }

  //NEW
  // A measurement plan keeps one shape on one initialized ACL stream while
  // completing its entire core-response curve.  An empty plan preserves the
  // original single-mode runner used by discovery.
  std::vector<std::string> measurementPlan;
  const char* planText = std::getenv("MATMUL_V3_MEASUREMENT_PLAN");
  if (planText != nullptr && planText[0] != '\0') {
    std::stringstream planStream(planText);
    std::string token;
    while (std::getline(planStream, token, ',')) {
      if (token.empty()) {
        continue;
      }
      if (token != "official_pre" && token != "official_post" &&
          token != "original" && token != "shrink" &&
          token != "shrink_pre" && token != "shrink_post" &&
          token != "wave_exact_pre" && token != "wave_exact_post") {
        char* end = nullptr;
        const unsigned long core = std::strtoul(token.c_str(), &end, 10);
        if (end == token.c_str() || *end != '\0' || core < 4UL || core > 20UL) {
          LOG_PRINT("invalid measurement plan token: %s\n", token.c_str());
          aclrtDestroyStream(stream);
          aclrtResetDevice(deviceId);
          aclFinalize();
          return 2;
        }
      }
      measurementPlan.push_back(token);
    }
  }
  if (measurementPlan.empty()) {
    measurementPlan.push_back("");
  }

  for (size_t index = 0; index < results.size(); ++index) {
    const auto seed = results[index];
    for (const auto& planItem : measurementPlan) {
      //NEW
      if (planItem == "official_pre" || planItem == "official_post" || planItem == "original") {
        (void)::setenv("MATMUL_V3_MEASUREMENT_MODE", planItem.c_str(), 1);
        (void)::unsetenv("MATMUL_V3_FORCE_CORE_NUM");
        (void)::setenv("MATMUL_V3_SHRINK_IDLE_CORES", "0", 1);
        (void)::setenv("MATMUL_V3_ENABLE_WAVE_EXACT_ND2NZ", "0", 1);
      } else if (planItem == "shrink" || planItem == "shrink_pre" || planItem == "shrink_post") {
        (void)::setenv("MATMUL_V3_MEASUREMENT_MODE", planItem.c_str(), 1);
        (void)::unsetenv("MATMUL_V3_FORCE_CORE_NUM");
        (void)::setenv("MATMUL_V3_SHRINK_IDLE_CORES", "1", 1);
        (void)::setenv("MATMUL_V3_ENABLE_WAVE_EXACT_ND2NZ", "0", 1);
      } else if (planItem == "wave_exact_pre" || planItem == "wave_exact_post") {
        (void)::setenv("MATMUL_V3_MEASUREMENT_MODE", planItem.c_str(), 1);
        (void)::unsetenv("MATMUL_V3_FORCE_CORE_NUM");
        (void)::setenv("MATMUL_V3_SHRINK_IDLE_CORES", "0", 1);
        (void)::setenv("MATMUL_V3_ENABLE_WAVE_EXACT_ND2NZ", "1", 1);
      } else if (!planItem.empty()) {
        (void)::setenv("MATMUL_V3_MEASUREMENT_MODE", "core_sweep", 1);
        (void)::setenv("MATMUL_V3_FORCE_CORE_NUM", planItem.c_str(), 1);
        (void)::setenv("MATMUL_V3_SHRINK_IDLE_CORES", "0", 1);
        (void)::setenv("MATMUL_V3_ENABLE_WAVE_EXACT_ND2NZ", "0", 1);
      }

      MeasurementResult result;
      result.m = seed.m;
      result.n = seed.n;
      result.k = seed.k;
      ClearSelectedBranch();
      std::string failureDetail;
      ret = MeasureShape(result.m, result.n, result.k, stream, &result.averageMs, &result.branch,
                         &result.workspaceBytes, &result.tilingJson, &result.officialCore,
                         &result.requestedCore, &result.actualCore, &result.timingComplete,
                         &result.failedStage, &failureDetail);
      //NEW
      // Preserve the requested core even when an error happens before the
      // tiling callback can export it.  The campaign can then retry or skip
      // exactly this core without discarding the rest of the shape curve.
      if (!planItem.empty() && planItem != "official_pre" && planItem != "official_post" &&
          planItem != "original" && planItem != "shrink" && planItem != "shrink_pre" &&
          planItem != "shrink_post" && planItem != "wave_exact_pre" &&
          planItem != "wave_exact_post" && result.requestedCore == 0) {
        result.requestedCore = static_cast<uint32_t>(std::strtoul(planItem.c_str(), nullptr, 10));
      }
      if (ret == kSkipNotMatMulV3) {
        continue;
      }
      result.resultCode = ret;
      result.complete = true;
      PrintMeasurementResult(result);
      (void)fflush(stdout);
    }
  }

  // 6. 释放device资源，需要根据具体API的接口定义修改
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
