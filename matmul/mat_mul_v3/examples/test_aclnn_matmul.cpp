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
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h> //NEW
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
uint32_t ReadEnvironmentUint(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  return static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
}

//NEW
bool UseTransposedB() {
  const char* value = std::getenv("MATMUL_B_TRANSPOSE");
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
int SelectOfficialMatMulV3Route(int64_t m, int64_t n, int64_t k, bool transposeB,
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
  selfTensor.MutableOriginShape() = gert::Shape({m, k});
  selfTensor.MutableStorageShape() = gert::Shape({m, k});
  selfTensor.SetOriginFormat(ge::FORMAT_ND);
  selfTensor.SetStorageFormat(ge::FORMAT_ND);
  selfTensor.SetDataType(ge::DT_FLOAT);

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
  mat2Tensor.SetDataType(ge::DT_FLOAT);

  //NEW
  return selector(&selfTensor, &mat2Tensor, nullptr, false, transposeB, ge::FORMAT_ND, false,
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
  const bool transposeB = UseTransposedB();
  //NEW
  // Reject official MatMulV2 routes before allocating or copying any tensor.
  const char* v3Only = std::getenv("MATMUL_V3_ONLY");
  if (v3Only != nullptr && v3Only[0] == '1' && v3Only[1] == '\0') {
    const int selectedRoute = SelectOfficialMatMulV3Route(m, n, k, transposeB, failureDetail);
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
  std::vector<float> selfHostData(GetShapeSize(selfShape), 1);
  std::vector<float> mat2HostData(GetShapeSize(mat2StorageShape), 1);
  std::vector<float> outHostData(GetShapeSize(outShape), 0);
  // 创建self aclTensor
  *failedStage = "create_self";
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat2 aclTensor
  *failedStage = "create_mat2";
  //NEW
  if (transposeB) {
    ret = CreateTransposedAclTensor(mat2HostData, mat2Shape, mat2StorageShape, &mat2DeviceAddr,
                                    aclDataType::ACL_FLOAT, &mat2);
  } else {
    ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
  }
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  *failedStage = "create_out";
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
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
  constexpr int warmup = 10;
  constexpr int repeat = 100;

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
  std::vector<float> resultData(size, 0);
  *failedStage = "copy_output";
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  //NEW
  *failedStage = "validate_output";
  for (int64_t i = 0; i < size; i++) {
    CHECK_RET(resultData[i] == static_cast<float>(k), return 3);
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
  const bool transposeB = UseTransposedB();
  const std::string mode = ReadEnvironment("MATMUL_V3_MEASUREMENT_MODE").empty() ?
      "official" : ReadEnvironment("MATMUL_V3_MEASUREMENT_MODE");
  const std::string soc = op::GetCurrentPlatformInfo().GetSocLongVersion();
  const char* status = result.resultCode == ACL_SUCCESS ? "OK" :
      (result.resultCode == 3 ? "INVALID_OUTPUT" : "ERROR");
  const char* correctness = result.resultCode == ACL_SUCCESS ? "PASS" :
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
      "{\"shape\":\"M%ld_N%ld_K%ld_%s\",\"dtype\":\"fp32\",\"layout\":\"%s\","
      "\"soc\":\"%s\",\"branch\":\"%s\",\"mode\":\"%s\","
      "\"experiment\":\"fixed_tiling_core_sweep\","
      "\"override_scope\":\"used_core_num_block_dim_workspace\",\"requested_core\":%s,"
      "\"official_core\":%u,\"actual_core\":%u,\"latency_ms\":%s,"
      "\"workspace_bytes\":%llu,\"warmup\":10,\"repeats\":100,"
      "\"status\":\"%s\",\"correctness\":\"%s\",\"failure_stage\":%s,\"tiling\":%s}\n",
      static_cast<long>(result.m), static_cast<long>(result.n), static_cast<long>(result.k),
      transposeB ? "NT" : "NN", transposeB ? "NT" : "NN", soc.c_str(), branch.c_str(), mode.c_str(),
      requestedCore.c_str(), result.officialCore, result.actualCore, latency.c_str(),
      static_cast<unsigned long long>(result.workspaceBytes), status, correctness,
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

  for (size_t index = 0; index < results.size(); ++index) {
    auto& result = results[index];

    //NEW
    ClearSelectedBranch();
    std::string failureDetail;
    ret = MeasureShape(result.m, result.n, result.k, stream, &result.averageMs, &result.branch,
                       &result.workspaceBytes, &result.tilingJson, &result.officialCore,
                       &result.requestedCore, &result.actualCore, &result.timingComplete,
                       &result.failedStage, &failureDetail);
    if (ret == kSkipNotMatMulV3) {
      continue;
    }
    result.resultCode = ret;
    result.complete = true;
    PrintMeasurementResult(result);
  }

  // 6. 释放device资源，需要根据具体API的接口定义修改
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
