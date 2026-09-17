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
#include <string>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h"

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
std::string ReadSelectedBranch() {
  const char* commonBranch = std::getenv("MATMUL_SELECTED_BRANCH");
  if (commonBranch != nullptr && commonBranch[0] != '\0') {
    return commonBranch;
  }
  const char* selectedBranch = std::getenv("MATMUL_V3_SELECTED_BRANCH");
  if (selectedBranch != nullptr && selectedBranch[0] != '\0') {
    return selectedBranch;
  }
  const char* shrinkMode = std::getenv("MATMUL_SHRINK_MODE");
  return shrinkMode != nullptr && shrinkMode[0] == '1' && shrinkMode[1] == '\0' ? "" : "OFFICIAL_BASELINE";
}

//NEW
void ClearSelectedBranch() {
  (void)::unsetenv("MATMUL_SELECTED_BRANCH");
  (void)::unsetenv("MATMUL_V3_SELECTED_BRANCH");
  (void)::unsetenv("MATMUL_SHRINK_EFFECTIVE");
  (void)::unsetenv("MATMUL_SHRINK_OLD_CORES");
  (void)::unsetenv("MATMUL_SHRINK_NEW_CORES");
  (void)::unsetenv("MATMUL_TILING_STAGE");
  (void)::unsetenv("MATMUL_SHRINK_API_ROUTE");
}

//NEW
extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char* soPath);

//NEW
int EnableMatMulShrink() {
  const char* v3LibraryPath = std::getenv("MATMUL_V3_HOST_LIBRARY");
  if (v3LibraryPath == nullptr || v3LibraryPath[0] == '\0') {
    fprintf(stderr, "tiling registration failed: MatMulV3 host library path is missing\n");
    return 4;
  }

  //NEW
  // The shrink process creates only MatMulV3 nodes.  Load only the matching
  // local MatMulV3 host library; there is no MatMulV2 hook or mixed registry.
  const uint32_t v3Status = TbeLoadSoAndSaveToRegistry(v3LibraryPath);
  if (v3Status != 0U) {
    fprintf(stderr, "tiling registration failed: cannot register MatMulV3 host library rc=%u\n", v3Status);
    return 4;
  }
  return ACL_SUCCESS;
}

//NEW
int MeasureShape(int64_t m, int64_t n, int64_t k, aclrtStream stream, float* averageMs,
                 std::string* branch, std::string* failedStage, std::string* failureDetail) {
  auto ret = ACL_SUCCESS;
  std::vector<int64_t> selfShape = {m, k};
  std::vector<int64_t> mat2Shape = {k, n};
  std::vector<int64_t> outShape = {m, n};
  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  //NEW
  std::vector<float> selfHostData(GetShapeSize(selfShape), 1);
  std::vector<float> mat2HostData(GetShapeSize(mat2Shape), 1);
  std::vector<float> outHostData(GetShapeSize(outShape), 0);
  // 创建self aclTensor
  *failedStage = "create_self";
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat2 aclTensor
  *failedStage = "create_mat2";
  ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
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
  const char* singleV3Route = std::getenv("MATMUL_SHRINK_SINGLE_V3");
  const char* apiRoute = std::getenv("MATMUL_SHRINK_API_ROUTE");
  if (singleV3Route != nullptr && singleV3Route[0] == '1' && singleV3Route[1] == '\0' &&
      (branch->empty() || apiRoute == nullptr || std::string(apiRoute) != "MATMUL_V3")) {
    (void)aclDestroyAclOpExecutor(executor);
    executor = nullptr;
    *failedStage = "single_v3_route_invariant";
    *failureDetail = apiRoute == nullptr ? "MatMulV3 API route was not entered" :
        "MatMulV3 tiling callback did not publish a branch";
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
  bool complete = false;
};

//NEW
int main(int argc, char** argv) {
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
  const char* singleV3Route = std::getenv("MATMUL_SHRINK_SINGLE_V3");
  const bool singleV3Enabled =
      singleV3Route != nullptr && singleV3Route[0] == '1' && singleV3Route[1] == '\0';
  if (singleV3Enabled) {
    ret = EnableMatMulShrink();
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("MatMul shrink setup failed. ERROR: %d\n", ret); return ret);
  }

  for (size_t index = 0; index < results.size(); ++index) {
    auto& result = results[index];

    //NEW
    ClearSelectedBranch();
    std::string failedStage;
    std::string failureDetail;
    ret = MeasureShape(result.m, result.n, result.k, stream, &result.averageMs, &result.branch,
                       &failedStage, &failureDetail);
    if (ret != ACL_SUCCESS) {
      //NEW
      const char* tilingStage = std::getenv("MATMUL_TILING_STAGE");
      fprintf(stderr, "measurement failed: M%ld_N%ld_K%ld_NN stage=%s rc=%d tiling_stage=%s detail=%s\n",
              static_cast<long>(result.m), static_cast<long>(result.n), static_cast<long>(result.k),
              failedStage.c_str(), ret,
              tilingStage == nullptr ? "not_reached" : tilingStage,
              failureDetail.empty() ? "unavailable" : failureDetail.c_str());
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return ret;
    }
    result.complete = true;
  }

  //NEW
  for (const auto& result : results) {
    CHECK_RET(result.complete && !result.branch.empty(), return 4);
    LOG_PRINT("%ld|%ld|%ld|%.9f|%s\n", static_cast<long>(result.m), static_cast<long>(result.n),
              static_cast<long>(result.k), result.averageMs, result.branch.c_str());
  }

  // 6. 释放device资源，需要根据具体API的接口定义修改
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
