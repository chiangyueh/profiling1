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
#include <cstdlib>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
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
  return 0;
}

//NEW BEGIN: minimal original-versus-shrink measurement
struct Measurement {
  std::string tiling;
  uint32_t core = 0;
  float latency = 0.0F;
};

int Measure(const aclTensor* self, const aclTensor* mat2, aclTensor* out,
            aclrtStream stream, bool shrink, Measurement* measurement) {
  constexpr int kWarmup = 1;
  constexpr int kRepeat = 10;
  (void)::setenv("MATMUL_V3_SHRINK_IDLE_CORES", shrink ? "1" : "0", 1);
  (void)::unsetenv("MATMUL_V3_COMPARE_TILING");
  (void)::unsetenv("MATMUL_V3_COMPARE_CORE");

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  auto ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, 1, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  std::unique_ptr<aclOpExecutor, aclnnStatus (*)(aclOpExecutor*)> executorPtr(
      executor, aclDestroyAclOpExecutor);

  const char* tiling = std::getenv("MATMUL_V3_COMPARE_TILING");
  const char* core = std::getenv("MATMUL_V3_COMPARE_CORE");
  CHECK_RET(tiling != nullptr && core != nullptr,
            LOG_PRINT("MatMulV3 comparison metadata is missing\n"); return 1);
  measurement->tiling = tiling;
  measurement->core = static_cast<uint32_t>(std::strtoul(core, nullptr, 10));

  ret = aclSetAclOpExecutorRepeatable(executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclSetAclOpExecutorRepeatable failed. ERROR: %d\n", ret); return ret);

  void* workspace = nullptr;
  std::unique_ptr<void, aclError (*)(void*)> workspacePtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    workspacePtr.reset(workspace);
  }

  for (int i = 0; i < kWarmup; ++i) {
    ret = aclnnMatmul(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMatmul warmup failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  ret = aclrtCreateEvent(&start);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateEvent failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateEvent(&end);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateEvent failed. ERROR: %d\n", ret); return ret);
  ret = aclrtRecordEvent(start, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtRecordEvent failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < kRepeat; ++i) {
    ret = aclnnMatmul(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclrtRecordEvent(end, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtRecordEvent failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeEvent(end);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeEvent failed. ERROR: %d\n", ret); return ret);
  float total = 0.0F;
  ret = aclrtEventElapsedTime(&total, start, end);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtEventElapsedTime failed. ERROR: %d\n", ret); return ret);
  (void)aclrtDestroyEvent(end);
  (void)aclrtDestroyEvent(start);
  measurement->latency = total / static_cast<float>(kRepeat);
  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {16, 32};
  std::vector<int64_t> mat2Shape = {32, 16};
  std::vector<int64_t> outShape = {16, 16};
  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData(512, 1);
  std::vector<float> mat2HostData(512, 1);
  std::vector<float> outHostData(256, 0);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat2 aclTensor
  ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  Measurement original;
  Measurement shrinked;
  ret = Measure(self, mat2, out, stream, false, &original);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = Measure(self, mat2, out, stream, true, &shrinked);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  CHECK_RET(original.tiling == shrinked.tiling,
            LOG_PRINT("original and shrinked tiling keys differ\n"); return 1);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("{\"tiling\":\"%s\",\"shrinked_core\":%u,\"shrinked_latency\":%.9f,"
            "\"original_core\":%u,\"original_latency\":%.9f}\n",
            original.tiling.c_str(), shrinked.core, shrinked.latency,
            original.core, original.latency);

  // 6. 释放device资源，需要根据具体API的接口定义修改
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
//NEW END: minimal original-versus-shrink measurement
