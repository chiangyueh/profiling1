/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.h"

// NEW BEGIN
extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char* soPath);

struct PreparedMatmul {
  uint64_t workspaceSize = 0;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
};

int PrepareMatmul(const aclTensor* self, const aclTensor* mat2, aclTensor* out, bool balanced,
                  PreparedMatmul* prepared) {
  unsetenv("MATMUL_THIRD_SHAPE_OFFICIAL_SEEN");
  unsetenv("MATMUL_THIRD_SHAPE_BALANCE_APPLIED");
  if (balanced) {
    setenv("MATMUL_THIRD_SHAPE_BALANCE", "1", 1);
  } else {
    unsetenv("MATMUL_THIRD_SHAPE_BALANCE");
  }
  int ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, 1, &prepared->workspaceSize, &prepared->executor);
  unsetenv("MATMUL_THIRD_SHAPE_BALANCE");
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  const bool seen = std::getenv("MATMUL_THIRD_SHAPE_OFFICIAL_SEEN") != nullptr;
  const bool applied = std::getenv("MATMUL_THIRD_SHAPE_BALANCE_APPLIED") != nullptr;
  if (!seen || applied != balanced) {
    return 4;
  }
  if (prepared->workspaceSize > 0) {
    ret = aclrtMalloc(&prepared->workspace, prepared->workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  return aclSetAclOpExecutorRepeatable(prepared->executor);
}

void ReleaseMatmul(PreparedMatmul* prepared) {
  if (prepared->executor != nullptr) {
    aclDestroyAclOpExecutor(prepared->executor);
  }
  if (prepared->workspace != nullptr) {
    aclrtFree(prepared->workspace);
  }
}

int LaunchMatmul(const PreparedMatmul& prepared, aclrtStream stream) {
  return aclnnMatmul(prepared.workspace, prepared.workspaceSize, prepared.executor, stream);
}

int MeasureMatmul(const PreparedMatmul& prepared, aclrtStream stream, float* latency) {
  constexpr int repeats = 10;
  aclrtEvent begin = nullptr;
  aclrtEvent end = nullptr;
  int ret = aclrtCreateEvent(&begin);
  if (ret == ACL_SUCCESS) {
    ret = aclrtCreateEvent(&end);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtRecordEvent(begin, stream);
  }
  for (int index = 0; ret == ACL_SUCCESS && index < repeats; ++index) {
    ret = LaunchMatmul(prepared, stream);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtRecordEvent(end, stream);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeEvent(end);
  }
  float elapsed = 0.0F;
  if (ret == ACL_SUCCESS) {
    ret = aclrtEventElapsedTime(&elapsed, begin, end);
  }
  if (end != nullptr) {
    aclrtDestroyEvent(end);
  }
  if (begin != nullptr) {
    aclrtDestroyEvent(begin);
  }
  if (ret == ACL_SUCCESS) {
    *latency = elapsed / repeats;
  }
  return ret;
}

float Median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}
// NEW END

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

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // NEW BEGIN
  const char* hostLibrary = std::getenv("MATMUL_HOST_LIBRARY");
  CHECK_RET(hostLibrary != nullptr && TbeLoadSoAndSaveToRegistry(hostLibrary) == 0U,
            LOG_PRINT("load local MatMulV3 host library failed\n"); return 4);
  // NEW END

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4096, 7168};
  std::vector<int64_t> mat2Shape = {7168, 512};
  std::vector<int64_t> outShape = {4096, 512};
  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<aclFloat16> selfHostData(GetShapeSize(selfShape), aclFloatToFloat16(1.0F));
  std::vector<aclFloat16> mat2HostData(GetShapeSize(mat2Shape), aclFloatToFloat16(1.0F));
  std::vector<aclFloat16> outHostData(GetShapeSize(outShape), aclFloatToFloat16(0.0F));
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat2 aclTensor
  ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // NEW BEGIN
  PreparedMatmul original;
  PreparedMatmul balanced;
  ret = PrepareMatmul(self, mat2, out, false, &original);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("prepare original failed. ERROR: %d\n", ret); return ret);
  ret = PrepareMatmul(self, mat2, out, true, &balanced);
  CHECK_RET(ret == ACL_SUCCESS,
            ReleaseMatmul(&original); LOG_PRINT("prepare balanced failed. ERROR: %d\n", ret); return ret);

  ret = LaunchMatmul(original, stream);
  if (ret == ACL_SUCCESS) {
    ret = LaunchMatmul(balanced, stream);
  }
  if (ret == ACL_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
  }
  CHECK_RET(ret == ACL_SUCCESS,
            ReleaseMatmul(&balanced); ReleaseMatmul(&original);
            LOG_PRINT("warmup failed. ERROR: %d\n", ret); return ret);

  std::vector<float> originalSamples;
  std::vector<float> balancedSamples;
  for (int sample = 0; sample < 5; ++sample) {
    float originalLatency = 0.0F;
    float balancedLatency = 0.0F;
    if ((sample & 1) == 0) {
      ret = MeasureMatmul(original, stream, &originalLatency);
      if (ret == ACL_SUCCESS) ret = MeasureMatmul(balanced, stream, &balancedLatency);
    } else {
      ret = MeasureMatmul(balanced, stream, &balancedLatency);
      if (ret == ACL_SUCCESS) ret = MeasureMatmul(original, stream, &originalLatency);
    }
    CHECK_RET(ret == ACL_SUCCESS,
              ReleaseMatmul(&balanced); ReleaseMatmul(&original);
              LOG_PRINT("measurement failed. ERROR: %d\n", ret); return ret);
    originalSamples.push_back(originalLatency);
    balancedSamples.push_back(balancedLatency);
  }

  auto size = GetShapeSize(outShape);
  std::vector<aclFloat16> originalResult(size);
  std::vector<aclFloat16> balancedResult(size);
  ret = LaunchMatmul(original, stream);
  if (ret == ACL_SUCCESS) ret = aclrtSynchronizeStream(stream);
  if (ret == ACL_SUCCESS) {
    ret = aclrtMemcpy(originalResult.data(), originalResult.size() * sizeof(originalResult[0]), outDeviceAddr,
                      originalResult.size() * sizeof(originalResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  if (ret == ACL_SUCCESS) ret = LaunchMatmul(balanced, stream);
  if (ret == ACL_SUCCESS) ret = aclrtSynchronizeStream(stream);
  if (ret == ACL_SUCCESS) {
    ret = aclrtMemcpy(balancedResult.data(), balancedResult.size() * sizeof(balancedResult[0]), outDeviceAddr,
                      balancedResult.size() * sizeof(balancedResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  }
  bool correct = ret == ACL_SUCCESS;
  for (int64_t index = 0; correct && index < size; ++index) {
    const float reference = aclFloat16ToFloat(originalResult[index]);
    const float candidate = aclFloat16ToFloat(balancedResult[index]);
    correct = std::isfinite(reference) && reference == 7168.0F && candidate == reference;
  }
  const float originalLatency = Median(originalSamples);
  const float balancedLatency = Median(balancedSamples);
  const float delta = (balancedLatency / originalLatency - 1.0F) * 100.0F;
  LOG_PRINT("{\"shape\":\"M4096_N512_K7168_NN_fp16\",\"original_branch\":\"BASE\","
            "\"original_core\":20,\"original_latency_ms\":%.9f,"
            "\"modified_branch\":\"BALANCED_BASE\",\"modified_core\":16,"
            "\"modified_latency_ms\":%.9f,\"delta_pct\":%.6f,"
            "\"correctness\":\"%s\"}\n",
            originalLatency, balancedLatency, delta, correct ? "PASS" : "FAIL");
  ReleaseMatmul(&balanced);
  ReleaseMatmul(&original);
  CHECK_RET(correct, return 3);
  // NEW END

  // 6. 释放device资源，需要根据具体API的接口定义修改
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
