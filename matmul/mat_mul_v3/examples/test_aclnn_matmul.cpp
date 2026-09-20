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
#include "acl/acl.h"
//NEW BEGIN
#include <cstdio>
#include <cstdlib>
#include <string>
#include "aclnn/acl_meta.h"
//NEW END
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

//NEW BEGIN
struct Shape {
  int64_t m;
  int64_t n;
  int64_t k;
  bool transposeB;
};

struct Measurement {
  std::string tiling;
  uint32_t core = 0;
  float latency = 0.0F;
};

struct Event {
  aclrtEvent value = nullptr;
  ~Event() {
    if (value != nullptr) {
      (void)aclrtDestroyEvent(value);
    }
  }
};

constexpr int kSkip = 10001;

int CreateMat2Tensor(const std::vector<float>& hostData, int64_t k, int64_t n, bool transposeB,
                     void** deviceAddr, aclTensor** tensor) {
  if (!transposeB) {
    return CreateAclTensor(hostData, {k, n}, deviceAddr, aclDataType::ACL_FLOAT, tensor);
  }
  const auto bytes = hostData.size() * sizeof(float);
  auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  const std::vector<int64_t> viewShape = {k, n};
  const std::vector<int64_t> storageShape = {n, k};
  const std::vector<int64_t> strides = {1, k};
  *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), aclDataType::ACL_FLOAT,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            storageShape.data(), storageShape.size(), *deviceAddr);
  return *tensor == nullptr ? 1 : ACL_SUCCESS;
}

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
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  std::unique_ptr<aclOpExecutor, aclnnStatus (*)(aclOpExecutor*)> executorPtr(
      executor, aclDestroyAclOpExecutor);

  const char* tiling = std::getenv("MATMUL_V3_COMPARE_TILING");
  const char* core = std::getenv("MATMUL_V3_COMPARE_CORE");
  if (tiling == nullptr || core == nullptr || std::string(tiling) == "SKIP") {
    return kSkip;
  }
  measurement->tiling = tiling;
  measurement->core = static_cast<uint32_t>(std::strtoul(core, nullptr, 10));

  ret = aclSetAclOpExecutorRepeatable(executor);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  void* workspace = nullptr;
  std::unique_ptr<void, aclError (*)(void*)> workspacePtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
    workspacePtr.reset(workspace);
  }

  for (int i = 0; i < kWarmup; ++i) {
    ret = aclnnMatmul(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  Event start;
  Event end;
  ret = aclrtCreateEvent(&start.value);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  ret = aclrtCreateEvent(&end.value);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  ret = aclrtRecordEvent(start.value, stream);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  for (int i = 0; i < kRepeat; ++i) {
    ret = aclnnMatmul(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      return ret;
    }
  }
  ret = aclrtRecordEvent(end.value, stream);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  ret = aclrtSynchronizeEvent(end.value);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  float total = 0.0F;
  ret = aclrtEventElapsedTime(&total, start.value, end.value);
  if (ret != ACL_SUCCESS) {
    return ret;
  }
  measurement->latency = total / static_cast<float>(kRepeat);
  return ACL_SUCCESS;
}

int RunShape(const Shape& shape, aclrtStream stream) {
  const std::vector<int64_t> selfShape = {shape.m, shape.k};
  const std::vector<int64_t> mat2Shape = {shape.k, shape.n};
  const std::vector<int64_t> outShape = {shape.m, shape.n};
  std::vector<float> selfHostData(static_cast<size_t>(shape.m * shape.k), 1.0F);
  std::vector<float> mat2HostData(static_cast<size_t>(shape.k * shape.n), 1.0F);
  std::vector<float> outHostData(static_cast<size_t>(shape.m * shape.n), 0.0F);

  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;

  auto ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<void, aclError (*)(void*)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> selfTensorPtr(self, aclDestroyTensor);
  if (ret != ACL_SUCCESS) {
    return 0;
  }

  ret = CreateMat2Tensor(mat2HostData, shape.k, shape.n, shape.transposeB, &mat2DeviceAddr, &mat2);
  std::unique_ptr<void, aclError (*)(void*)> mat2DeviceAddrPtr(mat2DeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mat2TensorPtr(mat2, aclDestroyTensor);
  if (ret != ACL_SUCCESS) {
    return 0;
  }

  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
  if (ret != ACL_SUCCESS) {
    return 0;
  }

  Measurement original;
  Measurement shrinked;
  ret = Measure(self, mat2, out, stream, false, &original);
  if (ret != ACL_SUCCESS) {
    return 0;
  }
  ret = Measure(self, mat2, out, stream, true, &shrinked);
  if (ret != ACL_SUCCESS || original.tiling != shrinked.tiling) {
    return 0;
  }

  float result = 0.0F;
  ret = aclrtMemcpy(&result, sizeof(result), outDeviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS || result != static_cast<float>(shape.k)) {
    return 0;
  }

  const char* layout = shape.transposeB ? "NT" : "NN";
  char tiling[128] = {};
  (void)snprintf(tiling, sizeof(tiling), "M%ld_N%ld_K%ld_%s/%s",
                 static_cast<long>(shape.m), static_cast<long>(shape.n),
                 static_cast<long>(shape.k), layout, original.tiling.c_str());
  LOG_PRINT("{\"tiling\":\"%s\",\"shrinked_core\":%u,\"shrinked_latency\":%.9f,"
            "\"original_core\":%u,\"original_latency\":%.9f}\n",
            tiling, shrinked.core, shrinked.latency, original.core, original.latency);
  return 1;
}

int main() {
  const std::vector<Shape> shapes = {
      {1, 48, 6144, true}, {1, 52, 6144, true}, {1, 64, 6144, true},
      {2, 64, 6144, true}, {3, 64, 7168, true}, {5, 64, 6656, true},
      {6, 64, 6144, true}, {7, 64, 6144, true}, {6, 256, 7168, true},
      {7, 64, 6656, true},
      {17, 56, 10496, true}, {18, 40, 12928, true}, {19, 40, 12800, true},
      {20, 56, 12672, true}, {21, 40, 15104, true}, {23, 40, 14976, true},
      {30, 40, 17408, true}, {33, 40, 9088, true}, {43, 40, 11264, true},
      {60, 56, 17536, true},
      {11731, 32, 17, false}, {11828, 128, 41, false}, {12211, 64, 25, false},
      {12405, 64, 21, false}, {12599, 32, 19, false}, {15515, 144, 73, false},
      {15600, 144, 73, false}, {15792, 160, 71, false}, {16624, 176, 69, false},
      {17322, 176, 69, false},
  };

  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  int completed = 0;
  for (const auto& shape : shapes) {
    completed += RunShape(shape, stream);
  }

  (void)aclrtDestroyStream(stream);
  (void)aclrtResetDevice(deviceId);
  (void)aclFinalize();
  return completed == 0 ? 1 : 0;
}
//NEW END
