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
#include <iostream>
#include <memory>
#include <vector>
//NEW
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <dlfcn.h>
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
  return "MATMUL_V2";
}

//NEW
void ClearSelectedBranch() {
  (void)::unsetenv("MATMUL_SELECTED_BRANCH");
  (void)::unsetenv("MATMUL_V3_SELECTED_BRANCH");
  (void)::unsetenv("MATMUL_SHRINK_EFFECTIVE");
  (void)::unsetenv("MATMUL_SHRINK_OLD_CORES");
  (void)::unsetenv("MATMUL_SHRINK_NEW_CORES");
}

//NEW
extern "C" uint32_t TbeLoadSoAndSaveToRegistry(const char* soPath);

//NEW
int LoadOfficialMatMulHostLibraries(const char* directoryPath) {
  DIR* directory = opendir(directoryPath);
  if (directory == nullptr) {
    fprintf(stderr, "tiling registration failed: cannot open official op_host directory: %s\n", directoryPath);
    return 4;
  }

  std::vector<std::string> libraries;
  for (dirent* entry = readdir(directory); entry != nullptr; entry = readdir(directory)) {
    const std::string name(entry->d_name);
    if (name.size() >= 3 && name.compare(name.size() - 3, 3, ".so") == 0) {
      libraries.emplace_back(std::string(directoryPath) + "/" + name);
    }
  }
  closedir(directory);
  std::sort(libraries.begin(), libraries.end());
  if (libraries.empty()) {
    fprintf(stderr, "tiling registration failed: official op_host directory contains no shared libraries: %s\n",
            directoryPath);
    return 4;
  }

  size_t loadedCount = 0;
  uint32_t lastStatus = UINT32_MAX;
  std::string lastFailedLibrary;
  for (const auto& library : libraries) {
    const uint32_t status = TbeLoadSoAndSaveToRegistry(library.c_str());
    if (status == 0U) {
      ++loadedCount;
    } else {
      lastStatus = status;
      lastFailedLibrary = library;
    }
  }
  if (loadedCount == 0) {
    fprintf(stderr, "tiling registration failed: no official op_host library loaded rc=%u library=%s\n",
            lastStatus, lastFailedLibrary.c_str());
    return 4;
  }
  return ACL_SUCCESS;
}

//NEW
int EnableMatMulTilingVariants() {
  const char* shrinkMode = std::getenv("MATMUL_SHRINK_MODE");
  const char* officialHostDirectory = std::getenv("MATMUL_OFFICIAL_HOST_DIRECTORY");
  const char* v3LibraryPath = std::getenv("MATMUL_V3_HOST_LIBRARY");
  if (officialHostDirectory == nullptr || officialHostDirectory[0] == '\0' ||
      v3LibraryPath == nullptr || v3LibraryPath[0] == '\0') {
    fprintf(stderr, "tiling registration failed: required library path is missing\n");
    return 4;
  }
  //NEW
  // Preserve natural dispatch: only install the two implementations selected by that dispatcher.
  if (LoadOfficialMatMulHostLibraries(officialHostDirectory) != ACL_SUCCESS) {
    return 4;
  }
  const uint32_t v3Status = TbeLoadSoAndSaveToRegistry(v3LibraryPath);
  if (v3Status != 0U) {
    fprintf(stderr, "tiling registration failed: cannot register MatMulV3 host library rc=%u\n", v3Status);
    return 4;
  }
  if (dlopen(v3LibraryPath, RTLD_NOW | RTLD_GLOBAL) == nullptr) {
    fprintf(stderr, "tiling registration failed: cannot load MatMulV3 host library: %s\n", dlerror());
    return 4;
  }
  if (shrinkMode == nullptr || shrinkMode[0] != '1' || shrinkMode[1] != '\0') {
    return ACL_SUCCESS;
  }

  const char* v2ShrinkLibraryPath = std::getenv("MATMUL_V2_SHRINK_LIBRARY");
  if (v2ShrinkLibraryPath == nullptr || v2ShrinkLibraryPath[0] == '\0') {
    fprintf(stderr, "tiling registration failed: MatMulV2 shrink library path is missing\n");
    return 4;
  }
  const uint32_t v2ShrinkStatus = TbeLoadSoAndSaveToRegistry(v2ShrinkLibraryPath);
  if (v2ShrinkStatus != 0U) {
    fprintf(stderr, "tiling registration failed: cannot register MatMulV2 shrink library rc=%u\n", v2ShrinkStatus);
    return 4;
  }
  void* handle = dlopen(v2ShrinkLibraryPath, RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) {
    fprintf(stderr, "tiling registration failed: cannot load MatMulV2 shrink library: %s\n", dlerror());
    return 4;
  }
  using ReadyFunction = int (*)();
  auto ready = reinterpret_cast<ReadyFunction>(dlsym(handle, "MatMulV2ShrinkRegistrationReady"));
  if (ready == nullptr) {
    fprintf(stderr, "tiling registration failed: MatMulV2 readiness symbol is missing\n");
    return 4;
  }
  if (ready() != 1) {
    fprintf(stderr, "tiling registration failed: MatMulV2 official callback was not captured\n");
    return 4;
  }
  return ACL_SUCCESS;
}

//NEW
int MeasureShape(int64_t m, int64_t n, int64_t k, aclrtStream stream, float* averageMs,
                 std::string* branch) {
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

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  int8_t cubeMathType = 1;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  std::unique_ptr<void, aclError (*)(void*)> executorAddrPtr(nullptr, aclrtFree);
  // 调用aclnnMatmul第一段接口
  ret = aclnnMatmulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  //NEW
  *branch = ReadSelectedBranch();
  ret = aclSetAclOpExecutorRepeatable(executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclSetAclOpExecutorRepeatable failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
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

  for (int i = 0; i < warmup; ++i) {
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclrtEvent startEvent = nullptr;
  aclrtEvent endEvent = nullptr;
  ret = aclrtCreateEvent(&startEvent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtCreateEvent(&endEvent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = aclrtRecordEvent(startEvent, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  for (int i = 0; i < repeat; ++i) {
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  ret = aclrtRecordEvent(endEvent, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtSynchronizeEvent(endEvent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  float totalMs = 0.0F;
  ret = aclrtEventElapsedTime(&totalMs, startEvent, endEvent);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  *averageMs = totalMs / repeat;

  aclrtDestroyEvent(endEvent);
  aclrtDestroyEvent(startEvent);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  //NEW
  for (int64_t i = 0; i < size; i++) {
    CHECK_RET(resultData[i] == static_cast<float>(k), return 3);
  }

  return 0;
}

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
  ret = EnableMatMulTilingVariants();
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("MatMul tiling registration failed. ERROR: %d\n", ret); return ret);

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

    //NEW
    ClearSelectedBranch();
    float averageMs = 0.0F;
    //NEW
    std::string branch;
    ret = MeasureShape(m, n, k, stream, &averageMs, &branch);
    if (ret != ACL_SUCCESS) {
      //NEW
      fprintf(stderr, "measurement failed: M%ld_N%ld_K%ld_NN rc=%d\n",
              static_cast<long>(m), static_cast<long>(n), static_cast<long>(k), ret);
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return ret;
    }
    //NEW
    LOG_PRINT("%ld|%ld|%ld|%.9f|%s\n", static_cast<long>(m), static_cast<long>(n),
              static_cast<long>(k), averageMs, branch.c_str());
  }

  // 6. 释放device资源，需要根据具体API的接口定义修改
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
