/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

//NEW
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/runtime/tiling_context.h"

namespace {
using TilingFunc = uint32_t (*)(gert::TilingContext *);
TilingFunc g_officialMatMulV2Tiling = nullptr;

//NEW
// Offsets of the hardware fields in the CANN 8.5 CubeCompileInfo base of
// GemmCompileInfo. Access is through memcpy so no foreign C++ type is aliased.
constexpr size_t kCoreNumOffset = 276;
constexpr size_t kUbSizeOffset = 280;
constexpr size_t kL1SizeOffset = 288;
constexpr size_t kL0aSizeOffset = 304;
constexpr size_t kL0bSizeOffset = 312;
constexpr size_t kL0cSizeOffset = 320;
constexpr uint64_t kCubeBlock = 16;
constexpr uint32_t kMaximumExpectedAicoreCount = 64;
constexpr uint64_t kMinimumLocalMemory = 32 * 1024;
constexpr uint64_t kMaximumLocalMemory = 16 * 1024 * 1024;

template <typename T>
T ReadCompileInfoField(const void *compileInfo, size_t offset)
{
    T value {};
    std::memcpy(&value, static_cast<const uint8_t *>(compileInfo) + offset, sizeof(value));
    return value;
}

template <typename T>
void WriteCompileInfoField(void *compileInfo, size_t offset, T value)
{
    std::memcpy(static_cast<uint8_t *>(compileInfo) + offset, &value, sizeof(value));
}

bool IsPlausibleLocalMemory(uint64_t size)
{
    return size >= kMinimumLocalMemory && size <= kMaximumLocalMemory && size % 1024 == 0;
}

bool IsCompileInfoAbiValid(const void *compileInfo)
{
    const uint32_t coreNum = ReadCompileInfoField<uint32_t>(compileInfo, kCoreNumOffset);
    return coreNum > 0 && coreNum <= kMaximumExpectedAicoreCount &&
        IsPlausibleLocalMemory(ReadCompileInfoField<uint64_t>(compileInfo, kUbSizeOffset)) &&
        IsPlausibleLocalMemory(ReadCompileInfoField<uint64_t>(compileInfo, kL1SizeOffset)) &&
        IsPlausibleLocalMemory(ReadCompileInfoField<uint64_t>(compileInfo, kL0aSizeOffset)) &&
        IsPlausibleLocalMemory(ReadCompileInfoField<uint64_t>(compileInfo, kL0bSizeOffset)) &&
        IsPlausibleLocalMemory(ReadCompileInfoField<uint64_t>(compileInfo, kL0cSizeOffset));
}

class CoreLimitRestoreGuard {
public:
    CoreLimitRestoreGuard(void *compileInfo, uint32_t originalCoreLimit)
        : compileInfo_(compileInfo), originalCoreLimit_(originalCoreLimit)
    {
    }

    ~CoreLimitRestoreGuard()
    {
        WriteCompileInfoField<uint32_t>(compileInfo_, kCoreNumOffset, originalCoreLimit_);
    }

    CoreLimitRestoreGuard(const CoreLimitRestoreGuard &) = delete;
    CoreLimitRestoreGuard &operator=(const CoreLimitRestoreGuard &) = delete;

private:
    void *compileInfo_;
    uint32_t originalCoreLimit_;
};

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

void SetTilingStage(const char *stage)
{
    (void)::setenv("MATMUL_TILING_STAGE", stage, 1);
}

void SetRetileResult(const char *branch, const char *effective, uint32_t oldCoreNum, uint32_t newCoreNum)
{
    char oldCoreText[16] = {};
    char newCoreText[16] = {};
    (void)snprintf(oldCoreText, sizeof(oldCoreText), "%u", oldCoreNum);
    (void)snprintf(newCoreText, sizeof(newCoreText), "%u", newCoreNum);
    (void)::setenv("MATMUL_SELECTED_BRANCH", branch, 1);
    (void)::setenv("MATMUL_SHRINK_EFFECTIVE", effective, 1);
    (void)::setenv("MATMUL_SHRINK_OLD_CORES", oldCoreText, 1);
    (void)::setenv("MATMUL_SHRINK_NEW_CORES", newCoreText, 1);
}

bool GetOutputMN(const gert::TilingContext &context, uint64_t &m, uint64_t &n)
{
    const auto *output = context.GetOutputShape(0);
    if (output == nullptr) {
        return false;
    }
    const auto &shape = output->GetStorageShape();
    const size_t dimNum = shape.GetDimNum();
    if (dimNum < 2) {
        return false;
    }
    const int64_t rawM = shape.GetDim(dimNum - 2);
    const int64_t rawN = shape.GetDim(dimNum - 1);
    if (rawM <= 0 || rawN <= 0) {
        return false;
    }
    m = static_cast<uint64_t>(rawM);
    n = static_cast<uint64_t>(rawN);
    return true;
}

//NEW
uint32_t MatMulV2RetiledTiling(gert::TilingContext *context)
{
    SetTilingStage("v2_retile_entered");
    if (context == nullptr || g_officialMatMulV2Tiling == nullptr) {
        SetTilingStage("v2_official_callback_missing");
        return ge::GRAPH_FAILED;
    }

    const char *shrinkMode = std::getenv("MATMUL_SHRINK_MODE");
    if (shrinkMode == nullptr || shrinkMode[0] != '1' || shrinkMode[1] != '\0') {
        return g_officialMatMulV2Tiling(context);
    }

    void *compileInfo = const_cast<void *>(context->GetCompileInfo());
    if (compileInfo == nullptr || !IsCompileInfoAbiValid(compileInfo)) {
        SetTilingStage("v2_retile_compile_info_abi_mismatch");
        return ge::GRAPH_FAILED;
    }

    uint64_t m = 0;
    uint64_t n = 0;
    if (!GetOutputMN(*context, m, n)) {
        SetTilingStage("v2_retile_invalid_output_shape");
        return ge::GRAPH_FAILED;
    }

    const uint32_t originalCoreLimit = ReadCompileInfoField<uint32_t>(compileInfo, kCoreNumOffset);
    const uint64_t independentOutputTiles = CeilDiv(m, kCubeBlock) * CeilDiv(n, kCubeBlock);
    const uint32_t retileCoreLimit = static_cast<uint32_t>(std::max<uint64_t>(
        1, std::min<uint64_t>(originalCoreLimit, independentOutputTiles)));

    WriteCompileInfoField<uint32_t>(compileInfo, kCoreNumOffset, retileCoreLimit);
    CoreLimitRestoreGuard restoreCoreLimit(compileInfo, originalCoreLimit);
    const uint32_t status = g_officialMatMulV2Tiling(context);
    if (status != ge::GRAPH_SUCCESS) {
        SetTilingStage("v2_official_retile_failed");
        return status;
    }

    const uint32_t generatedCoreNum = context->GetBlockDim();
    if (generatedCoreNum == 0 || generatedCoreNum > retileCoreLimit) {
        SetTilingStage("v2_retile_generated_invalid_core_count");
        return ge::GRAPH_FAILED;
    }

    const bool effective = retileCoreLimit < originalCoreLimit;
    SetRetileResult(effective ? "MATMUL_V2_RETILED" : "MATMUL_V2_CORE_CAP_NOT_APPLICABLE",
                    effective ? "1" : "0", originalCoreLimit, generatedCoreNum);
    SetTilingStage(effective ? "v2_retile_applied" : "v2_retile_not_applicable");
    return ge::GRAPH_SUCCESS;
}
} // namespace

//NEW
// The official legacy host library is loaded first. Replace only its tiling
// function pointer; infer-shape, parser, compile-info and all other fields stay
// in the same complete official MatMulV2 registry entry.
extern "C" int InstallMatMulV2RetileHook()
{
    const auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (registry == nullptr) {
        return 0;
    }
    const auto *registered = registry->GetOpImpl("MatMulV2");
    if (registered == nullptr || registered->tiling == nullptr || registered->infer_shape == nullptr ||
        registered->tiling_parse == nullptr || registered->compile_info_creator == nullptr ||
        registered->compile_info_deleter == nullptr) {
        return 0;
    }
    if (registered->tiling != MatMulV2RetiledTiling) {
        g_officialMatMulV2Tiling = registered->tiling;
        auto *mutableEntry = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registered);
        mutableEntry->tiling = MatMulV2RetiledTiling;
    }
    return g_officialMatMulV2Tiling != nullptr && registered->tiling == MatMulV2RetiledTiling ? 1 : 0;
}

//NEW
extern "C" int MatMulV2RetileHookReady()
{
    const auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (registry == nullptr || g_officialMatMulV2Tiling == nullptr) {
        return 0;
    }
    const auto *registered = registry->GetOpImpl("MatMulV2");
    return registered != nullptr && registered->tiling == MatMulV2RetiledTiling &&
        registered->infer_shape != nullptr && registered->tiling_parse != nullptr &&
        registered->compile_info_creator != nullptr && registered->compile_info_deleter != nullptr ? 1 : 0;
}
