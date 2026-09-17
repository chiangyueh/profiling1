/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

//NEW
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "exe_graph/runtime/tiling_context.h"
#include "register/op_impl_registry.h"

//NEW
// MatMulV2's official registration includes this parser and these compile-info
// lifetime functions in addition to TilingForMatMul. They are exported by the
// official liboptiling.so loaded before this combined host library.
namespace gert {
class GemmCompileInfo;
uint32_t GemmParseFunc(TilingParseContext *context);
template <> void *OpImplRegisterV2::CreateCompileInfo<GemmCompileInfo, 0>();
template <> void OpImplRegisterV2::DeleteCompileInfo<GemmCompileInfo>(void *object);
} // namespace gert

namespace {
using TilingFunc = gert::OpImplRegisterV2::TilingKernelFunc;

TilingFunc g_officialMatMulV2Tiling = nullptr;

constexpr size_t kEnablePadAttrIndex = 6;
constexpr size_t kPaddedDimensionWordCount = 3;
constexpr size_t kBatchDimIndex = 9;
constexpr size_t kNDimIndex = 10;
constexpr size_t kMDimIndex = 11;
constexpr size_t kKDimIndex = 12;

void SetShrinkResult(const char *effective, uint32_t oldCoreNum, uint32_t newCoreNum)
{
    char oldCoreText[16] = {};
    char newCoreText[16] = {};
    (void)snprintf(oldCoreText, sizeof(oldCoreText), "%u", oldCoreNum);
    (void)snprintf(newCoreText, sizeof(newCoreText), "%u", newCoreNum);
    (void)::setenv("MATMUL_SELECTED_BRANCH", "MATMUL_V2", 1);
    (void)::setenv("MATMUL_SHRINK_EFFECTIVE", effective, 1);
    (void)::setenv("MATMUL_SHRINK_OLD_CORES", oldCoreText, 1);
    (void)::setenv("MATMUL_SHRINK_NEW_CORES", newCoreText, 1);
}

uint32_t MatMulV2ShrinkTiling(gert::TilingContext *context)
{
    if (context == nullptr || g_officialMatMulV2Tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t status = g_officialMatMulV2Tiling(context);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    const uint32_t oldCoreNum = context->GetBlockDim();
    const char *shrinkMode = std::getenv("MATMUL_SHRINK_MODE");
    if (shrinkMode == nullptr || shrinkMode[0] != '1' || shrinkMode[1] != '\0') {
        SetShrinkResult("0", oldCoreNum, oldCoreNum);
        return ge::GRAPH_SUCCESS;
    }

    const auto *raw = context->GetRawTilingData();
    const auto *attrs = context->GetAttrs();
    size_t dimensionOffset = 0;
    if (attrs != nullptr && attrs->GetAttrNum() > kEnablePadAttrIndex) {
        const auto *enablePad = attrs->GetAttrPointer<int64_t>(kEnablePadAttrIndex);
        if (enablePad != nullptr && *enablePad != 0) {
            dimensionOffset = kPaddedDimensionWordCount;
        }
    }
    const size_t minimumWordCount = kKDimIndex + dimensionOffset + 1;
    if (raw == nullptr || raw->GetData() == nullptr ||
        raw->GetDataSize() < minimumWordCount * sizeof(int32_t)) {
        SetShrinkResult("0", oldCoreNum, oldCoreNum);
        return ge::GRAPH_SUCCESS;
    }

    const auto *words = static_cast<const int32_t *>(raw->GetData());
    const int32_t dimensions[] = {
        words[kBatchDimIndex + dimensionOffset], words[kNDimIndex + dimensionOffset],
        words[kMDimIndex + dimensionOffset], words[kKDimIndex + dimensionOffset]};
    uint64_t activeTasks = 1;
    for (const int32_t dimension : dimensions) {
        if (dimension <= 0 ||
            activeTasks > std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(dimension)) {
            SetShrinkResult("0", oldCoreNum, oldCoreNum);
            return ge::GRAPH_SUCCESS;
        }
        activeTasks *= static_cast<uint64_t>(dimension);
    }

    const uint32_t newCoreNum = static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(oldCoreNum, activeTasks)));
    if (newCoreNum >= oldCoreNum) {
        SetShrinkResult("0", oldCoreNum, oldCoreNum);
        return ge::GRAPH_SUCCESS;
    }

    if (context->SetBlockDim(newCoreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetShrinkResult("1", oldCoreNum, newCoreNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace

//NEW
IMPL_OP_OPTILING(MatMulV2)
    .Tiling(MatMulV2ShrinkTiling, 2048)
    .TilingParse<gert::GemmCompileInfo>(gert::GemmParseFunc);

//NEW
extern "C" __attribute__((visibility("default"))) int ConfigureMatMulV2OfficialTiling(void *callback)
{
    g_officialMatMulV2Tiling = reinterpret_cast<TilingFunc>(callback);
    return g_officialMatMulV2Tiling == nullptr ? 0 : 1;
}

//NEW
extern "C" __attribute__((visibility("default"))) int MatMulV2ShrinkRegistrationReady()
{
    return g_officialMatMulV2Tiling == nullptr ? 0 : 1;
}
