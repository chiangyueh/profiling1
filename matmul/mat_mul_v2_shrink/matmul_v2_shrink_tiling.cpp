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
#include <memory>

#include "exe_graph/runtime/tiling_context.h"
#include "register/op_impl_kernel_registry.h"
#include "register/op_impl_registry.h"

//NEW
// CANN does not install this internal declaration in every runtime package.
// Use the exported ABI only to update the tiling pointer of the already loaded
// official MatMulV2 registration; every other official field remains intact.
namespace gert {
enum class OppImplVersionTag {
    kOpp,
    kOppKernel,
    kVersionEnd = 20
};

class OpImplSpaceRegistryV2 {
public:
    const OpImplKernelRegistry::OpImplFunctionsV2 *GetOpImpl(const char *opType) const;
};

class DefaultOpImplSpaceRegistryV2 {
public:
    static DefaultOpImplSpaceRegistryV2 &GetInstance();
    const std::shared_ptr<OpImplSpaceRegistryV2> GetSpaceRegistry(
        OppImplVersionTag versionTag = OppImplVersionTag::kOpp) const;
};
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

void SetTilingStage(const char *stage)
{
    (void)::setenv("MATMUL_TILING_STAGE", stage, 1);
}

uint32_t MatMulV2ShrinkTiling(gert::TilingContext *context)
{
    SetTilingStage("v2_wrapper_entered");
    if (context == nullptr || g_officialMatMulV2Tiling == nullptr) {
        SetTilingStage("v2_wrapper_missing_context_or_official_callback");
        return ge::GRAPH_FAILED;
    }

    const uint32_t status = g_officialMatMulV2Tiling(context);
    if (status != ge::GRAPH_SUCCESS) {
        SetTilingStage("v2_official_callback_failed");
        return status;
    }
    SetTilingStage("v2_official_callback_passed");

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
        SetTilingStage("v2_set_block_dim_failed");
        return ge::GRAPH_FAILED;
    }
    SetShrinkResult("1", oldCoreNum, newCoreNum);
    SetTilingStage("v2_shrink_applied");
    return ge::GRAPH_SUCCESS;
}
} // namespace

//NEW
extern "C" __attribute__((visibility("default"))) int ConfigureMatMulV2OfficialTiling()
{
    const auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (registry == nullptr) {
        return 0;
    }
    const auto *registered = registry->GetOpImpl("MatMulV2");
    auto *official = const_cast<gert::OpImplKernelRegistry::OpImplFunctionsV2 *>(registered);
    if (official == nullptr || official->infer_shape == nullptr || official->tiling == nullptr ||
        official->tiling_parse == nullptr ||
        official->compile_info_creator == nullptr || official->compile_info_deleter == nullptr) {
        return 0;
    }
    g_officialMatMulV2Tiling = official->tiling;
    official->tiling = MatMulV2ShrinkTiling;
    return 1;
}

//NEW
extern "C" __attribute__((visibility("default"))) int MatMulV2ShrinkRegistrationReady()
{
    if (g_officialMatMulV2Tiling == nullptr) {
        return 0;
    }
    const auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (registry == nullptr) {
        return 0;
    }
    const auto *impl = registry->GetOpImpl("MatMulV2");
    return impl != nullptr && impl->infer_shape != nullptr && impl->tiling == MatMulV2ShrinkTiling &&
        impl->tiling_parse != nullptr &&
        impl->compile_info_creator != nullptr && impl->compile_info_deleter != nullptr ? 1 : 0;
}
