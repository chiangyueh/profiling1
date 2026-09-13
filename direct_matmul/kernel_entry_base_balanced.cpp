#include "kernel_operator.h"
#include "mat_mul_v3_tiling_data.h"

struct DirectBalancedLocalTilingData {
    MatmulTilingData base;
    uint32_t header[8];
    DirectBalancedTaskRange ranges[DIRECT_BALANCED_GROUP_COUNT];
};

__aicore__ inline void DirectReadBalancedTiling(
    GM_ADDR tilingGM, DirectBalancedLocalTilingData &tiling)
{
    const __gm__ uint32_t *global =
        reinterpret_cast<const __gm__ uint32_t *>(tilingGM);
    uint32_t *base = reinterpret_cast<uint32_t *>(&tiling.base);
    for (uint32_t index = 0;
         index < sizeof(MatmulTilingData) / sizeof(uint32_t); ++index) {
        base[index] = global[index];
    }
    constexpr uint32_t baseWords = sizeof(MatmulTilingData) / sizeof(uint32_t);
    for (uint32_t index = 0; index < 8; ++index) {
        tiling.header[index] = global[baseWords + index];
    }
    const uint32_t rangeWord = baseWords + 8 +
        static_cast<uint32_t>(block_idx) * DIRECT_BALANCED_GROUP_COUNT * 2;
    uint32_t *ranges = reinterpret_cast<uint32_t *>(&tiling.ranges[0]);
    for (uint32_t index = 0; index < DIRECT_BALANCED_GROUP_COUNT * 2; ++index) {
        ranges[index] = global[rangeWord + index];
    }
}

#include "mat_mul_v3_common.h"
#include "mat_mul_base_kernel.h"

using namespace AscendC;
using namespace matmul;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

// The retained Matmul pipeline consumes the same legal 128x256x64 BASE
// packet.  Only output-tile ownership changes.  The host supplies four exact
// task-class ranges for every core; this block maps its own ranges back to the
// two-dimensional output grid.
class MatmulBaseBalancedBlock : public MatmulBaseBlock {
public:
    __aicore__ inline MatmulBaseBalancedBlock() : MatmulBaseBlock() {}

    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData)
    {
        const DirectBalancedLocalTilingData *local =
            static_cast<const DirectBalancedLocalTilingData *>(tilingData);
        MatmulBaseBlock::Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(
            static_cast<const void *>(&local->base));
        header_ = &local->header[0];
        ranges_ = &local->ranges[0];
    }

    __aicore__ inline void InitBlockIndex(uint64_t index)
    {
        (void)index;
        const bool valid =
            header_[0] == DIRECT_BALANCED_SCHEDULE_MAGIC &&
            header_[1] == DIRECT_BALANCED_SCHEDULE_VERSION &&
            header_[2] == DIRECT_BALANCED_CORE_COUNT &&
            header_[3] == DIRECT_BALANCED_GROUP_COUNT &&
            header_[4] == params_.totalTileCnt &&
            header_[5] == params_.mCntUse &&
            header_[6] == params_.nCntUse;
        params_.realRound = 0;
        if (!valid) {
            return;
        }
        for (uint32_t group = 0; group < DIRECT_BALANCED_GROUP_COUNT; ++group) {
            params_.realRound += ranges_[group].count;
        }
        params_.index = 0;
    }

    __aicore__ inline void UpdateBasicIndex(const uint64_t roundIdx)
    {
        uint64_t remaining = roundIdx;
        uint32_t group = 0;
        for (; group < DIRECT_BALANCED_GROUP_COUNT; ++group) {
            if (remaining < ranges_[group].count) {
                break;
            }
            remaining -= ranges_[group].count;
        }
        if (group >= DIRECT_BALANCED_GROUP_COUNT) {
            params_.index = params_.totalTileCnt;
            return;
        }
        const uint64_t offset = ranges_[group].start + remaining;
        const bool hasMTail = params_.mBaseTail !=
            matmulTilingData_->matmulTiling.singleCoreM;
        const bool hasNTail = params_.nBaseTail !=
            matmulTilingData_->matmulTiling.singleCoreN;
        const uint64_t fullM = params_.mCntUse - static_cast<uint64_t>(hasMTail);
        const uint64_t fullN = params_.nCntUse - static_cast<uint64_t>(hasNTail);
        uint64_t mIndex = 0;
        uint64_t nIndex = 0;
        if (group == 0) {
            mIndex = offset / fullN;
            nIndex = offset % fullN;
        } else if (group == 1) {
            mIndex = offset;
            nIndex = params_.nCntUse - 1;
        } else if (group == 2) {
            mIndex = params_.mCntUse - 1;
            nIndex = offset;
        } else {
            mIndex = params_.mCntUse - 1;
            nIndex = params_.nCntUse - 1;
        }
        params_.index = mIndex * params_.nCntUse + nIndex;
    }

    __aicore__ inline void UpdateBlockIndex() {}

private:
    const uint32_t *header_ = nullptr;
    const DirectBalancedTaskRange *ranges_ = nullptr;
};

template <bool TRANS_B>
__aicore__ inline void DirectRunBalancedBase(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
    GM_ADDR cGM, GM_ADDR workspaceGM,
    DirectBalancedLocalTilingData &tilingData)
{
    __gm__ uint8_t *user = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);
    using aType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X2, TRANS_B>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_Y>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;
    TPipe pipe;
    MatmulBaseKernel<aType, bType, cType, biasType,
        MatmulBaseBalancedBlock, MM_CFG_NO_PRELOAD> op;
    op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void MATMUL_DIRECT_KERNEL(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    DirectBalancedLocalTilingData tilingData;
    DirectReadBalancedTiling(tilingGM, tilingData);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(MATMUL_DIRECT_TILING_KEY, KERNEL_TYPE_AIC_ONLY);

    // Transposed A requires MatMulV3's ND2NZ head and is rejected by the host
    // proof.  B transpose changes only the retained BASE template type, so the
    // same audited ownership schedule is valid for both NN and NT layouts.
    if (tilingData.base.matmulRunInfo.transB != 0) {
        DirectRunBalancedBase<true>(
            aGM, bGM, biasGM, offsetWGM, cGM, workspaceGM, tilingData);
    } else {
        DirectRunBalancedBase<false>(
            aGM, bGM, biasGM, offsetWGM, cGM, workspaceGM, tilingData);
    }
}
