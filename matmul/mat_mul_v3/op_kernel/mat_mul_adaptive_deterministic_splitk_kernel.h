// NEW BEGIN
#ifndef __OP_KERNEL_MATMUL_V3_ADAPTIVE_DETERMINISTIC_SPLITK_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_ADAPTIVE_DETERMINISTIC_SPLITK_KERNEL_H__

#include "mat_mul_deterministic_splitk_kernel.h"

namespace MatmulV3 {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          FIXPIPE_OPT_SELECT FIXPIPE_OPT = FIXPIPE_OPT_SELECT::BASE>
__aicore__ inline void MatMulAdaptiveDeterministicSplitK(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
    const MatmulTilingData &matmulTilingData, GM_ADDR workspaceGM,
    uint8_t enAtomic = 0)
{
    const TCubeTiling &tiling = matmulTilingData.matmulTiling;
    const uint64_t singleSize = static_cast<uint64_t>(tiling.singleCoreM) * tiling.N;

    if ASCEND_IS_AIV {
        if (GetBlockIdx() >= static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO) {
            return;
        }
        const uint64_t coreSize = MMV3DivCeil(
            singleSize, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO);
        TPipe pipe;
        TBuf<TPosition::VECCALC> tmpBuf;
        pipe.InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        ReduceKInUb<C_TYPE>(
            cGM, workspaceGM, coreSize, singleSize,
            singleSize * static_cast<uint64_t>(tiling.usedCoreNum),
            static_cast<uint64_t>(tiling.M) * tiling.N, 1, tiling.singleCoreN,
            tiling.N, tmpBuf, false, tiling);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= tiling.usedCoreNum) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        TPipe pipe;
        GM_ADDR partialGM = reinterpret_cast<GM_ADDR>(
            workspaceGM + GetBlockIdx() * singleSize * NUM_TWO * sizeof(float));
        using PartialType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
        MatMulMultiCoreSplitKDivide<A_TYPE, B_TYPE, PartialType, BIAS_TYPE>(
            aGM, bGM, biasGM, partialGM, singleSize,
            matmulTilingData.matmulRunInfo.isHf32, &pipe, tiling, false);
    }
}

} // namespace MatmulV3

#endif
// NEW END
