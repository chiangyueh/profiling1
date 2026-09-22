// NEW BEGIN
#ifndef __OP_KERNEL_MATMUL_V3_TAIL_STREAM_K_KERNEL_H__
#define __OP_KERNEL_MATMUL_V3_TAIL_STREAM_K_KERNEL_H__

#include "mat_mul_multi_core_splitk_kernel.h"

namespace MatmulV3 {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulBlockTailStreamK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                               const TCubeTiling &tiling)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), tiling.M * tiling.N);

    TPipe pipe;
    ClearOutput<C_T>(cGlobal, tiling, pipe);
    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIC_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_FLAG);

    const uint64_t core = GetBlockIdx();
    const uint64_t cores = tiling.usedCoreNum;
    if (core >= cores) {
        return;
    }
    const uint64_t mCnt = MMV3DivCeil(tiling.M, tiling.singleCoreM);
    const uint64_t nCnt = MMV3DivCeil(tiling.N, tiling.singleCoreN);
    const uint64_t tileCount = mCnt * nCnt;
    const uint64_t tailCount = tileCount % cores;
    const uint64_t fullTileCount = tileCount - tailCount;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_NO_PRELOAD> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&tiling, &pipe);
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.N);
    mm.SetHF32(false, 0);

    for (uint64_t tile = core; tile < fullTileCount; tile += cores) {
        const uint64_t mIndex = tile % mCnt;
        const uint64_t nIndex = tile / mCnt;
        const uint64_t mStart = mIndex * tiling.singleCoreM;
        const uint64_t nStart = nIndex * tiling.singleCoreN;
        const uint64_t mUse = tiling.singleCoreM < tiling.M - mStart ? tiling.singleCoreM : tiling.M - mStart;
        const uint64_t nUse = tiling.singleCoreN < tiling.N - nStart ? tiling.singleCoreN : tiling.N - nStart;
        mm.SetSingleShape(mUse, nUse, tiling.Ka);
        mm.SetTensorA(aGlobal[mStart * tiling.Ka], A_TYPE::isTrans);
        mm.SetTensorB(bGlobal[nStart * tiling.Kb], B_TYPE::isTrans);
        mm.Iterate();
        mm.GetTensorC(cGlobal[mStart * tiling.N + nStart], 0);
    }

    if (tailCount != 0) {
        const uint64_t parts = cores / tailCount;
        const uint64_t extra = cores % tailCount;
        const uint64_t longTasks = (parts + 1) * extra;
        uint64_t tailIndex = 0;
        uint64_t partIndex = 0;
        uint64_t partCount = parts;
        if (core < longTasks) {
            partCount = parts + 1;
            tailIndex = core / partCount;
            partIndex = core % partCount;
        } else {
            const uint64_t shortIndex = core - longTasks;
            tailIndex = extra + shortIndex / parts;
            partIndex = shortIndex % parts;
        }
        const uint64_t tile = fullTileCount + tailIndex;
        const uint64_t mIndex = tile % mCnt;
        const uint64_t nIndex = tile / mCnt;
        const uint64_t mStart = mIndex * tiling.singleCoreM;
        const uint64_t nStart = nIndex * tiling.singleCoreN;
        const uint64_t mUse = tiling.singleCoreM < tiling.M - mStart ? tiling.singleCoreM : tiling.M - mStart;
        const uint64_t nUse = tiling.singleCoreN < tiling.N - nStart ? tiling.singleCoreN : tiling.N - nStart;
        const uint64_t kUnits = tiling.Ka / BLOCK_SIZE;
        const uint64_t firstUnit = partIndex * kUnits / partCount;
        const uint64_t lastUnit = (partIndex + 1) * kUnits / partCount;
        const uint64_t kStart = firstUnit * BLOCK_SIZE;
        const uint64_t kUse = (lastUnit - firstUnit) * BLOCK_SIZE;
        mm.SetSingleShape(mUse, nUse, kUse);
        mm.SetTensorA(aGlobal[mStart * tiling.Ka + kStart], A_TYPE::isTrans);
        mm.SetTensorB(bGlobal[nStart * tiling.Kb + kStart], B_TYPE::isTrans);
        mm.Iterate();
        mm.GetTensorC(cGlobal[mStart * tiling.N + nStart], 1);
    }
    SetAtomicNone();
    mm.SetHF32(false, 0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
__aicore__ inline void MatMulTailStreamK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                         const MatmulTilingData &matmulTilingData,
                                         GM_ADDR workspaceGM, uint8_t enAtomic = 0)
{
    if ASCEND_IS_AIV {
        return;
    }
    if ASCEND_IS_AIC {
        MatMulBlockTailStreamK<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(
            aGM, bGM, cGM, matmulTilingData.matmulTiling);
    }
}

} // namespace MatmulV3

#endif
// NEW END
