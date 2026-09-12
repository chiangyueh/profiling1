#ifndef DIRECT_MATMUL_C220_GM_TO_L1_KERNEL_H
#define DIRECT_MATMUL_C220_GM_TO_L1_KERNEL_H

#include "mat_mul_v3_common.h"

namespace DirectMatmulC220 {

template <typename T>
__aicore__ inline void CopyAFromGmToL1(
    LocalTensor<T> dst, const GlobalTensor<T>& src, uint64_t offset,
    uint32_t rows, uint32_t depth, uint32_t fullK)
{
    Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = rows;
    params.dValue = depth;
    params.srcNdMatrixStride = 0;
    params.srcDValue = fullK;
    params.dstNzC0Stride = MMV3CeilAlign(static_cast<uint64_t>(rows), ALIGNED_H);
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    DataCopy(dst, src[offset], params);
}

template <typename T>
__aicore__ inline void CopyBFromGmToL1(
    LocalTensor<T> dst, const GlobalTensor<T>& src, uint64_t offset,
    uint32_t depth, uint32_t cols, uint32_t fullN)
{
    Nd2NzParams params;
    params.ndNum = 1;
    params.nValue = depth;
    params.dValue = cols;
    params.srcNdMatrixStride = 0;
    params.srcDValue = fullN;
    params.dstNzC0Stride = MMV3CeilAlign(static_cast<uint64_t>(depth), ALIGNED_H);
    params.dstNzNStride = 1;
    params.dstNzMatrixStride = 0;
    DataCopy(dst, src[offset], params);
}

template <typename T>
__aicore__ inline void RunGmToL1(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    const MatmulTilingData& packet)
{
    const TCubeTiling& tiling = packet.matmulTiling;
    const uint32_t used = tiling.usedCoreNum;
    const uint32_t coreIndex = GetBlockIdx();

    if ASCEND_IS_AIV {
        if (coreIndex >= used * NUM_AIV_TO_AIC_RATIO) {
            return;
        }
        const uint32_t cubeIndex = coreIndex / NUM_AIV_TO_AIC_RATIO;
        const uint32_t vectorPart = coreIndex % NUM_AIV_TO_AIC_RATIO;
        const uint32_t mCount = MMV3DivCeil(tiling.M, tiling.singleCoreM);
        const uint32_t nCount = MMV3DivCeil(tiling.N, tiling.singleCoreN);
        if (cubeIndex >= mCount * nCount) {
            return;
        }
        const uint32_t mIndex = cubeIndex % mCount;
        const uint32_t nIndex = cubeIndex / mCount;
        const uint32_t mStart = mIndex * tiling.singleCoreM;
        const uint32_t nStart = nIndex * tiling.singleCoreN;
        const uint32_t mRemain = static_cast<uint32_t>(tiling.M - mStart);
        const uint32_t nRemain = static_cast<uint32_t>(tiling.N - nStart);
        const uint32_t mUse = tiling.singleCoreM < mRemain ? tiling.singleCoreM : mRemain;
        const uint32_t nUse = tiling.singleCoreN < nRemain ? tiling.singleCoreN : nRemain;
        const uint32_t firstRows = MMV3DivCeil(mUse, NUM_AIV_TO_AIC_RATIO);
        const uint32_t rowStart = vectorPart * firstRows;
        if (rowStart >= mUse) {
            return;
        }
        const uint32_t rowsRemain = mUse - rowStart;
        const uint32_t rows = firstRows < rowsRemain ? firstRows : rowsRemain;
        const uint64_t outputOffset = static_cast<uint64_t>(mStart + rowStart) * tiling.N + nStart;
        const uint64_t workspaceOffset = MMV3CeilAlign(
            MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t), static_cast<uint64_t>(512));
        TPipe pipe;
        TBuf<TPosition::VECCALC> ub;
        pipe.InitBuffer(ub, TOTAL_UB_SIZE);
        WaitFlagDevLocal(AIC_SYNC_AIV_FLAG);
        Cast32to16V220(reinterpret_cast<__gm__ T*>(cGM) + outputOffset,
                       reinterpret_cast<__gm__ float*>(workspaceGM + workspaceOffset) + outputOffset,
                       static_cast<uint64_t>(rows) * nUse, nUse, tiling.N, ub);
        return;
    }

    if ASCEND_IS_AIC {
        const uint32_t mCount = MMV3DivCeil(tiling.M, tiling.singleCoreM);
        const uint32_t nCount = MMV3DivCeil(tiling.N, tiling.singleCoreN);
        if (coreIndex >= used || coreIndex >= mCount * nCount) {
            return;
        }
        const uint32_t mIndex = coreIndex % mCount;
        const uint32_t nIndex = coreIndex / mCount;
        const uint32_t mStart = mIndex * tiling.singleCoreM;
        const uint32_t nStart = nIndex * tiling.singleCoreN;
        const uint32_t mRemain = static_cast<uint32_t>(tiling.M - mStart);
        const uint32_t nRemain = static_cast<uint32_t>(tiling.N - nStart);
        const uint32_t mUse = tiling.singleCoreM < mRemain ? tiling.singleCoreM : mRemain;
        const uint32_t nUse = tiling.singleCoreN < nRemain ? tiling.singleCoreN : nRemain;
        const uint32_t maxK = tiling.singleCoreK;
        const uint32_t maxTileM = tiling.baseM < mUse ? tiling.baseM : mUse;
        const uint32_t maxTileN = tiling.baseN < nUse ? tiling.baseN : nUse;
        const uint32_t aRowsAligned = MMV3CeilAlign(static_cast<uint64_t>(maxTileM), ALIGNED_H);
        const uint32_t bKAligned = MMV3CeilAlign(static_cast<uint64_t>(maxK), ALIGNED_H);

        GlobalTensor<T> inputA;
        GlobalTensor<T> inputB;
        GlobalTensor<float> accumulated;
        inputA.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(aGM),
                               static_cast<uint64_t>(tiling.M) * tiling.Ka);
        inputB.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bGM),
                               static_cast<uint64_t>(tiling.Kb) * tiling.N);
        const uint64_t workspaceOffset = MMV3CeilAlign(
            MAX_BLOCK_NUM * DEFAULT_BLOCK_LEN * sizeof(int32_t), static_cast<uint64_t>(512));
        accumulated.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspaceGM + workspaceOffset),
                                    static_cast<uint64_t>(tiling.M) * tiling.N);

        TPipe pipe;
        TQue<QuePosition::A1, 1> queueA;
        TQue<QuePosition::B1, 1> queueB;
        pipe.InitBuffer(queueA, 1, static_cast<uint64_t>(aRowsAligned) * maxK * sizeof(T));
        pipe.InitBuffer(queueB, 1, static_cast<uint64_t>(bKAligned) *
                                      MMV3CeilAlign(static_cast<uint64_t>(maxTileN), ALIGNED_H) * sizeof(T));

        using aL1Type = MatmulType<TPosition::TSCM, CubeFormat::NZ, T, false>;
        using bL1Type = MatmulType<TPosition::TSCM, CubeFormat::NZ, T, false>;
        using cGmType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using biasType = MatmulType<TPosition::GM, CubeFormat::ND, T>;
        MatmulImpl<aL1Type, bL1Type, cGmType, biasType, MM_CFG_NO_PRELOAD> mm;
        mm.SetSubBlockIdx(0);
        mm.Init(&tiling, &pipe);
        // A TSCM/B TSCM tensor contains one base tile.  Passing a wider
        // singleCoreN directly makes the matmul engine advance as if a second
        // baseN tile were present in the local tensor.  Materialize and launch
        // every baseM/baseN tile explicitly so local NZ strides and GM output
        // offsets remain identical at each boundary.
        for (uint32_t mLocal = 0; mLocal < mUse; mLocal += tiling.baseM) {
            const uint32_t tileMRemain = mUse - mLocal;
            const uint32_t tileM = tiling.baseM < tileMRemain ? tiling.baseM : tileMRemain;
            for (uint32_t kStart = 0, kIndex = 0; kStart < tiling.Ka;
                 kStart += maxK, ++kIndex) {
                const uint32_t kRemain = static_cast<uint32_t>(tiling.Ka - kStart);
                const uint32_t kUse = maxK < kRemain ? maxK : kRemain;
                LocalTensor<T> localA = queueA.AllocTensor<T>();
                CopyAFromGmToL1(
                    localA, inputA,
                    static_cast<uint64_t>(mStart + mLocal) * tiling.Ka + kStart,
                    tileM, kUse, tiling.Ka);
                queueA.EnQue(localA);
                localA = queueA.DeQue<T>();
                // Keep one A base tile resident while all N base tiles use it.
                for (uint32_t nLocal = 0; nLocal < nUse; nLocal += tiling.baseN) {
                    const uint32_t tileNRemain = nUse - nLocal;
                    const uint32_t tileN = tiling.baseN < tileNRemain ? tiling.baseN : tileNRemain;
                    const uint64_t outputOffset =
                        static_cast<uint64_t>(mStart + mLocal) * tiling.N + nStart + nLocal;
                    LocalTensor<T> localB = queueB.AllocTensor<T>();
                    CopyBFromGmToL1(
                        localB, inputB,
                        static_cast<uint64_t>(kStart) * tiling.N + nStart + nLocal,
                        kUse, tileN, tiling.N);
                    queueB.EnQue(localB);
                    localB = queueB.DeQue<T>();
                    mm.SetOrgShape(tileM, tileN, kUse, kUse, tiling.N);
                    mm.SetSingleShape(tileM, tileN, kUse);
                    mm.SetTensorA(localA, false);
                    mm.SetTensorB(localB, false);
                    mm.Iterate();
                    mm.GetTensorC(accumulated[outputOffset], kIndex == 0 ? 0 : 1);
                    queueB.FreeTensor(localB);
                }
                queueA.FreeTensor(localA);
            }
        }
        mm.End();
        SetAtomicNone();
        NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
    }
}

}  // namespace DirectMatmulC220

#endif
