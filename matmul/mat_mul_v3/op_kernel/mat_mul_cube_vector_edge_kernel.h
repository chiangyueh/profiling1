// NEW BEGIN
#ifndef MAT_MUL_CUBE_VECTOR_EDGE_KERNEL_H
#define MAT_MUL_CUBE_VECTOR_EDGE_KERNEL_H

#include "mat_mul_base_kernel.h"
#include "mat_mul_vector_dot.h"

namespace MatmulV3 {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulCubeInterior(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, const TCubeTiling &tiling)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    const uint64_t mTiles = tiling.M / tiling.singleCoreM;
    const uint64_t nTiles = tiling.N / tiling.singleCoreN;
    const uint64_t tileCount = mTiles * nTiles;
    const uint64_t core = GetBlockIdx();
    if (core >= tiling.usedCoreNum || tileCount == 0) return;

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), tiling.N * tiling.Kb);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), tiling.M * tiling.N);

    TPipe pipe;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_NO_PRELOAD> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&tiling, &pipe);
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.N);
    mm.SetHF32(false, 0);
    for (uint64_t tile = core; tile < tileCount; tile += tiling.usedCoreNum) {
        const uint64_t mStart = (tile / nTiles) * tiling.singleCoreM;
        const uint64_t nStart = (tile % nTiles) * tiling.singleCoreN;
        mm.SetSingleShape(tiling.singleCoreM, tiling.singleCoreN, tiling.Ka);
        mm.SetTensorA(aGlobal[mStart * tiling.Ka], A_TYPE::isTrans);
        mm.SetTensorB(bGlobal[nStart * tiling.Kb], B_TYPE::isTrans);
        mm.Iterate();
        mm.GetTensorC(cGlobal[mStart * tiling.N + nStart], 0);
    }
    mm.SetHF32(false, 0);
}

__aicore__ inline void MatMulVectorEdge(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, const TCubeTiling &tiling)
{
    const uint64_t m = tiling.M;
    const uint64_t n = tiling.N;
    const uint64_t k = tiling.Ka;
    const uint64_t interiorM = m / tiling.singleCoreM * tiling.singleCoreM;
    const uint64_t interiorN = n / tiling.singleCoreN * tiling.singleCoreN;
    const uint64_t bottomCount = (m - interiorM) * n;
    const uint64_t rightCount = interiorM * (n - interiorN);
    const uint64_t edgeCount = bottomCount + rightCount;
    const uint64_t workers = static_cast<uint64_t>(tiling.usedCoreNum) * NUM_AIV_TO_AIC_RATIO;
    const uint64_t worker = GetBlockIdx();
    if (edgeCount == 0 || worker >= workers) return;

    GlobalTensor<float> aGlobal;
    GlobalTensor<float> bGlobal;
    GlobalTensor<float> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aGM), m * k);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bGM), n * k);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cGM), m * n);

    TPipe pipe;
    TBuf<TPosition::VECCALC> ubBuffer;
    pipe.InitBuffer(ubBuffer, VECTOR_DOT_UB_BYTES);
    LocalTensor<float> ub = ubBuffer.Get<float>();
    LocalTensor<float> aLocal = ub;
    LocalTensor<float> bLocal = ub[VECTOR_DOT_CHUNK];
    LocalTensor<float> productLocal = ub[2 * VECTOR_DOT_CHUNK];
    LocalTensor<float> sumLocal = ub[3 * VECTOR_DOT_CHUNK];
    LocalTensor<uint8_t> reduceTmp = ub[3 * VECTOR_DOT_CHUNK + 16].ReinterpretCast<uint8_t>();

    for (uint64_t edge = worker; edge < edgeCount; edge += workers) {
        uint64_t row = 0;
        uint64_t column = 0;
        if (edge < bottomCount) {
            row = interiorM + edge / n;
            column = edge % n;
        } else {
            const uint64_t right = edge - bottomCount;
            row = right / (n - interiorN);
            column = interiorN + right % (n - interiorN);
        }
        float sum = 0.0f;
        for (uint64_t kOffset = 0; kOffset < k; kOffset += VECTOR_DOT_CHUNK) {
            const uint64_t count = min(VECTOR_DOT_CHUNK, k - kOffset);
            DataCopy(aLocal, aGlobal[row * k + kOffset], count);
            DataCopy(bLocal, bGlobal[column * k + kOffset], count);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Mul(productLocal, aLocal, bLocal, count);
            PipeBarrier<PIPE_V>();
            const uint32_t reduceShape[2] = {1, static_cast<uint32_t>(count)};
            ReduceSum<float, Pattern::Reduce::AR, false>(sumLocal, productLocal, reduceTmp, reduceShape, true);
            PipeBarrier<PIPE_ALL>();
            sum += sumLocal.GetValue(0);
        }
        sumLocal.SetValue(0, sum);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams outputParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(cGlobal[row * n + column], sumLocal, outputParams);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          FIXPIPE_OPT_SELECT FIXPIPE_OPT = FIXPIPE_OPT_SELECT::BASE>
__aicore__ inline void MatMulCubeVectorEdge(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
    const MatmulTilingData &tilingData, GM_ADDR workspaceGM, uint8_t enAtomic = 0)
{
    if ASCEND_IS_AIC {
        MatMulCubeInterior<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(aGM, bGM, cGM, tilingData.matmulTiling);
        return;
    }
    if ASCEND_IS_AIV {
        MatMulVectorEdge(aGM, bGM, cGM, tilingData.matmulTiling);
    }
}

}

#endif
// NEW END
