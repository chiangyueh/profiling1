/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef __OP_KERNEL_MATMUL_V3_VECTOR_SPLIT_K_DOT_H__
#define __OP_KERNEL_MATMUL_V3_VECTOR_SPLIT_K_DOT_H__

#include "mat_mul_v3_common.h"
#include "lib/reduce/reduce.h"

using namespace AscendC;

constexpr uint64_t VECTOR_DOT_CHUNK = 4096;
constexpr uint64_t VECTOR_DOT_UB_BYTES = 128 * 1024;

__aicore__ inline void MatMulVectorSplitKDot(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, const MatmulTilingData &tilingData)
{
    const uint64_t m = static_cast<uint64_t>(tilingData.matmulTiling.M);
    const uint64_t n = static_cast<uint64_t>(tilingData.matmulTiling.N);
    const uint64_t k = static_cast<uint64_t>(tilingData.matmulTiling.Ka);
    const uint64_t outputDots = m * n;
    const uint64_t workers = static_cast<uint64_t>(tilingData.matmulTiling.usedCoreNum);
    const uint64_t worker = static_cast<uint64_t>(GetBlockIdx());
    if (outputDots == 0 || workers == 0 || worker >= workers) {
        return;
    }

    GlobalTensor<float> aGlobal;
    GlobalTensor<float> bGlobal;
    GlobalTensor<float> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aGM), m * k);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bGM), n * k);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cGM), outputDots);

    TPipe pipe;
    TBuf<TPosition::VECCALC> ubBuf;
    pipe.InitBuffer(ubBuf, VECTOR_DOT_UB_BYTES);
    LocalTensor<float> ub = ubBuf.Get<float>();
    LocalTensor<float> aLocal = ub;
    LocalTensor<float> bLocal = ub[VECTOR_DOT_CHUNK];
    LocalTensor<float> productLocal = ub[2 * VECTOR_DOT_CHUNK];
    LocalTensor<float> sumLocal = ub[3 * VECTOR_DOT_CHUNK];
    LocalTensor<uint8_t> reduceTmp = ub[3 * VECTOR_DOT_CHUNK + 16].ReinterpretCast<uint8_t>();

    const bool splitK = workers > outputDots;
    if (splitK && worker < outputDots) {
        sumLocal.SetValue(0, 0.0f);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams clearParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(cGlobal[worker], sumLocal, clearParams);
    }
    if (splitK) {
        SyncAll();
    }

    const bool outputParallel = outputDots > workers;
    const uint64_t firstOutput = outputParallel ? worker : worker % outputDots;
    const uint64_t outputStride = outputParallel ? workers : outputDots;
    const uint64_t lane = outputParallel ? 0 : worker / outputDots;
    const uint64_t lanesForOutput = outputParallel ? 1 : workers / outputDots;
    const uint64_t kBlocks = k / (BLOCK_BYTE_SIZE / sizeof(float));
    const uint64_t blocksPerLane = (kBlocks + lanesForOutput - 1) / lanesForOutput;
    const uint64_t firstBlock = lane * blocksPerLane;
    const uint64_t lastBlock = min(kBlocks, firstBlock + blocksPerLane);

    for (uint64_t outputIndex = firstOutput; outputIndex < outputDots; outputIndex += outputStride) {
        const uint64_t row = outputIndex / n;
        const uint64_t column = outputIndex % n;
        const uint64_t firstK = firstBlock * (BLOCK_BYTE_SIZE / sizeof(float));
        const uint64_t lastK = lastBlock * (BLOCK_BYTE_SIZE / sizeof(float));
        float partial = 0.0f;
        for (uint64_t kOffset = firstK; kOffset < lastK; kOffset += VECTOR_DOT_CHUNK) {
            const uint64_t count = min(VECTOR_DOT_CHUNK, lastK - kOffset);
            DataCopy(aLocal, aGlobal[row * k + kOffset], count);
            DataCopy(bLocal, bGlobal[column * k + kOffset], count);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Mul(productLocal, aLocal, bLocal, count);
            PipeBarrier<PIPE_V>();
            const uint32_t reduceShape[2] = {1, static_cast<uint32_t>(count)};
            ReduceSum<float, Pattern::Reduce::AR, false>(
                sumLocal, productLocal, reduceTmp, reduceShape, true);
            PipeBarrier<PIPE_ALL>();
            partial += sumLocal.GetValue(0);
        }

        sumLocal.SetValue(0, partial);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams outputParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        if (splitK) {
            SetAtomicAdd<float>();
        }
        DataCopyPad(cGlobal[outputIndex], sumLocal, outputParams);
        if (splitK) {
            SetAtomicNone();
        }
    }
}

#endif
