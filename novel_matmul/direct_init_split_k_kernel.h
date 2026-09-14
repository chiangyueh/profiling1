#ifndef NOVEL_MATMUL_DIRECT_INIT_SPLIT_K_KERNEL_H
#define NOVEL_MATMUL_DIRECT_INIT_SPLIT_K_KERNEL_H

#include "mat_mul_v3_common.h"

namespace NovelMatmul {

using namespace AscendC;
using namespace matmul;

constexpr uint32_t DIRECT_INIT_WHOLE_OUTPUT_MODE = 1;
constexpr uint32_t DIRECT_INIT_TAIL_WAVE_MODE = 2;

__aicore__ inline uint64_t CeilDivU64(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

struct TailOwner {
    uint64_t tile = 0;
    uint64_t rank = 0;
    uint64_t group = 0;
};

// Distribute all AICs over the final MN wave.  Earlier tail tiles receive one
// extra K owner when the core count is not divisible by the number of tiles.
// This is a closed-form ownership rule; no device-side candidate search occurs.
__aicore__ inline TailOwner MapTailOwner(uint64_t block, uint64_t cores, uint64_t tailTiles)
{
    TailOwner owner;
    const uint64_t smallGroup = cores / tailTiles;
    const uint64_t extraGroups = cores % tailTiles;
    const uint64_t largeGroup = smallGroup + 1;
    const uint64_t largeOwners = extraGroups * largeGroup;
    if (block < largeOwners) {
        owner.tile = block / largeGroup;
        owner.rank = block % largeGroup;
        owner.group = largeGroup;
    } else {
        const uint64_t local = block - largeOwners;
        owner.tile = extraGroups + local / smallGroup;
        owner.rank = local % smallGroup;
        owner.group = smallGroup;
    }
    return owner;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
class DirectInitSplitKKernel {
public:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;

    __aicore__ inline DirectInitSplitKKernel() {}

    __aicore__ inline void Init(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
        const MatmulTilingData &tilingData, TPipe *pipe)
    {
        tiling_ = &tilingData.matmulTiling;
        mode_ = tiling_->reserved;
        aGlobal_.SetGlobalBuffer(
            reinterpret_cast<__gm__ A_T *>(aGM),
            static_cast<uint64_t>(tiling_->M) * tiling_->Ka);
        bGlobal_.SetGlobalBuffer(
            reinterpret_cast<__gm__ B_T *>(bGM),
            static_cast<uint64_t>(tiling_->Kb) * tiling_->N);
        cGlobal_.SetGlobalBuffer(
            reinterpret_cast<__gm__ C_T *>(cGM),
            static_cast<uint64_t>(tiling_->M) * tiling_->N);

        mm_.SetSubBlockIdx(0);
        mm_.Init(tiling_, pipe);
        mm_.SetOrgShape(tiling_->M, tiling_->N, tiling_->Ka, tiling_->Kb, tiling_->N);
    }

    __aicore__ inline void Process()
    {
        SetAtomicNone();
        if (mode_ == DIRECT_INIT_WHOLE_OUTPUT_MODE) {
            ProcessWholeOutputSplitK();
        } else if (mode_ == DIRECT_INIT_TAIL_WAVE_MODE) {
            ProcessTailWaveSplitK();
        }
        SetAtomicNone();
    }

private:
    __aicore__ inline void SetTile(
        uint64_t mStart, uint64_t nStart, uint64_t kStart,
        uint64_t mUse, uint64_t nUse, uint64_t kUse)
    {
        uint64_t offsetA;
        if constexpr (A_TYPE::isTrans) {
            offsetA = kStart * tiling_->M + mStart;
        } else {
            offsetA = mStart * tiling_->Ka + kStart;
        }
        uint64_t offsetB;
        if constexpr (B_TYPE::isTrans) {
            offsetB = nStart * tiling_->Kb + kStart;
        } else {
            offsetB = kStart * tiling_->N + nStart;
        }
        mm_.SetSingleShape(mUse, nUse, kUse);
        mm_.SetTensorA(aGlobal_[offsetA], A_TYPE::isTrans);
        mm_.SetTensorB(bGlobal_[offsetB], B_TYPE::isTrans);
    }

    __aicore__ inline uint64_t TileOffsetC(uint64_t tileIndex, uint64_t &mUse, uint64_t &nUse)
    {
        const uint64_t mTiles = CeilDivU64(tiling_->M, tiling_->singleCoreM);
        const uint64_t mIndex = tileIndex % mTiles;
        const uint64_t nIndex = tileIndex / mTiles;
        const uint64_t mStart = mIndex * tiling_->singleCoreM;
        const uint64_t nStart = nIndex * tiling_->singleCoreN;
        mUse = MinU64(tiling_->singleCoreM, tiling_->M - mStart);
        nUse = MinU64(tiling_->singleCoreN, tiling_->N - nStart);
        return mStart * tiling_->N + nStart;
    }

    __aicore__ inline void SetOutputTile(uint64_t tileIndex, uint64_t kStart, uint64_t kUse)
    {
        const uint64_t mTiles = CeilDivU64(tiling_->M, tiling_->singleCoreM);
        const uint64_t mIndex = tileIndex % mTiles;
        const uint64_t nIndex = tileIndex / mTiles;
        const uint64_t mStart = mIndex * tiling_->singleCoreM;
        const uint64_t nStart = nIndex * tiling_->singleCoreN;
        const uint64_t mUse = MinU64(tiling_->singleCoreM, tiling_->M - mStart);
        const uint64_t nUse = MinU64(tiling_->singleCoreN, tiling_->N - nStart);
        SetTile(mStart, nStart, kStart, mUse, nUse, kUse);
    }

    // Every owner computes one disjoint K interval.  Rank zero directly initializes
    // C; only the remaining owners use atomic accumulation.  Unlike the later
    // official multi-core Split-K protocol, this performs no prior C clear.
    __aicore__ inline void ProcessWholeOutputSplitK()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t owners = tiling_->usedCoreNum;
        const uint64_t kQuantum = tiling_->baseK;
        const uint64_t kTiles = CeilDivU64(tiling_->Ka, kQuantum);
        const uint64_t beginTile = kTiles * block / owners;
        const uint64_t endTile = kTiles * (block + 1) / owners;
        const uint64_t kStart = beginTile * kQuantum;
        const uint64_t kEnd = MinU64(endTile * kQuantum, tiling_->Ka);

        SetOutputTile(0, kStart, kEnd - kStart);
        mm_.Iterate();
        if (block == 0) {
            mm_.GetTensorC(cGlobal_[0], 0);
        }
        SyncAll();
        if (block != 0) {
            mm_.GetTensorC(cGlobal_[0], 1);
        }
    }

    // Full 20-tile waves retain ordinary disjoint-MN ownership and direct C
    // stores.  Only the final incomplete wave is repartitioned along K.  Its
    // first owner initializes C and the other owners atomically add, so no workspace
    // reducer and no C clear are required.
    __aicore__ inline void ProcessTailWaveSplitK()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t cores = tiling_->usedCoreNum;
        const uint64_t mTiles = CeilDivU64(tiling_->M, tiling_->singleCoreM);
        const uint64_t nTiles = CeilDivU64(tiling_->N, tiling_->singleCoreN);
        const uint64_t totalTiles = mTiles * nTiles;
        const uint64_t tailTiles = totalTiles % cores;
        const uint64_t fullTiles = totalTiles - tailTiles;
        const uint64_t fullRounds = fullTiles / cores;

        for (uint64_t round = 0; round < fullRounds; ++round) {
            const uint64_t tile = round * cores + block;
            uint64_t mUse = 0;
            uint64_t nUse = 0;
            const uint64_t offsetC = TileOffsetC(tile, mUse, nUse);
            SetOutputTile(tile, 0, tiling_->Ka);
            mm_.Iterate();
            mm_.GetTensorC(cGlobal_[offsetC], 0);
        }

        const TailOwner owner = MapTailOwner(block, cores, tailTiles);
        const uint64_t tile = fullTiles + owner.tile;
        const uint64_t kQuantum = tiling_->baseK;
        const uint64_t kTiles = CeilDivU64(tiling_->Ka, kQuantum);
        const uint64_t beginTile = kTiles * owner.rank / owner.group;
        const uint64_t endTile = kTiles * (owner.rank + 1) / owner.group;
        const uint64_t kStart = beginTile * kQuantum;
        const uint64_t kEnd = MinU64(endTile * kQuantum, tiling_->Ka);
        uint64_t mUse = 0;
        uint64_t nUse = 0;
        const uint64_t offsetC = TileOffsetC(tile, mUse, nUse);

        SetOutputTile(tile, kStart, kEnd - kStart);
        mm_.Iterate();
        if (owner.rank == 0) {
            mm_.GetTensorC(cGlobal_[offsetC], 0);
        }
        SyncAll();
        if (owner.rank != 0) {
            mm_.GetTensorC(cGlobal_[offsetC], 1);
        }
    }

private:
    const TCubeTiling *tiling_ = nullptr;
    uint32_t mode_ = 0;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_NO_PRELOAD> mm_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void RunDirectInitSplitK(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
    const MatmulTilingData &tilingData)
{
    if ASCEND_IS_AIV {
        return;
    }
    if ASCEND_IS_AIC {
        TPipe pipe;
        DirectInitSplitKKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> op;
        op.Init(aGM, bGM, cGM, tilingData, &pipe);
        op.Process();
    }
}

} // namespace NovelMatmul

#endif
