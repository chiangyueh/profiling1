#ifndef INDEPENDENT_TILED_MATMUL_KERNEL_H
#define INDEPENDENT_TILED_MATMUL_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mat_mul_v3_tiling_data_280.h"

namespace IndependentMatmul {

using namespace AscendC;
using namespace matmul;

constexpr uint32_t MODE_MICRO_DIRECT = 11;
constexpr uint32_t MODE_BALANCED_MN = 12;
constexpr uint32_t MODE_SEEDED_SPLIT_K = 13;
constexpr uint32_t MODE_RESIDENT_B_M_STRIPE = 14;
constexpr uint32_t MODE_RESIDENT_A_N_STRIPE = 15;
constexpr uint32_t MODE_SEEDED_TAIL_WAVE = 16;

constexpr MatmulConfig INDEPENDENT_MATMUL_CONFIG =
    GetMDLConfig(false, false, 0, false, false, false, true);

__aicore__ inline uint64_t IndependentCeilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline uint64_t IndependentMin(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

struct TailOwner {
    uint64_t tile = 0;
    uint64_t rank = 0;
    uint64_t group = 0;
};

// The first remainder groups receive one extra AIC.  This closed-form map is
// bijective over [0, cores) and never needs a device-side queue or search.
__aicore__ inline TailOwner MapTailOwner(
    uint64_t block, uint64_t cores, uint64_t tailTiles)
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
class IndependentTiledKernel {
public:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;

    __aicore__ inline IndependentTiledKernel() {}

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
        mm_.SetOrgShape(
            tiling_->M, tiling_->N, tiling_->Ka, tiling_->Kb, tiling_->N);
    }

    __aicore__ inline void Process()
    {
        SetAtomicNone();
        if (mode_ == MODE_SEEDED_SPLIT_K) {
            ProcessWholeOutputSplitK();
        } else if (mode_ == MODE_SEEDED_TAIL_WAVE) {
            ProcessTailWaveSplitK();
        } else if (
            mode_ == MODE_MICRO_DIRECT || mode_ == MODE_BALANCED_MN ||
            mode_ == MODE_RESIDENT_B_M_STRIPE ||
            mode_ == MODE_RESIDENT_A_N_STRIPE) {
            ProcessDisjointOutputTiles();
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

    __aicore__ inline void MapOutputTile(
        uint64_t logicalTile, uint64_t &mStart, uint64_t &nStart,
        uint64_t &mUse, uint64_t &nUse)
    {
        const uint64_t mTiles =
            IndependentCeilDiv(tiling_->M, tiling_->singleCoreM);
        const uint64_t nIndex = logicalTile / mTiles;
        uint64_t mIndex = logicalTile - nIndex * mTiles;
        // Alternate the M direction on successive N stripes.  The ownership
        // remains bijective while the boundary between stripes stays local.
        if ((nIndex & 1U) != 0U) {
            mIndex = mTiles - 1 - mIndex;
        }
        mStart = mIndex * tiling_->singleCoreM;
        nStart = nIndex * tiling_->singleCoreN;
        mUse = IndependentMin(tiling_->singleCoreM, tiling_->M - mStart);
        nUse = IndependentMin(tiling_->singleCoreN, tiling_->N - nStart);
    }

    __aicore__ inline void CalculateOutputTile(
        uint64_t logicalTile, uint64_t kStart, uint64_t kUse,
        uint8_t atomic)
    {
        uint64_t mStart = 0;
        uint64_t nStart = 0;
        uint64_t mUse = 0;
        uint64_t nUse = 0;
        MapOutputTile(logicalTile, mStart, nStart, mUse, nUse);
        SetTile(mStart, nStart, kStart, mUse, nUse, kUse);
        mm_.Iterate();
        mm_.GetTensorC(cGlobal_[mStart * tiling_->N + nStart], atomic);
    }

    // All direct families use the same collision-free scheduler.  Their
    // different tiling packets change the unit of ownership: one whole micro
    // output, balanced 2-D tiles, M stripes with B reuse, or N stripes with A
    // reuse.  Every block owns a contiguous number of logical tiles and the
    // maximum task-count difference is exactly one.
    __aicore__ inline void ProcessDisjointOutputTiles()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t cores = tiling_->usedCoreNum;
        const uint64_t mTiles =
            IndependentCeilDiv(tiling_->M, tiling_->singleCoreM);
        const uint64_t nTiles =
            IndependentCeilDiv(tiling_->N, tiling_->singleCoreN);
        const uint64_t totalTiles = mTiles * nTiles;
        const uint64_t begin = totalTiles * block / cores;
        const uint64_t end = totalTiles * (block + 1) / cores;
        for (uint64_t tile = begin; tile < end; ++tile) {
            CalculateOutputTile(tile, 0, tiling_->Ka, 0);
        }
    }

    // Rank zero publishes the initial value of C before the other K owners
    // are released.  This removes both the full-C clear and one atomic update
    // per output element from the ordinary clear-then-all-atomic protocol.
    __aicore__ inline void ProcessWholeOutputSplitK()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t owners = tiling_->usedCoreNum;
        const uint64_t kQuantum = tiling_->baseK;
        const uint64_t kTiles = IndependentCeilDiv(tiling_->Ka, kQuantum);
        const uint64_t beginTile = kTiles * block / owners;
        const uint64_t endTile = kTiles * (block + 1) / owners;
        const uint64_t kStart = beginTile * kQuantum;
        const uint64_t kEnd = IndependentMin(endTile * kQuantum, tiling_->Ka);
        SetTile(0, 0, kStart, tiling_->M, tiling_->N, kEnd - kStart);
        mm_.Iterate();
        if (block == 0) {
            mm_.GetTensorC(cGlobal_[0], 0);
        }
        SyncAll();
        if (block != 0) {
            mm_.GetTensorC(cGlobal_[0], 1);
        }
    }

    // Complete MN waves retain direct, disjoint output ownership.  Only the
    // final under-filled wave is expanded into (MN,K) work.  Each tail group
    // uses the same direct-seed-before-atomic protocol and needs no reducer.
    __aicore__ inline void ProcessTailWaveSplitK()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t cores = tiling_->usedCoreNum;
        const uint64_t mTiles =
            IndependentCeilDiv(tiling_->M, tiling_->singleCoreM);
        const uint64_t nTiles =
            IndependentCeilDiv(tiling_->N, tiling_->singleCoreN);
        const uint64_t totalTiles = mTiles * nTiles;
        const uint64_t tailTiles = totalTiles % cores;
        const uint64_t fullTiles = totalTiles - tailTiles;
        const uint64_t fullRounds = fullTiles / cores;

        for (uint64_t round = 0; round < fullRounds; ++round) {
            CalculateOutputTile(round * cores + block, 0, tiling_->Ka, 0);
        }

        const TailOwner owner = MapTailOwner(block, cores, tailTiles);
        const uint64_t logicalTile = fullTiles + owner.tile;
        const uint64_t kQuantum = tiling_->baseK;
        const uint64_t kTiles = IndependentCeilDiv(tiling_->Ka, kQuantum);
        const uint64_t beginTile = kTiles * owner.rank / owner.group;
        const uint64_t endTile = kTiles * (owner.rank + 1) / owner.group;
        const uint64_t kStart = beginTile * kQuantum;
        const uint64_t kEnd = IndependentMin(endTile * kQuantum, tiling_->Ka);

        uint64_t mStart = 0;
        uint64_t nStart = 0;
        uint64_t mUse = 0;
        uint64_t nUse = 0;
        MapOutputTile(logicalTile, mStart, nStart, mUse, nUse);
        SetTile(mStart, nStart, kStart, mUse, nUse, kEnd - kStart);
        mm_.Iterate();
        if (owner.rank == 0) {
            mm_.GetTensorC(cGlobal_[mStart * tiling_->N + nStart], 0);
        }
        SyncAll();
        if (owner.rank != 0) {
            mm_.GetTensorC(cGlobal_[mStart * tiling_->N + nStart], 1);
        }
    }

private:
    const TCubeTiling *tiling_ = nullptr;
    uint32_t mode_ = 0;
    GlobalTensor<A_T> aGlobal_;
    GlobalTensor<B_T> bGlobal_;
    GlobalTensor<C_T> cGlobal_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
               INDEPENDENT_MATMUL_CONFIG> mm_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void RunIndependentTiled(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
    const MatmulTilingData &tilingData)
{
    if ASCEND_IS_AIV {
        return;
    }
    if ASCEND_IS_AIC {
        TPipe pipe;
        IndependentTiledKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> op;
        op.Init(aGM, bGM, cGM, tilingData, &pipe);
        op.Process();
    }
}

} // namespace IndependentMatmul

#endif
