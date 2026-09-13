#include "kernel_operator.h"
#include "mat_mul_v3_tiling_data.h"

__aicore__ inline void DirectReadBalancedTiling(
    GM_ADDR tilingGM, MatmulTilingData &tiling)
{
    uint32_t *local = reinterpret_cast<uint32_t *>(&tiling);
    const __gm__ uint32_t *global =
        reinterpret_cast<const __gm__ uint32_t *>(tilingGM);
    for (uint32_t index = 0;
         index < sizeof(MatmulTilingData) / sizeof(uint32_t); ++index) {
        local[index] = global[index];
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
// packet.  Only output-tile ownership changes.  This class deliberately
// supports one L2 rectangle: the host-side proof rejects every other layout.
class MatmulBaseBalancedBlock : public MatmulBaseBlock {
public:
    __aicore__ inline MatmulBaseBalancedBlock() : MatmulBaseBlock() {}

    __aicore__ inline void InitBlockIndex(uint64_t index)
    {
        (void)index;
        const uint64_t cores =
            static_cast<uint64_t>(matmulTilingData_->matmulTiling.usedCoreNum);
        const uint64_t completeRounds = params_.totalTileCnt / cores;
        const uint64_t remainder = params_.totalTileCnt % cores;
        params_.realRound = completeRounds +
            (static_cast<uint64_t>(block_idx) < remainder ? 1UL : 0UL);
        params_.index = static_cast<uint64_t>(block_idx);
    }

    __aicore__ inline void UpdateBasicIndex(const uint64_t roundIdx)
    {
        params_.index = static_cast<uint64_t>(block_idx) + roundIdx *
            static_cast<uint64_t>(matmulTilingData_->matmulTiling.usedCoreNum);
    }

    __aicore__ inline void UpdateBlockIndex() {}
};

extern "C" __global__ __aicore__ void MATMUL_DIRECT_KERNEL(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    MatmulTilingData tilingData;
    DirectReadBalancedTiling(tilingGM, tilingData);
    __gm__ uint8_t *user = reinterpret_cast<__gm__ uint8_t *>(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(MATMUL_DIRECT_TILING_KEY, KERNEL_TYPE_AIC_ONLY);

    using aType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_Y>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;
    TPipe pipe;
    MatmulBaseKernel<aType, bType, cType, biasType,
        MatmulBaseBalancedBlock, MM_CFG_NO_PRELOAD> op;
    op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
    op.Process();
}
