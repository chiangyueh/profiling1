#include "kernel_operator.h"

// AscendC's generated host-only translation unit re-includes the original
// source without the target compile definitions.  Its generated launch symbol
// is available as a macro, so recover the exact fixed variant here as well.
#if !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp32_k41)
#define MATMUL_C220_SUFFIX_VALUE 41
#define MATMUL_DIRECT_KERNEL direct_matmul_fp32_k41
#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp32_k51)
#define MATMUL_C220_SUFFIX_VALUE 51
#define MATMUL_DIRECT_KERNEL direct_matmul_fp32_k51
#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp16_k60)
#define MATMUL_C220_SUFFIX_VALUE 60
#define MATMUL_DIRECT_KERNEL direct_matmul_fp16_k60
#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_Y half
#define DTYPE_BIAS half
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_bf16_k60)
#define MATMUL_C220_SUFFIX_VALUE 60
#define MATMUL_DIRECT_KERNEL direct_matmul_bf16_k60
#define DTYPE_X1 bfloat16_t
#define DTYPE_X2 bfloat16_t
#define DTYPE_Y bfloat16_t
#define DTYPE_BIAS bfloat16_t
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp16_k61)
#define MATMUL_C220_SUFFIX_VALUE 61
#define MATMUL_DIRECT_KERNEL direct_matmul_fp16_k61
#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_Y half
#define DTYPE_BIAS half
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_bf16_k61)
#define MATMUL_C220_SUFFIX_VALUE 61
#define MATMUL_DIRECT_KERNEL direct_matmul_bf16_k61
#define DTYPE_X1 bfloat16_t
#define DTYPE_X2 bfloat16_t
#define DTYPE_Y bfloat16_t
#define DTYPE_BIAS bfloat16_t
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp16_k121)
#define MATMUL_C220_SUFFIX_VALUE 121
#define MATMUL_DIRECT_KERNEL direct_matmul_fp16_k121
#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_Y half
#define DTYPE_BIAS half
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_bf16_k121)
#define MATMUL_C220_SUFFIX_VALUE 121
#define MATMUL_DIRECT_KERNEL direct_matmul_bf16_k121
#define DTYPE_X1 bfloat16_t
#define DTYPE_X2 bfloat16_t
#define DTYPE_Y bfloat16_t
#define DTYPE_BIAS bfloat16_t
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp16_k100001)
#define MATMUL_C220_SUFFIX_VALUE 100001
#define MATMUL_DIRECT_KERNEL direct_matmul_fp16_k100001
#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_Y half
#define DTYPE_BIAS half
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp32_k202)
#define MATMUL_C220_SUFFIX_VALUE 202
#define MATMUL_DIRECT_KERNEL direct_matmul_fp32_k202
#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp32_k20030)
#define MATMUL_C220_SUFFIX_VALUE 20030
#define MATMUL_DIRECT_KERNEL direct_matmul_fp32_k20030
#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp32_k20031)
#define MATMUL_C220_SUFFIX_VALUE 20031
#define MATMUL_DIRECT_KERNEL direct_matmul_fp32_k20031
#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp32_k90001)
#define MATMUL_C220_SUFFIX_VALUE 90001
#define MATMUL_DIRECT_KERNEL direct_matmul_fp32_k90001
#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float
#elif !defined(MATMUL_C220_SUFFIX_VALUE) && defined(direct_matmul_fp32_k90002)
#define MATMUL_C220_SUFFIX_VALUE 90002
#define MATMUL_DIRECT_KERNEL direct_matmul_fp32_k90002
#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float
#endif

#if !defined(MATMUL_C220_SUFFIX_VALUE)
// The framework performs one additional host-only syntax pass without target
// definitions.  This body is never embedded as the device implementation; a
// fixed, valid specialization keeps that pass independent of target metadata.
#define MATMUL_C220_SUFFIX_VALUE 41
#define MATMUL_DIRECT_KERNEL matmul_c220_host_syntax_only
#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float
#endif

#include "mat_mul_v3_tiling_data_280.h"

#if MATMUL_C220_SUFFIX_VALUE == 41
#include "mat_mul_multi_core_splitk_kernel.h"
#elif MATMUL_C220_SUFFIX_VALUE == 51
#include "mat_mul_sc_splitk_kernel.h"
#elif MATMUL_C220_SUFFIX_VALUE == 60
#include "mat_mul_unaligned_sc_splitk_kernel_gm_to_l1.h"
#elif MATMUL_C220_SUFFIX_VALUE == 61
#include "mat_mul_sc_splitk_kernel_gm_to_l1.h"
#elif MATMUL_C220_SUFFIX_VALUE == 121
#include "mat_mul_sc_splitk_al1_fullload_kernel.h"
#elif MATMUL_C220_SUFFIX_VALUE == 100001
#include "mat_mul_base_kernel.h"
#elif MATMUL_C220_SUFFIX_VALUE == 202
#include "mat_mul_cvp_base_kernel.h"
#elif MATMUL_C220_SUFFIX_VALUE == 20030
#include "mat_mul_unaligned_deterministic_splitk_kernel.h"
#elif MATMUL_C220_SUFFIX_VALUE == 20031
#include "mat_mul_deterministic_splitk_kernel.h"
#elif MATMUL_C220_SUFFIX_VALUE == 90001 || MATMUL_C220_SUFFIX_VALUE == 90002
#include "direct_init_split_k_kernel.h"
#else
#error Unsupported C220 direct family suffix
#endif

using namespace AscendC;
using namespace matmul;

__aicore__ inline void DirectReadMatmulTiling280(
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

#define MATMUL_DIRECT_DECLARE_TYPES(trans_b_value)                                    \
    using aType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;         \
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X2, trans_b_value>; \
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_Y>;                  \
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_BIAS>

#define MATMUL_DIRECT_DECLARE_NZ_C_TYPES(trans_b_value)                               \
    using aType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;         \
    using bType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_X2, trans_b_value>; \
    using cType = MatmulType<TPosition::GM, CubeFormat::NZ, DTYPE_Y>;                  \
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_BIAS>

extern "C" __global__ __aicore__ void MATMUL_DIRECT_KERNEL(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    MatmulTilingData tilingData;
    DirectReadMatmulTiling280(tilingGM, tilingData);
    __gm__ uint8_t *user = workspaceGM;

#if MATMUL_C220_SUFFIX_VALUE == 41
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);
    MATMUL_DIRECT_DECLARE_TYPES(true);
    MatmulV3::MatMulMultiCoreSplitK<aType, bType, cType, biasType,
                                    FIXPIPE_OPT_SELECT::BASE>(
        aGM, bGM, cGM, biasGM, tilingData, user);
#elif MATMUL_C220_SUFFIX_VALUE == 51
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    MATMUL_DIRECT_DECLARE_TYPES(true);
    TPipe pipe;
    MatMulSingleCoreSplitKKernel<aType, bType, cType, biasType,
        MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_NK, true> op;
    op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
    op.Process();
#elif MATMUL_C220_SUFFIX_VALUE == 60
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    MATMUL_DIRECT_DECLARE_TYPES(false);
    MatMulUnAlignedSingleCoreSplitKKernelGmToL1<aType, bType, cType, biasType,
        MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK> op;
    op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
    op.Process();
#elif MATMUL_C220_SUFFIX_VALUE == 61
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    MATMUL_DIRECT_DECLARE_TYPES(false);
    MatMulSingleCoreSplitKKernelGmToL1<aType, bType, cType, biasType,
        MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK> op;
    op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
    op.Process();
#elif MATMUL_C220_SUFFIX_VALUE == 121
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if (tilingData.matmulRunInfo.transB == 0) {
        MATMUL_DIRECT_DECLARE_TYPES(false);
        MatMulSingleCoreSplitKAL1FullLoadKernel<aType, bType, cType, biasType,
            MatmulSingleCoreSplitKAL1FullLoadBlock, MM_CFG_NO_PRELOAD> op;
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
        op.Process();
    } else {
        MATMUL_DIRECT_DECLARE_TYPES(true);
        MatMulSingleCoreSplitKAL1FullLoadKernel<aType, bType, cType, biasType,
            MatmulSingleCoreSplitKAL1FullLoadBlock, MM_CFG_NO_PRELOAD> op;
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
        op.Process();
    }
#elif MATMUL_C220_SUFFIX_VALUE == 100001
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    TPipe pipe;
    if (tilingData.matmulRunInfo.transB == 0) {
        MATMUL_DIRECT_DECLARE_TYPES(false);
        MatmulBaseKernel<aType, bType, cType, biasType,
            MatmulBaseBlock, MM_CFG_K_SHIFT> op;
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
        op.Process();
    } else {
        MATMUL_DIRECT_DECLARE_TYPES(true);
        MatmulBaseKernel<aType, bType, cType, biasType,
            MatmulBaseBlock, MM_CFG_K_SHIFT> op;
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
        op.Process();
    }
#elif MATMUL_C220_SUFFIX_VALUE == 202
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if (tilingData.matmulRunInfo.transB == 0) {
        MATMUL_DIRECT_DECLARE_TYPES(false);
        MatmulV3::MatmulCvpBaseKernel<aType, bType, cType, biasType,
            MatmulBaseBlock, MM_CFG_NO_PRELOAD> op;
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
        op.Process();
    } else {
        MATMUL_DIRECT_DECLARE_TYPES(true);
        MatmulV3::MatmulCvpBaseKernel<aType, bType, cType, biasType,
            MatmulBaseBlock, MM_CFG_NO_PRELOAD> op;
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);
        op.Process();
    }
#elif MATMUL_C220_SUFFIX_VALUE == 20030
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    MATMUL_DIRECT_DECLARE_NZ_C_TYPES(false);
    MatMulUnAlignedKernelDeterministicSplitK<aType, bType, cType, biasType,
        FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT>(
        aGM, bGM, cGM, biasGM, tilingData, user);
#elif MATMUL_C220_SUFFIX_VALUE == 20031
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    MATMUL_DIRECT_DECLARE_NZ_C_TYPES(false);
    MatMulKernelDeterministicSplitK<aType, bType, cType, biasType,
        FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT>(
        aGM, bGM, cGM, biasGM, tilingData, user);
#elif MATMUL_C220_SUFFIX_VALUE == 90001 || MATMUL_C220_SUFFIX_VALUE == 90002
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);
    MATMUL_DIRECT_DECLARE_TYPES(true);
    ExperimentalDerivativeMatmul::RunDirectInitSplitK<aType, bType, cType, biasType>(
        aGM, bGM, cGM, tilingData);
#endif
}

#undef MATMUL_DIRECT_DECLARE_TYPES
#undef MATMUL_DIRECT_DECLARE_NZ_C_TYPES
