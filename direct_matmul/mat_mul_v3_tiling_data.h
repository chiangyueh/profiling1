#ifndef DIRECT_MATMUL_V3_TILING_DATA_H
#define DIRECT_MATMUL_V3_TILING_DATA_H

#include <cstddef>
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// This POD mirrors the CANN 8.1 MatMulV3 tiling ABI.  Both the host runner and
// the directly compiled CANN kernel include this exact definition, so the
// measured candidate is the buffer copied below rather than a tiler lookup.
#pragma pack(push, 8)
struct alignas(8) DirectL2cacheUseInfo {
    uint32_t l2CacheFlag;
};

struct alignas(8) DirectL2cacheTilePara {
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
};

struct alignas(8) DirectMatMulRunInfo {
    uint32_t transA;
    uint32_t transB;
    uint32_t nd2nzA;
    uint32_t nd2nzB;
    uint32_t isHf32;
};

using L2cacheUseInfo = DirectL2cacheUseInfo;
using L2cacheTilePara = DirectL2cacheTilePara;
using MatMulRunInfo = DirectMatMulRunInfo;

struct alignas(8) MatmulTilingData {
    TCubeTiling matmulTiling;
    DirectL2cacheTilePara tileL2cacheTiling;
    DirectMatMulRunInfo matmulRunInfo;
    DirectL2cacheUseInfo l2cacheUseInfo;
    uint32_t baseAN;
    uint32_t baseAD;
    uint32_t baseBN;
    uint32_t baseBD;
};

constexpr uint32_t DIRECT_BALANCED_SCHEDULE_MAGIC = 0x32425342U;
constexpr uint32_t DIRECT_BALANCED_SCHEDULE_VERSION = 2U;
constexpr uint32_t DIRECT_BALANCED_CORE_COUNT = 20U;
constexpr uint32_t DIRECT_BALANCED_GROUP_COUNT = 4U;

struct DirectBalancedTaskRange {
    uint32_t start;
    uint32_t count;
};

struct alignas(8) DirectBalancedSchedule {
    uint32_t magic;
    uint32_t version;
    uint32_t coreCount;
    uint32_t groupCount;
    uint32_t totalTasks;
    uint32_t mCount;
    uint32_t nCount;
    uint32_t reserved;
    DirectBalancedTaskRange ranges[DIRECT_BALANCED_CORE_COUNT]
                                  [DIRECT_BALANCED_GROUP_COUNT];
};

struct alignas(8) DirectBalancedMatmulTilingData {
    MatmulTilingData base;
    DirectBalancedSchedule schedule;
};
#pragma pack(pop)

static_assert(sizeof(TCubeTiling) == 200, "unexpected CANN 8.1 TCubeTiling ABI");
static_assert(offsetof(MatmulTilingData, tileL2cacheTiling) == 200,
              "unexpected CANN 8.1 L2 tiling offset");
static_assert(offsetof(MatmulTilingData, matmulRunInfo) == 224,
              "unexpected CANN 8.1 run-info offset");
static_assert(offsetof(MatmulTilingData, l2cacheUseInfo) == 248,
              "unexpected CANN 8.1 L2-use offset");
static_assert(offsetof(MatmulTilingData, baseAN) == 256,
              "unexpected CANN 8.1 ND2NZ offset");
static_assert(sizeof(MatmulTilingData) == 272,
              "unexpected CANN 8.1 MatmulTilingData ABI");
static_assert(sizeof(DirectBalancedSchedule) == 672,
              "unexpected balanced schedule ABI");
static_assert(offsetof(DirectBalancedMatmulTilingData, schedule) == 272,
              "balanced schedule must follow the retained CANN packet");
static_assert(sizeof(DirectBalancedMatmulTilingData) == 944,
              "unexpected extended balanced tiling ABI");

#endif
