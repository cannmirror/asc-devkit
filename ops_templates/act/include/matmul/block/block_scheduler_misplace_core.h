/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_misplace_core.h
 * \brief
 */

#ifndef ACT_BLOCK_SCHEDULER_MISPLACE_CORE_H
#define ACT_BLOCK_SCHEDULER_MISPLACE_CORE_H

#include "include/matmul/block/block_scheduler_utils.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class BlockSchedulerMisplaceCore {
public:
    int64_t mTileNum{0};
    int64_t nTileNum{0};
    int64_t kTileNum{0};
    int64_t blockIdx{0};
    int64_t perCoreBlockNum{0};
    int64_t blockNum{0};
    int64_t b{0};
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t totalTileNum{0};

    using BlockShape = Std::tuple<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Std::tuple<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape_>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape_>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape_>();
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape_>();

    __aicore__ inline BlockSchedulerMisplaceCore(ProblemShape shape, int64_t blockIdx, int64_t blockNum) :
        blockIdx(blockIdx), blockNum(blockNum)
    {
        m = shape.m;
        n = shape.n;
        k = shape.k;
        b = shape.b ? shape.b : 1;
        mTileNum = Act::Gemm::CeilDiv(m, l1M);
        nTileNum = Act::Gemm::CeilDiv(n, l1N);
        kTileNum = Act::Gemm::CeilDiv(k, l1K);
        perCoreBlockNum = GetPerBlockNum(blockNum, mTileNum, nTileNum, b);
        totalTileNum = mTileNum * nTileNum * b;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return totalTileNum;
    }

    __aicore__ inline BlockShape GetBlockShape(int tileIdx)
    {
        // calc tail l1block mnk
        int64_t tailL1M = (m % l1M == 0) ? l1M : m % l1M;
        int64_t tailL1N = (n % l1N == 0) ? l1N : n % l1N;
        int64_t tailL1K = (k % l1K == 0) ? l1K : k % l1K;
        int mTileIdx = tileIdx % mTileNum;
        int64_t batchTileIdx = tileIdx / (mTileNum * nTileNum);
        mTileIdx = mTileIdx - batchTileIdx * mTileNum;
        int64_t nTileIdx = 0;
        if (mTileNum != 0 && nTileNum != 0) {
            int64_t tmp = tileIdx / MMLcm(mTileNum, nTileNum);
            nTileIdx = (tileIdx + tmp) % nTileNum;
        }

        int64_t blockShapeM = IsMTail(mTileIdx, mTileNum) ? tailL1M : l1M;
        int64_t blockShapeN = IsNTail(nTileIdx, nTileNum) ? tailL1N : l1N;
        return {blockShapeM, blockShapeN, k, b};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        int mTileIdx = tileIdx % mTileNum;
        int64_t batchTileIdx = tileIdx / (mTileNum * nTileNum);
        mTileIdx = mTileIdx - batchTileIdx * mTileNum;
        int64_t nTileIdx = 0;
        if (mTileNum != 0 && nTileNum != 0) {
            int64_t tmp = tileIdx / MMLcm(mTileNum, nTileNum);
            nTileIdx = (tileIdx + tmp) % nTileNum;
        }
        return {mTileIdx * l1M, nTileIdx * l1N, 0, batchTileIdx};
    }

    static int64_t GetBlockNum(ProblemShape shape)
    {
        return DoGetBlockNum(l1M, l1N, shape);
    }

    __host_aicore__ static size_t GetWorkSpaceSize(ProblemShape shape)
    {
        return 0;
    }

    __host_aicore__ static Status CheckArgs(ProblemShape shape)
    {
        return DoCheckArgs(shape, l1M, l1N, l1K, l0M, l0N, l0K);
    }
};

// Specialized selector for iterateK template
template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, MisplaceCoreScheduler, TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerIterateK<ProblemShape_, L1TileShape_, L0TileShape_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif