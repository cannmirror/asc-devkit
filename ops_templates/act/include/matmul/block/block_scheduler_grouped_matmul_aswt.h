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
 * \file block_scheduler_grouped_matmul_aswt.h
 * \brief
 */

#ifndef ACT_BLOCK_GROUPED_MATMUL_SCHEDULER_H
#define ACT_BLOCK_GROUPED_MATMUL_SCHEDULER_H
#include "include/matmul/block/block_scheduler_utils.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {

#ifndef TILING_TYPE
#if defined(CONST_TILING)
#define TILING_TYPE const int32_t
#else
#define TILING_TYPE __gm__ int32_t
#endif
#endif // TILING_TYPE

template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class BlockSchedulerGroupedMatmulAswt {
private:
    const int64_t WINDOW_LEN = 4;
    const int64_t EVEN_ROWS = 2;

public:
    int64_t mTileNum{0};
    int64_t nTileNum{0};
    int64_t totalTileNum{0};

    int64_t blockNum{0};
    int64_t blockIdx{0};
    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t b{1};
    int32_t baseM{0};
    int32_t baseN{0};
    int32_t baseK{0};
    uint64_t mTailCnt{1};
    uint64_t nTailCnt{1};
    uint64_t tailCnt{1};
    int64_t perCoreBlockNum{0};
    int64_t mainWindow{0};
    int64_t tailWindow{0};
    int64_t mainRow{0};

    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape_>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape_>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape_>();
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape_>();

public:
    __aicore__ inline BlockSchedulerGroupedMatmulAswt(int64_t m_, int64_t n_, int64_t k_, int32_t baseM_,
                                                      int32_t baseN_, int32_t baseK_, int64_t blockIdx_,
                                                      int64_t blockNum_, uint64_t mTailCnt_, uint64_t nTailCnt_) :
        m(m_), n(n_), k(k_), baseM(baseM_), baseN(baseN_), baseK(baseK_), blockNum(blockNum_), blockIdx(blockIdx_),
        mTailCnt(mTailCnt_), nTailCnt(nTailCnt_)
    {
        mTileNum = Act::Gemm::CeilDiv(m, baseM);
        nTileNum = Act::Gemm::CeilDiv(n, baseN);
        perCoreBlockNum = GetPerBlockNum(blockNum, mTileNum, nTileNum, b);
        totalTileNum = mTileNum * nTileNum;
        if ((mTailCnt > 1 || nTailCnt > 1)) {
            tailCnt = mTailCnt * nTailCnt;
            totalTileNum += (tailCnt - 1) * (totalTileNum % blockNum);
        }
        mainWindow = WINDOW_LEN < mTileNum ? WINDOW_LEN : mTileNum;
        mainRow = mTileNum / mainWindow - 1;
        tailWindow = mTileNum - mainWindow * mainRow;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return totalTileNum;
    }

    __aicore__ inline BlockShape GetTileIdx(int64_t curBlock, int64_t count)
    {
        uint64_t index = curBlock - count;
        uint64_t mTileIdx = 0;
        uint64_t nTileIdx = 0;
        if (index / blockNum == (perCoreBlockNum - 1) && tailCnt > 1) {
            index = (perCoreBlockNum - 1) * blockNum + blockIdx / tailCnt;
        }
        uint64_t rowIdx = index / nTileNum / mainWindow;
        if (rowIdx < mainRow) {
            mTileIdx = rowIdx * mainWindow + index % mainWindow;
            nTileIdx = (index / mainWindow) % nTileNum;
        } else {
            rowIdx = mainRow;
            uint64_t tailIndex = index - mainRow * mainWindow * nTileNum;
            mTileIdx = mainRow * mainWindow + tailIndex % tailWindow;
            nTileIdx = (tailIndex / tailWindow) % nTileNum;
        }
        if (rowIdx % EVEN_ROWS != 0) { // Reverse computation for even-numbered rows
            nTileIdx = nTileNum - 1 - nTileIdx;
        }
        return {mTileIdx, nTileIdx, k, b};
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t mTileIdx, int64_t nTileIdx, int64_t cureBlock, int64_t c0,
                                               bool weightNzFlag = false)
    {
        int64_t tailL1M = (m % baseM == 0) ? baseM : m % baseM;
        int64_t tailL1N = (n % baseN == 0) ? baseN : n % baseN;
        int64_t blockShapeM = IsMTail(mTileIdx, mTileNum) ? tailL1M : baseM;
        int64_t blockShapeN = IsNTail(nTileIdx, nTileNum) ? tailL1N : baseN;
        int64_t mSplitAddrOffset = 0;
        int64_t nSplitAddrOffset = 0;
        if (cureBlock / blockNum != (perCoreBlockNum - 1) || tailCnt == 1) {
            return {blockShapeM, blockShapeN, mSplitAddrOffset, nSplitAddrOffset};
        }
        int64_t singleCoreMSplit = Act::Gemm::CeilDiv(baseM, mTailCnt);
        int64_t singleCoreNSplit = Act::Gemm::CeilDiv(baseN, nTailCnt);
        if (weightNzFlag) {
            singleCoreNSplit = Act::Gemm::CeilDiv(singleCoreNSplit, c0);
        }
        mTailCnt = Act::Gemm::CeilDiv(baseM, singleCoreMSplit);
        nTailCnt = Act::Gemm::CeilDiv(baseN, singleCoreNSplit);
        int64_t mSplitIdx = (blockIdx % tailCnt) % mTailCnt;
        int64_t nSplitIdx = (blockIdx % tailCnt) / mTailCnt;
        mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset >= baseM || nSplitAddrOffset >= baseN) {
            return {0, 0, 0, 0};
        }
        tailL1M = AscendC::Std::min(baseM - mSplitAddrOffset, singleCoreMSplit);
        tailL1N = AscendC::Std::min(baseN - nSplitAddrOffset, singleCoreNSplit);
        return {tailL1M, tailL1N, mSplitAddrOffset, nSplitAddrOffset};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t mTileIdx, int64_t nTileIdx)
    {
        return {mTileIdx * l1M, nTileIdx * l1N, 0, b};
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
        return Status::success;
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Act::Gemm::GroupedMatmulAswtScheduler, TransA_,
                              TransB_> {
    using SchedulerOp = BlockSchedulerGroupedMatmulAswt<ProblemShape_, L1TileShape_, L0TileShape_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif