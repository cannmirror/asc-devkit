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
 * \file block_scheduler_gmm_aswt_with_tail_split.h
 * \brief
 */

#ifndef ACT_INCLUDE_MATMUL_BLOCK_SCHEDULER_GMM_ASWT_WITH_TAIL_SPLIT_H
#define ACT_INCLUDE_MATMUL_BLOCK_SCHEDULER_GMM_ASWT_WITH_TAIL_SPLIT_H
#include "include/matmul/block/block_scheduler_utils.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
constexpr int64_t INNER_AXIS_MIN_SPLIT_VAL = 128; // ND2NZ cacheline 128

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
class BlockSchedulerQuantGroupedMatmulAswt {
public:
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    static constexpr int64_t l1M = GetIntegralConstant<0, L1TileShape_>();
    static constexpr int64_t l1N = GetIntegralConstant<1, L1TileShape_>();
    static constexpr int64_t l1K = GetIntegralConstant<2, L1TileShape_>();
    static constexpr int64_t l0M = GetIntegralConstant<0, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<1, L0TileShape_>();
    static constexpr int64_t l0K = GetIntegralConstant<2, L0TileShape_>();

private:
    int64_t mCnt_;
    int64_t nCnt_;
    int64_t totalCnt_;
    int64_t blockNum_;
    int64_t blockIdx_;
    int64_t m_;
    int64_t n_;
    int64_t k_;
    int64_t batch_{0};
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    int32_t mBaseTail_;
    int32_t nBaseTail_;
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t tailCnt_{1}; // only update when last group
    int64_t perCoreBlockNum_;
    int64_t mainMWindow_;
    int64_t tailWindow_;
    int64_t mainRow_;
    int64_t round_;
    uint32_t startBlockIdx_;
    uint32_t endBlockIdx_;

public:
    __aicore__ inline BlockSchedulerQuantGroupedMatmulAswt(int64_t m, int64_t n, int64_t k, int32_t baseM,
                                                           int32_t baseN, int32_t baseK, int64_t blockIdx,
                                                           int64_t blockNum) :
        m_(m), n_(n), k_(k), baseM_(baseM), baseN_(baseN), baseK_(baseK), blockNum_(blockNum), blockIdx_(blockIdx)
    {
        endBlockIdx_ = blockNum - 1;
        mCnt_ = CeilDiv(m_, baseM_);
        nCnt_ = CeilDiv(n_, baseN_);
        mBaseTail_ = m_ - (mCnt_ - 1) * baseM_;
        nBaseTail_ = n_ - (nCnt_ - 1) * baseN_;
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mCnt_, nCnt_);
        totalCnt_ = mCnt_ * nCnt_;
        mainMWindow_ = WINDOW_LEN < mCnt_ ? WINDOW_LEN : mCnt_;
        mainRow_ = mCnt_ / mainMWindow_ - 1;
        tailWindow_ = mCnt_ - mainMWindow_ * mainRow_;
    }

    __aicore__ inline int64_t GetRound()
    {
        return round_;
    }

    // update k, round, startBlockIdx and endBlockIdx when split k
    __aicore__ inline void UpdateSplitKParams(uint32_t k)
    {
        k_ = k;
        round_ = CeilDiv(totalCnt_, blockNum_);
        // the first of blockIdx for new group
        startBlockIdx_ = endBlockIdx_ == blockNum_ - 1 ? 0 : (endBlockIdx_ + 1);
        // the end of blockIdx for new group
        endBlockIdx_ = (totalCnt_ + startBlockIdx_ - 1) % blockNum_;
        // calc real round for new group
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
    }

    __aicore__ inline void UpdateTailTile(uint32_t mTailCnt, uint32_t nTailCnt)
    {
        mTailCnt_ = mTailCnt;
        nTailCnt_ = nTailCnt;
        tailCnt_ = mTailCnt_ * mTailCnt_;
        int64_t newEndBlockIdx = tailCnt_ * (endBlockIdx_ + 1) - 1;
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            round_ += 1;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    __aicore__ inline BlockShape GetTileIdx(int64_t roundIdx)
    {
        int64_t newBlockIdx = blockIdx_ / tailCnt_;
        int64_t index = newBlockIdx + roundIdx * blockNum_;
        int64_t mTileIdx = 0;
        int64_t nTileIdx = 0;
        // add startBlockIdx
        if (blockIdx_ < startBlockIdx_) {
            index += blockNum_ - startBlockIdx_;
        } else {
            index -= startBlockIdx_;
        }
        int64_t rowIdx = index / nCnt_ / mainMWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx = rowIdx * mainMWindow_ + index % mainMWindow_;
            nTileIdx = (index / mainMWindow_) % nCnt_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = index - mainRow_ * mainMWindow_ * nCnt_;
            mTileIdx = mainRow_ * mainMWindow_ + tailIndex % tailWindow_;
            nTileIdx = (tailIndex / tailWindow_) % nCnt_;
        }

        if (rowIdx & 1) {
            nTileIdx = nCnt_ - 1 - nTileIdx;
        }
        return {mTileIdx, nTileIdx, k_, batch_};
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t mTileIdx, int64_t nTileIdx)
    {
        int64_t singleCoreM = mTileIdx != (mCnt_ - 1) ? baseM_ : mBaseTail_;
        int64_t singleCoreN = nTileIdx != (nCnt_ - 1) ? baseN_ : nBaseTail_;
        if (mTailCnt_ == 1 && nTailCnt_ == 1) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        int64_t singleCoreMSplit = (singleCoreM + mTailCnt_ - 1) / mTailCnt_;
        int64_t singleCoreNSplit = (singleCoreN + nTailCnt_ - 1) / nTailCnt_;
        if constexpr (TransA_) { // (k, m)
            singleCoreMSplit = Align(singleCoreMSplit, INNER_AXIS_MIN_SPLIT_VAL);
        }
        if constexpr (!TransB_) { // (k, n)
            singleCoreNSplit = Align(singleCoreNSplit, INNER_AXIS_MIN_SPLIT_VAL);
        }
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        int64_t mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        int64_t nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset >= singleCoreM || nSplitAddrOffset >= singleCoreN) {
            return {0, 0, 0, 0};
        }
        singleCoreM = AscendC::Std::min(singleCoreM - mSplitAddrOffset, singleCoreMSplit);
        singleCoreN = AscendC::Std::min(singleCoreN - nSplitAddrOffset, singleCoreNSplit);
        return {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t mTileIdx, int64_t nTileIdx)
    {
        return {mTileIdx * l1M, nTileIdx * l1N, 0, batch_};
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
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_,
                              Act::Gemm::GroupedMatmulAswtWithTailSplitScheduler, TransA_, TransB_> {
    using SchedulerOp =
        BlockSchedulerQuantGroupedMatmulAswt<ProblemShape_, L1TileShape_, L0TileShape_, TransA_, TransB_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif