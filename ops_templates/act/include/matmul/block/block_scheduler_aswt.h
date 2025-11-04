/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_aswt.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_SCHEDULER_ASWT_H
#define MATMUL_BLOCK_BLOCK_SCHEDULER_ASWT_H

#include "./block_scheduler_utils.h"
#include "./block_scheduler_policy.h"
#include "../../utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class BlockSchedulerAswt {
public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t kTileNum_{0};
    int64_t blockIdx_{0};
    int64_t perCoreBlockNum_{0};
    int64_t blockNum_{0};
    int64_t b_{0};
    int64_t k_{0};
    int64_t tailL1M_{0};
    int64_t tailL1N_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t tailCnt_{1};
    int64_t tileNum_{1};
    int64_t mainWindow_{1};
    int64_t mainRow_{1};
    int64_t tailWindow_{1};
    int64_t mTileIdx_{1};
    int64_t nTileIdx_{1};
    int64_t lastTileIdx_{-1};
    int64_t nSplitOffset_{0};
    int64_t mSplitOffset_{0};

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape_>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape_>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape_>();
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape_>();

public:
    __aicore__ inline BlockSchedulerAswt(ProblemShape shape, int64_t blockIdx, int64_t blockNum) :
        blockIdx_(blockIdx), blockNum_(blockNum)
    {
        k_ = shape.k;
        b_ = AscendC::Std::max(shape.b, 1L);
        mTileNum_ = Act::Gemm::CeilDiv(shape.m, l1M);
        nTileNum_ = Act::Gemm::CeilDiv(shape.n, l1N);
        kTileNum_ = Act::Gemm::CeilDiv(k_, l1K);
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mTileNum_, nTileNum_, b_);
        tileNum_ = mTileNum_ * nTileNum_ * b_;
        int64_t tailTileNum = tileNum_ % blockNum_;
        int32_t tailCnt = 1;
        tailL1M_ = shape.m - (mTileNum_ - 1) * l1M;
        tailL1N_ = shape.n - (nTileNum_ - 1) * l1N;
        if (tailTileNum > 0 && b_ == 1) {
            tailCnt = blockNum_ / tailTileNum;
            for (int32_t i = 1; i * i <= tailCnt; i++) { nTailCnt_ = i; }
            mTailCnt_ = tailCnt / nTailCnt_;
            // tail tile num
            mTailCnt_ = AscendC::Std::min(tailL1M_, mTailCnt_);
            nTailCnt_ = AscendC::Std::min(tailL1N_, nTailCnt_);
            tailCnt_ = mTailCnt_ * nTailCnt_;
            tileNum_ += (tailCnt_ - 1) * tailTileNum;
        }
        mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainRow_ * mainWindow_;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return tileNum_;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t blkM = (mTileIdx_ == (mTileNum_ - 1)) ? tailL1M_ : l1M;
        int64_t blkN = (nTileIdx_ == (nTileNum_ - 1)) ? tailL1N_ : l1N;
        if (tileIdx / blockNum_ != (perCoreBlockNum_ - 1) || tailCnt_ == 1) {
            return {blkM, blkN, k_, b_};
        }
        int64_t splitBlkM = Act::Gemm::CeilDiv(blkM, mTailCnt_);
        int64_t splitBlkN = Act::Gemm::CeilDiv(blkN, nTailCnt_);
        mTailCnt_ = Act::Gemm::CeilDiv(blkM, splitBlkM);
        nTailCnt_ = Act::Gemm::CeilDiv(blkN, splitBlkN);
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        mSplitOffset_ = mSplitIdx * splitBlkM;
        nSplitOffset_ = nSplitIdx * splitBlkN;
        if (mSplitOffset_ >= blkM || nSplitOffset_ >= blkN) {
            return {0, 0, k_, b_};
        }
        splitBlkM = AscendC::Std::min(blkM - mSplitOffset_, splitBlkM);
        splitBlkN = AscendC::Std::min(blkN - nSplitOffset_, splitBlkN);
        return {splitBlkM, splitBlkN, k_, b_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t batchIdx = 0;
        if (b_ > 1) {
            batchIdx = tileIdx / (mTileNum_ * nTileNum_);
        }

        return {mTileIdx_ * l1M + mSplitOffset_, nTileIdx_ * l1N + nSplitOffset_, 0, batchIdx};
    }

private:
    __aicore__ inline void UpdateMNTileIdx(int64_t tmpIdx)
    {
        if (lastTileIdx_ == tmpIdx) {
            return;
        }
        lastTileIdx_ = tmpIdx;

        int64_t tileIdx = tmpIdx;
        if (tileIdx / blockNum_ == (perCoreBlockNum_ - 1) && tailCnt_ > 1) {
            tileIdx = (perCoreBlockNum_ - 1) * blockNum_ + blockIdx_ / tailCnt_;
        }
        int64_t rowIdx = tileIdx / nTileNum_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx_ = rowIdx * mainWindow_ + tileIdx % mainWindow_;
            nTileIdx_ = (tileIdx / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = tileIdx - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx_ = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx_ = (tailIndex / tailWindow_) % nTileNum_;
        }
        if (rowIdx % 2 != 0) { // 2: mode 2 means even row, need reverse scan
            nTileIdx_ = nTileNum_ - 1 - nTileIdx_;
        }
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Act::Gemm::AswtScheduler, TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerAswt<ProblemShape_, L1TileShape_, L0TileShape_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
