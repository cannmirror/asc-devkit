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
 * \file block_scheduler_l2_misplace_core.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_SCHEDULER_L2_MISPLACE_CORE_H
#define MATMUL_BLOCK_BLOCK_SCHEDULER_L2_MISPLACE_CORE_H
#include "./block_scheduler_utils.h"
#include "../../utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {
enum class L2TilePolicy {
    L2_TILE_NORMAL = 0,   // calc mn l2 tile block nums based on mL2TileNum_ = nL2TileNum_
    L2_TILE_TAIL_OPT = 1, // traversal and find tail optimal solution
};

constexpr int64_t L2_TILE_THRESHOLD = 104857600;
constexpr int64_t L1_MIN_UST_DIM = 4;
constexpr int64_t L1_MAX_UST_DIM = 8;

template <class ProblemShape_, class L1TileShape_, class L0TileShape_,
          L2TilePolicy L2TilePolicy_ = L2TilePolicy::L2_TILE_NORMAL, bool TransA_ = false, bool TransB_ = false>
class BlockSchedulerL2MisplaceCore {
public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t kTileNum_{0};
    int64_t blockIdx_{0};
    int64_t perCoreBlockNum_{0};
    int64_t blockNum_{0};
    int64_t b_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t totalTileNum_{0};
    // l2 spilit attribute
    int64_t newBlockIdx_{0};
    int64_t mL2TileNumTmp_{0};
    int64_t nL2TileNumTmp_{0};
    int64_t nL2Idx_{0};
    int64_t mL2Idx_{0};
    int64_t mL2Num_{0};     // l2 m block num
    int64_t nL2Num_{0};     // l2 n block num
    int64_t mL2TileNum_{0}; // a1b1 m tile num of one l2 block
    int64_t nL2TileNum_{0}; // a1b1 n tile num of one l2 block

    using BlockShape = AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    static constexpr bool isTransA = TransA_;
    static constexpr bool isTransB = TransB_;
    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape_>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape_>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape_>();
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape_>();
    static constexpr L2TilePolicy l2TilePolicy = L2TilePolicy_;

    __aicore__ inline BlockSchedulerL2MisplaceCore(ProblemShape shape, int64_t blockIdx, int64_t blockNum) :
        blockIdx_(blockIdx), blockNum_(blockNum)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        b_ = shape.b ? shape.b : 1;
        mTileNum_ = Act::Gemm::CeilDiv(m_, l1M);
        nTileNum_ = Act::Gemm::CeilDiv(n_, l1N);
        kTileNum_ = Act::Gemm::CeilDiv(k_, l1K);
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mTileNum_, nTileNum_, b_);
        totalTileNum_ = mTileNum_ * nTileNum_ * b_;
        InitL2Tile();
    }

    __aicore__ inline int64_t GetTotalSize(int64_t mL2, int64_t nL2, int64_t kL2)
    {
        int64_t sizeA = mL2 * kL2 * sizeof(half);
        int64_t sizeB = kL2 * nL2 * sizeof(half);
        int64_t sizeC = mL2 * nL2 * sizeof(half);
        return sizeA + sizeB + sizeC;
    }

    __aicore__ inline bool EnableL2Tile()
    {
        return GetTotalSize(m_, n_, k_) > L2_TILE_THRESHOLD;
    }

    __aicore__ inline void InitL2TileTail()
    {
        int64_t mConflict = INT64_MAX;
        int64_t nConflict = INT64_MAX;

        bool isInnerBad = false;
        int64_t maxi = 0;
        int64_t maxj = 0;
        if (l1N > l1M) {
            isInnerBad = isTransA;
            maxi = (blockNum_ > nTileNum_) ? nTileNum_ : blockNum_;
            maxj = (blockNum_ > mTileNum_) ? mTileNum_ : blockNum_;
        } else {
            isInnerBad = !isTransB;
            maxi = (blockNum_ > mTileNum_) ? mTileNum_ : blockNum_;
            maxj = (blockNum_ > nTileNum_) ? nTileNum_ : blockNum_;
        }
        int64_t innerMinUseDim = isInnerBad ? L1_MAX_UST_DIM : L1_MIN_UST_DIM;

        for (int64_t i = maxi; i >= L1_MIN_UST_DIM; i--) { // if l1N greater than l1M, indicates n
            for (int64_t j = maxj; j >= innerMinUseDim; j--) {
                if (GetTotalSize(j * l1M, i * l1N, k_) <= L2_TILE_THRESHOLD) {
                    int64_t mL2TileNumTmp = (l1N > l1M) ? j : i;
                    int64_t nL2TileNumTmp = (l1N > l1M) ? i : j;

                    int64_t mL2TileNumTailTmp = GetTailNum(mTileNum_, mL2TileNumTmp);
                    int64_t nL2TileNumTailTmp = GetTailNum(nTileNum_, nL2TileNumTmp);

                    uint64_t mConflictTmp = Act::Gemm::CeilDiv(blockNum_, mL2TileNumTailTmp);
                    uint64_t nConflictTmp = Act::Gemm::CeilDiv(blockNum_, nL2TileNumTailTmp);
                    if (mConflict >= mConflictTmp && nConflict >= nConflictTmp) {
                        mConflict = mConflictTmp;
                        nConflict = nConflictTmp;
                        mL2TileNum_ = mL2TileNumTmp;
                        nL2TileNum_ = nL2TileNumTmp;
                    }
                }
            }
        }
        if (mL2TileNum_ == 0 || nL2TileNum_ == 0) {
            mL2TileNum_ = mTileNum_;
            nL2TileNum_ = nTileNum_;
        }
    }

    __aicore__ inline void InitL2Tile()
    {
        if ((mTileNum_ < L1_MIN_UST_DIM && nTileNum_ < L1_MIN_UST_DIM) || (!EnableL2Tile())) {
            mL2TileNum_ = mTileNum_;
            nL2TileNum_ = nTileNum_;
            mL2Num_ = 1;
            nL2Num_ = 1;
            return;
        }

        if constexpr (l2TilePolicy == L2TilePolicy::L2_TILE_NORMAL) {
            float p = (l1M + l1N) * k_ / (l1M * l1N);
            // calc x^2 + p * x + (p / 2) ^ 2 = L2_TILE_THRESHOLD / 2mn + (p / 2) ^ 2
            float sqrt_tmp = sqrt(L2_TILE_THRESHOLD / (2 * l1M * l1N) + p * p / 4);
            int64_t l2TileNum = static_cast<int64_t>(sqrt_tmp - p / 2);
            mL2TileNum_ = mTileNum_ >= l2TileNum ? l2TileNum : mTileNum_;
            nL2TileNum_ = nTileNum_ >= l2TileNum ? l2TileNum : nTileNum_;
        } else if constexpr (l2TilePolicy == L2TilePolicy::L2_TILE_TAIL_OPT) {
            InitL2TileTail();
        }

        mL2Num_ = Act::Gemm::CeilDiv(mTileNum_, mL2TileNum_);
        nL2Num_ = Act::Gemm::CeilDiv(nTileNum_, nL2TileNum_);
    }

    __aicore__ inline void GetCommonTileIndex(int64_t tileIdx)
    {
        int64_t batchTileIdx = tileIdx / (nTileNum_ * mTileNum_);
        if (batchTileIdx != 0) {
            tileIdx = tileIdx - batchTileIdx * nTileNum_ * mTileNum_;
        }
        mL2Idx_ = tileIdx / (mL2TileNum_ * nTileNum_);
        mL2TileNumTmp_ = (mL2Idx_ == mL2Num_ - 1) ? GetTailNum(mTileNum_, mL2TileNum_) : mL2TileNum_;

        nL2Idx_ = (tileIdx % (mL2TileNum_ * nTileNum_)) / (mL2TileNumTmp_ * nL2TileNum_);
        nL2TileNumTmp_ = (nL2Idx_ == nL2Num_ - 1) ? GetTailNum(nTileNum_, nL2TileNum_) : nL2TileNum_;

        int64_t startIdx = mL2Idx_ * mL2TileNum_ * nTileNum_ + nL2Idx_ * nL2TileNum_ * mL2TileNumTmp_;
        int64_t startBlockIdx = startIdx % blockNum_;
        newBlockIdx_ = tileIdx - startIdx;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return totalTileNum_;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        GetCommonTileIndex(tileIdx);
        int64_t mTileIdx = newBlockIdx_ % mL2TileNumTmp_;
        mTileIdx = mTileIdx + mL2Idx_ * mL2TileNum_;

        int64_t nTileIdx = 0;
        if (mL2TileNumTmp_ != 0 && nL2TileNumTmp_ != 0) {
            int64_t tmp = newBlockIdx_ / MMLcm(mL2TileNumTmp_, nL2TileNumTmp_);
            nTileIdx = (newBlockIdx_ + tmp) % nL2TileNumTmp_;
        }
        nTileIdx = nTileIdx + nL2Idx_ * nL2TileNum_;

        // calc tail l1block mnk
        int64_t tailL1M = (m_ % l1M == 0) ? l1M : m_ % l1M;
        int64_t tailL1N = (n_ % l1N == 0) ? l1N : n_ % l1N;
        int64_t tailL1K = (k_ % l1K == 0) ? l1K : k_ % l1K;
        int64_t blockShapeM = IsMTail(mTileIdx, mTileNum_) ? tailL1M : l1M;
        int64_t blockShapeN = IsNTail(nTileIdx, nTileNum_) ? tailL1N : l1N;

        return {blockShapeM, blockShapeN, k_, b_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t tileIdx)
    {
        int64_t batchTileIdx = tileIdx / (nTileNum_ * mTileNum_);
        GetCommonTileIndex(tileIdx);
        int64_t mTileIdx = newBlockIdx_ % mL2TileNumTmp_;
        mTileIdx = mTileIdx + mL2Idx_ * mL2TileNum_;

        int64_t nTileIdx = 0;
        if (mL2TileNumTmp_ != 0 && nL2TileNumTmp_ != 0) {
            int64_t tmp = newBlockIdx_ / MMLcm(mL2TileNumTmp_, nL2TileNumTmp_);
            nTileIdx = (newBlockIdx_ + tmp) % nL2TileNumTmp_;
        }
        nTileIdx = nTileIdx + nL2Idx_ * nL2TileNum_;

        return {mTileIdx * l1M, nTileIdx * l1N, 0, batchTileIdx};
    }

    static int64_t GetBlockNum(ProblemShape shape)
    {
        return DoGetBlockNum(l1M, l1N, shape);
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape)
    {
        return 0;
    }

    __host_aicore__ static Status CanImplement(ProblemShape shape)
    {
        return DoCheckArgs(shape, l1M, l1N, l1K, l0M, l0N, l0K);
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Act::Gemm::L2NormMisplaceCoreScheduler,
                              TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerL2MisplaceCore<ProblemShape_, L1TileShape_, L0TileShape_,
                                                     L2TilePolicy::L2_TILE_NORMAL, TransA_, TransB_>;
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Act::Gemm::L2TailOptMisplaceCoreScheduler,
                              TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerL2MisplaceCore<ProblemShape_, L1TileShape_, L0TileShape_,
                                                     L2TilePolicy::L2_TILE_TAIL_OPT, TransA_, TransB_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif