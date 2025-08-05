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
 * \file kernel_matmul_without_que.h
 * \brief
 */

#ifndef ACT_KERNEL_MATMUL_H
#define ACT_KERNEL_MATMUL_H

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "include/utils/common_utils.h"
#include "include/utils/layout_utils.h"
#include "include/utils/tuple_utils.h"
#include "include/utils/coord_utils.h"
#include "include/utils/tensor_utils.h"
#include "include/utils/status_utils.h"
#include "./semaphore.h"
#include "include/matmul/matmul_intf.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/epilogue/block_epilogue_empty.h"
#include "include/matmul/block/block_scheduler_utils.h"
#include "include/matmul/block/block_scheduler_aswt.h"
namespace Act {
namespace Gemm {
namespace Kernel {

template <class ProblemShape, class BlockMmadBuilder, class BlockEpilogue, class BlockScheduler, typename Enable = void>
class KernelMatmulWithoutQue {
    static_assert(AscendC::Std::always_false_v<BlockMmadBuilder>,
                  "BlockMmadBuilder is not implemented for this KernelMatmul.");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulWithoutQue<ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
                             std::enable_if_t<std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    __aicore__ inline KernelMatmulWithoutQue() {}
    __aicore__ inline ~KernelMatmulWithoutQue() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
                                               typename BlockMmadBuilder::L0TileShape, BlockScheduler, transA,
                                               transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    // come from cann
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using BiasType = typename BlockMmadBuilder::BiasType;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    // no need to have tensortrait
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    // shape
    TupleShape problemShape_{};
    bool isBias_ = false;

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline static TupleShape ToShapeTuple(ProblemShape const& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams_ = params.mmadParams;
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(blockMmadParams_.cGmAddr));
        if (blockMmadParams_.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(blockMmadParams_.biasGmAddr));
        }
    }

    __aicore__ inline void run(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);

        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schParams);
        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        int64_t isHf32 = bs.Gethf32Flag();
        if (curBlockIdx >= realBlockNum) {
            return;
        }
        // come from the block_mmad_pingpong_without_que.h
        if (isHf32) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
        blockMmadOp.Init(problemShape_, tileL1, tileL0, isBias_, bs.GetL1BuferNum_());
        // Process tiles in ping-pong mode
        if constexpr (BlockScheduler::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
            blockMmadOp.CopyInB1(bGlobal_, Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_));
        }
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockOffset = GetOffsetWithoutLayout(blockCoord, problemShape_, aGlobal_, bGlobal_, cGlobal_, transA,
                                                      transB, isBias_);
            // calculate block-level offset
            if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
                if (isHf32) {
                    AscendC::SetHF32Mode(0);
                }
                return;
            }
            int64_t offsetA = Get<0>(blockOffset);
            int64_t offsetB = Get<1>(blockOffset);
            int64_t offsetC = Get<2>(blockOffset);
            int64_t offsetBias = Get<3>(blockOffset);
            if constexpr (BlockScheduler::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
                blockMmadOp(cGlobal_[offsetC], aGlobal_[offsetA], blockShape, Get<MNK_K>(problemShape_));
            } else {
                blockMmadOp(cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB], biasGlobal_[offsetBias],
                            blockShape);
            }
        }
        if (isHf32) {
            AscendC::SetHF32Mode(0);
        }
    }

    __host_aicore__ static Status CheckShape(ProblemShape const& shape)
    {
        int64_t m = shape.m;
        int64_t n = shape.n;
        int64_t k = shape.k;
        int64_t b = shape.b;
        if (b > INT32_MAX) {
            return Status::batchErrorExcceedsLimit;
        }
        // Check m, n, k overlimit data type
        if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
            return Status::mnkErrorExceedsLimit;
        }
        // Check matrix size exceeds limit
        if (!transA && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // mk matrix k limit
            return Status::mkErrorMatrixExceedsLimit;
        }

        if (transA && m > MATRIX_INNER_DIM_LIMIT_SIZE) { // km matrix m limit
            return Status::kmErrorMatrixExceedsLimit;
        }
        if (!transB && n > MATRIX_INNER_DIM_LIMIT_SIZE) { // kn matrix n limit
            return Status::knErrorMatrixExceedsLimit;
        }

        if (transB && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // nk matrix k limit
            return Status::nkErrorMatrixExceedsLimit;
        }
        return Status::success;
    }

    __host_aicore__ static Status CheckArgs(Arguments const& args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        CHECK_AND_RETURN(BlockMmadBuilder::CheckArgs(args.mmadArgs));

        return Status::success;
    }

    __host_aicore__ static size_t GetWorkSpaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkSpaceSize();

        return workSpaceSize;
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        // mmad params with epiligue takes workspaceGm as output
        Params params = {args.problemShape, mmadParams, {}};
        return params;
    }

    __aicore__ inline void operator()(Params const& params)
    {
        run(params);
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif