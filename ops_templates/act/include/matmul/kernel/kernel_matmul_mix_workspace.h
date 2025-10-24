/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_matmul.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_MATMUL_MIX_WORKSPACE_H
#define MATMUL_KERNEL_KERNEL_MATMUL_MIX_WORKSPACE_H

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "../../utils/common_utils.h"
#include "../../utils/layout_utils.h"
#include "../../utils/tuple_utils.h"
#include "../../utils/coord_utils.h"
#include "../../utils/tensor_utils.h"
#include "../../utils/status_utils.h"

#include "./semaphore.h"
#include "../matmul_intf.h"
#include "../block/block_mmad_builder.h"
#include "../../epilogue/block_epilogue_empty.h"
#include "../../epilogue/block_epilogue_quant.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_iterateK.h"
#include "../block/block_scheduler_misplace_core.h"
#include "../block/block_scheduler_l2_misplace_core.h"

namespace Act {
namespace Gemm {
namespace Kernel {
template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulMixWorkspace {
public:
    __aicore__ inline KernelMatmulMixWorkspace() {}
    __aicore__ inline ~KernelMatmulMixWorkspace() {}

    using BlockEpilogue = BlockEpilogue_;
    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    static constexpr int64_t l1M = BlockMmadBuilder::l1M;
    static constexpr int64_t l1N = BlockMmadBuilder::l1N;
    static constexpr int64_t l1K = BlockMmadBuilder::l1K;
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
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;
    using ATensorTrait = AscendC::TensorTrait<AType, AscendC::TPosition::GM, NDLayout>;
    using BTensorTrait = AscendC::TensorTrait<BType, AscendC::TPosition::GM, NDLayout>;
    using CTensorTrait = AscendC::TensorTrait<CType, AscendC::TPosition::GM, NDLayout>;
    using AGlobalTensorType = AscendC::GlobalTensor<ATensorTrait>;
    using BGlobalTensorType = AscendC::GlobalTensor<BTensorTrait>;
    using CGlobalTensorType = AscendC::GlobalTensor<CTensorTrait>;
    // attribute
    AGlobalTensorType aGlobal_;
    BGlobalTensorType bGlobal_;
    CGlobalTensorType cGlobal_;
    // mmad
    BlockMmadParams blockMmadParams_{};
    // shape
    TupleShape problemShape_{};

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
        Params() = default;
    };

    __aicore__ inline static TupleShape ToShapeTuple(ProblemShape const& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        blockMmadParams_ = params.mmadParams;
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        // Init Tensor
        InitGlobalTensorA<NDLayout, AGlobalTensorType, ATensorTrait, AType>(aGlobal_, blockMmadParams_.aGmAddr, transA,
                                                                            m, k);
        InitGlobalTensorB<NDLayout, BGlobalTensorType, BTensorTrait, BType>(bGlobal_, blockMmadParams_.bGmAddr, transB,
                                                                            n, k);
        InitGlobalTensorC<NDLayout, CGlobalTensorType, CTensorTrait, CType>(cGlobal_, blockMmadParams_.cGmAddr, m, n);
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

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        CHECK_AND_RETURN(BlockMmadBuilder::CanImplement(args.mmadArgs));
        // Check args for block scheduler
        CHECK_AND_RETURN(BlockSchedulerOp::CanImplement(args.problemShape));
        // Check args fro block epilogue
        CHECK_AND_RETURN(BlockEpilogue::CanImplement(args.epilogueArgs));
        return Status::success;
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkspaceSize();
        // Calculate extra workspace size for epilogue
        workSpaceSize += BlockEpilogue::GetWorkspaceSize(blockNum, l1M, l1N);
        // Calculate extra workspace size for block scheduler
        workSpaceSize += BlockSchedulerOp::GetWorkspaceSize(shape);
        return workSpaceSize;
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        // mmad params with epiligue takes workspaceGm as output
        mmadParams.cGmAddr = workspace;
        // epilogue params takes workspaceGm as input
        BlockEpilogueParams epilogueParams = BlockEpilogue::InitParams(args.epilogueArgs, workspace);
        Params params = {args.problemShape, mmadParams, epilogueParams};
        return params;
    }

    static int64_t GetBlockNum(ProblemShape shape)
    {
        return BlockSchedulerOp::GetBlockNum(shape);
    }

    __aicore__ inline void operator()(Params const& params)
    {
        // Instantiate mmadOp and epilogueOp
        BlockMmadOp blockMmadOp;
        BlockEpilogue epilogueOp;
        // Get blockIdx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        if ASCEND_IS_AIV {
            curBlockIdx /= AscendC::GetTaskRation();
        }
        if (curBlockIdx >= blockNum) {
            return;
        }
        // Init
        Init(params);
        blockMmadOp.Init();
        epilogueOp.Init(params.epilogueParams, l1M, l1N, problemShape_);
        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum);

        int64_t tileNum = bs.GetTileNum();
        // Send event when using aiv_1
        if (AscendC::GetSubBlockIdx() > 0) {
            SendEvent<BlockEpilogue>(curBlockIdx, tileNum, blockNum);
            return;
        }

        // Process tiles in ping-pong mode
        int64_t loopIdx = 0;
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockOffset = GetOffset(blockCoord, problemShape_, aGlobal_, bGlobal_, cGlobal_, transA, transB);
            // calculate block-level offset
            int64_t workspaceOffset = GetWorkspaceOffset(loopIdx & PINGPONG_FLAG, curBlockIdx, l1M, l1N);
            // AIC Process
            if ASCEND_IS_AIC {
                // Synchronize with aiv
                if (loopIdx >= FIRST_PINGPONG) {
                    AicWaitAiv<BlockEpilogue>(loopIdx);
                }
                auto aGlobalT = aGlobal_[Get<0>(blockOffset)];
                auto bGlobalT = bGlobal_[Get<1>(blockOffset)];
                // Compute block-level mmad with epilogue
                auto workspaceGlobal =
                    GetWorkSpaceGlobal<NDLayout, CTensorTrait, CType, BlockShape>(blockShape, blockMmadParams_.cGmAddr);
                auto workspaceGlobalT = workspaceGlobal[workspaceOffset];
                blockMmadOp.IterateAll(workspaceGlobalT, aGlobalT, bGlobalT, blockShape);
                // Notify aiv
                AicNotifyAiv<BlockEpilogue>(loopIdx);
            }
            // AIV Process
            if ASCEND_IS_AIV {
                // Synchronize with aic
                AivWaitAic<BlockEpilogue>(loopIdx);
                // Calulate epilogue
                epilogueOp(blockShape, blockCoord, Get<2>(blockOffset), workspaceOffset);
                // Notify aic
                AivNotifyAic<BlockEpilogue>(loopIdx);
            }
            loopIdx += 1;
        }
        // Match extra event after aic process finished
        if ASCEND_IS_AIC {
            AicWaitEvent<BlockEpilogue>(loopIdx);
        }
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif