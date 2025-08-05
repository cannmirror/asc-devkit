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
 * \file kernel_matmul.h
 * \brief
 */

#ifndef ACT_KERNEL_MATMUL_MIX_H
#define ACT_KERNEL_MATMUL_MIX_H

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "include/utils/common_utils.h"
#include "include/utils/layout_utils.h"
#include "include/utils/tuple_utils.h"
#include "include/utils/coord_utils.h"
#include "include/utils/tensor_utils.h"
#include "include/utils/status_utils.h"
#include "include/utils/device_utils.h"

#include "include/matmul/matmul_intf.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/epilogue/block_epilogue.h"
#include "include/matmul/block/block_scheduler_utils.h"
#include "include/matmul/block/block_scheduler_aswt.h"
#include "include/matmul/block/block_scheduler_iterateK.h"

namespace Act {
namespace Gemm {
namespace Kernel {

constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
const int16_t AIV_SYNC_AIC_FLAG = 6;
const int16_t AIC_SYNC_AIV_FLAG = 8;
const int16_t FLAG_ID_MAX = 16;

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulMix {
public:
    __aicore__ inline KernelMatmulMix() {}
    __aicore__ inline ~KernelMatmulMix() {}

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
    using TileCopyParam = typename BlockMmadBuilder::TileCopyParam;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using YType = typename BlockEpilogue::DataTypeOut;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;
    using ATensorTrait = TensorTrait<AType, AscendC::TPosition::GM, NDLayout>;
    using BTensorTrait = TensorTrait<BType, AscendC::TPosition::GM, NDLayout>;
    using YTensorTrait = TensorTrait<YType, AscendC::TPosition::GM, NDLayout>;

    using AGlobalTensorType = AscendC::GlobalTensor<ATensorTrait>;
    using BGlobalTensorType = AscendC::GlobalTensor<BTensorTrait>;
    using YGlobalTensorType = AscendC::GlobalTensor<YTensorTrait>;
    // attribute
    AGlobalTensorType aGlobal_;
    BGlobalTensorType bGlobal_;
    YGlobalTensorType yGlobal_;
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
        InitGlobalTensorC<NDLayout, YGlobalTensorType, YTensorTrait, YType>(yGlobal_, params.epilogueParams.outGmAddr,
                                                                            m, n);
    }

    __aicore__ inline void Run(Params const& params)
    {
        // Instantiate mmadOp and epilogueOp
        BlockMmadOp blockMmadOp;
        BlockEpilogue epilogueOp;
        // Get blockIdx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        bool enable2UB =
            AscendC::Std::is_same_v<TileCopyParam, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitMWithParams>>;
        if ASCEND_IS_AIV {
            if (!enable2UB && GetSubBlockIdx() > 0) {
                return;
            }
            curBlockIdx /= AscendC::GetTaskRation();
        }
        if (curBlockIdx >= blockNum) {
            return;
        }
        // Init
        Init(params);
        blockMmadOp.Init();
        int64_t calM = enable2UB ? Act::Gemm::CeilDiv(l1M, AscendC::GetTaskRation()) : l1M;
        epilogueOp.Init(params.epilogueParams, calM, l1N, problemShape_);
        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum);

        int64_t tileNum = bs.GetTileNum();
        // Send event when using aiv_1
        // Process tiles in ping-pong mode
        bool enableCVSync = false;
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockOffset = GetOffset(blockCoord, problemShape_, aGlobal_, bGlobal_, yGlobal_, transA, transB);
            // calculate block-level offset
            if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
                break;
            }
            int64_t offsetA = Get<0>(blockOffset);
            int64_t offsetB = Get<1>(blockOffset);
            int64_t offsetC = Get<2>(blockOffset);
            // AIC Process
            if ASCEND_IS_AIC {
                // Synchronize with aiv
                if (enableCVSync) {
                    CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
                    if (enable2UB) {
                        CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
                    }
                }
                auto aGlobalT = aGlobal_[offsetA];
                auto bGlobalT = bGlobal_[offsetB];
                auto cLocalT = epilogueOp.GetTensor(blockShape);
                blockMmadOp.IterateAll(cLocalT, aGlobalT, bGlobalT, blockShape);
                // Compute block-level mmad with epilogue
                enableCVSync = true;
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                if (enable2UB) {
                    CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
                }
            }
            // AIV Process
            if ASCEND_IS_AIV {
                // Synchronize with aic
                CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_V>(AIC_SYNC_AIV_FLAG);
                // Calulate epilogue
                epilogueOp(blockShape, offsetC, enable2UB);
                // Notify aic
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            }
        }
        // Match extra event after aic process finished
        if ASCEND_IS_AIC {
            if (enableCVSync) {
                CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
                if (enable2UB) {
                    CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
                }
            }
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
        // Check args for block scheduler
        CHECK_AND_RETURN(BlockSchedulerOp::CheckArgs(args.problemShape));
        // Check args fro block epilogue
        CHECK_AND_RETURN(BlockEpilogue::CheckArgs(args.epilogueArgs));
        return Status::success;
    }

    __host_aicore__ static size_t GetWorkSpaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkSpaceSize();
        // Calculate extra workspace size for epilogue
        workSpaceSize += BlockEpilogue::GetWorkSpaceSize(blockNum, l1M, l1N);
        // Calculate extra workspace size for block scheduler
        workSpaceSize += BlockSchedulerOp::GetWorkSpaceSize(shape);
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
        Run(params);
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif
