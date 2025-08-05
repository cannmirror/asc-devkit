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
 * \file kernel_sparse_matmul.h
 * \brief
 */

#ifndef ACT_KERNEL_SPARSE_MATMUL_H
#define ACT_KERNEL_SPARSE_MATMUL_H

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
#include "include/matmul/block/block_sparse_matmul_builder.h"
#include "include/epilogue/block_epilogue_empty.h"
#include "include/matmul/block/block_scheduler_utils.h"
#include "include/matmul/block/block_scheduler_iterateK.h"

namespace Act {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelSparseMatmul {
    static_assert(AscendC::Std::always_false_v<BlockEpilogue_>,
                  "KernelSparseMatmul is not implemented for this BlockEpilogue");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelSparseMatmul<ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
                         std::enable_if_t<std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    __aicore__ inline KernelSparseMatmul() {}
    __aicore__ inline ~KernelSparseMatmul() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool TRANS_A = BlockMmadBuilder::transA;
    static constexpr bool TRANS_B = BlockMmadBuilder::transB;
    static constexpr int64_t L1_M = BlockMmadBuilder::l1M;
    static constexpr int64_t L1_N = BlockMmadBuilder::l1N;
    static constexpr int64_t L1_K = BlockMmadBuilder::l1K;
    static constexpr int64_t DENSE_MATRIX_B_OFFSET = 2;
    static constexpr int64_t INDEX_MATRIX_OFFSET = 8;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
                                               typename BlockMmadBuilder::L0TileShape, BlockScheduler, TRANS_A,
                                               TRANS_B>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using IndexType = typename BlockMmadBuilder::IndexType;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;
    using ATensorTrait = TensorTrait<AType, AscendC::TPosition::GM, NDLayout>;
    using BTensorTrait = TensorTrait<BType, AscendC::TPosition::GM, NDLayout>;
    using CTensorTrait = TensorTrait<CType, AscendC::TPosition::GM, NDLayout>;
    using AGlobalTensorType = AscendC::GlobalTensor<ATensorTrait>;
    using BGlobalTensorType = AscendC::GlobalTensor<BTensorTrait>;
    using CGlobalTensorType = AscendC::GlobalTensor<CTensorTrait>;
    // NZ layout for index
    using NZLayout = AscendC::Layout<
        AscendC::Shape<AscendC::Shape<_16, int64_t>, AscendC::Shape<_8, int64_t>>, // 8 = 32 / (sizeof(uint8) * 4)
        AscendC::Stride<AscendC::Stride<_8, _128>, AscendC::Stride<_1, int64_t>>   // 128 = 16 * 8
        >;
    using IndexTensorTrait = TensorTrait<IndexType, AscendC::TPosition::GM, NZLayout>;
    using IndexGlobalTensorType = AscendC::GlobalTensor<IndexTensorTrait>;

    // attribute
    AGlobalTensorType aGlobal_;
    BGlobalTensorType bGlobal_;
    CGlobalTensorType cGlobal_;
    IndexGlobalTensorType indexGlobal_;

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
        InitGlobalTensorA<NDLayout, AGlobalTensorType, ATensorTrait, AType>(aGlobal_, blockMmadParams_.aGmAddr, TRANS_A,
                                                                            m, k);
        InitGlobalTensorB<NDLayout, BGlobalTensorType, BTensorTrait, BType>(bGlobal_, blockMmadParams_.bGmAddr, TRANS_B,
                                                                            n, k / DENSE_MATRIX_B_OFFSET);
        InitGlobalTensorC<NDLayout, CGlobalTensorType, CTensorTrait, CType>(cGlobal_, blockMmadParams_.cGmAddr, m, n);
        NZLayout indexLayout = AscendC::MakeLayout(
            AscendC::MakeShape(AscendC::MakeShape(_16{}, AscendC::Ceil<int64_t>(n, 16)),    // 16: fractal size
                               AscendC::MakeShape(_8{}, AscendC::Ceil<int64_t>(k / 8, 8))), // 8: 32/(4 * sizeof(uint8))
            AscendC::MakeStride(AscendC::MakeStride(_8{}, _128{}),
                                AscendC::MakeStride(_1{}, AscendC::CeilAlign<int64_t>(n, 16) * 8)));
        indexGlobal_.SetTensorTrait(IndexTensorTrait(indexLayout));
        indexGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ IndexType*>(blockMmadParams_.indexGmAddr),
                                     n * k / INDEX_MATRIX_OFFSET);
    }

    template <class BlockCoord>
    __aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t, int64_t> GetOffset(const BlockCoord& blockCoord)
    {
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        AscendC::Coord<int64_t, int64_t> aCoord;
        if constexpr (!TRANS_A) {
            aCoord = AscendC::MakeCoord(Get<0>(blockCoord), Get<2>(blockCoord));
        } else {
            aCoord = AscendC::MakeCoord(Get<2>(blockCoord), Get<0>(blockCoord));
        }
        AscendC::Coord<int64_t, int64_t> bCoord = AscendC::MakeCoord(Get<1>(blockCoord), Get<2>(blockCoord));
        AscendC::Coord<int64_t, int64_t> cCoord = AscendC::MakeCoord(Get<0>(blockCoord), Get<1>(blockCoord));

        int64_t offsetA = aGlobal_.GetTensorTrait().GetLayout()(aCoord) + Get<3>(blockCoord) * m * k;
        int64_t offsetB =
            bGlobal_.GetTensorTrait().GetLayout()(bCoord) + Get<3>(blockCoord) * n * k / DENSE_MATRIX_B_OFFSET;
        int64_t offsetC = cGlobal_.GetTensorTrait().GetLayout()(cCoord) + Get<3>(blockCoord) * m * n;
        int64_t offsetIndex =
            indexGlobal_.GetTensorTrait().GetLayout()(bCoord) + Get<3>(blockCoord) * n * k / INDEX_MATRIX_OFFSET;

        return {offsetA, offsetB, offsetC, offsetIndex};
    }

    __aicore__ inline void run(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        // Get blockIdx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        if (curBlockIdx >= blockNum) {
            return;
        }
        // Init
        Init(params);
        blockMmadOp.Init();
        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum);

        int64_t tileNum = bs.GetTileNum();
        // Send event when using aiv_1

        // Process tiles in ping-pong mode
        int64_t loopIdx = 0;
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            auto blockShape = bs.GetBlockShape(tileIdx);
            auto blockCoord = bs.GetBlockCoord(tileIdx);
            auto blockOffset = GetOffset(blockCoord);

            // calculate block-level offset
            if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
                return;
            }
            int64_t offsetA = Get<0>(blockOffset);
            int64_t offsetB = Get<1>(blockOffset);
            int64_t offsetC = Get<2>(blockOffset);
            int64_t offsetIndex = Get<3>(blockOffset);

            auto aGlobalT = aGlobal_[offsetA];
            auto bGlobalT = bGlobal_[offsetB];
            auto cGlobalT = cGlobal_[offsetC];
            auto indexGlobalT = indexGlobal_[offsetIndex];
            blockMmadOp.IterateAll(cGlobalT, aGlobalT, bGlobalT, indexGlobalT, blockShape);
        }
    }

    __host_aicore__ static Status CheckShape(ProblemShape const& shape)
    {
        int64_t m = shape.m;
        int64_t n = shape.n;
        int64_t k = shape.k;
        int64_t b = shape.b;
        if (b > 1) { // Sparse only support batch 1
            return Status::batchErrorExcceedsLimit;
        }
        if (k % 8 != 0) { // 8: Sparse k must be multiple of 8
            return Status::nkErrorMatrixExceedsLimit;
        }
        // Check m, n, k overlimit data type
        if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
            return Status::mnkErrorExceedsLimit;
        }
        // Check matrix size exceeds limit
        if (!TRANS_A && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // mk matrix k limit
            return Status::mkErrorMatrixExceedsLimit;
        }

        if (TRANS_A && m > MATRIX_INNER_DIM_LIMIT_SIZE) { // km matrix m limit
            return Status::kmErrorMatrixExceedsLimit;
        }
        if (!TRANS_B && n > MATRIX_INNER_DIM_LIMIT_SIZE) { // kn matrix n limit
            return Status::knErrorMatrixExceedsLimit;
        }

        if (TRANS_B && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // nk matrix k limit
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
        return Status::success;
    }

    __host_aicore__ static size_t GetWorkSpaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkSpaceSize();
        // Calculate extra workspace size for block scheduler
        workSpaceSize += BlockSchedulerOp::GetWorkSpaceSize(shape);
        return workSpaceSize;
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        // mmad params with epiligue takes workspaceGm as output
        Params params = {args.problemShape, mmadParams, {}};
        return params;
    }

    static int64_t GetBlockNum(ProblemShape shape)
    {
        return BlockSchedulerOp::GetBlockNum(shape);
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
