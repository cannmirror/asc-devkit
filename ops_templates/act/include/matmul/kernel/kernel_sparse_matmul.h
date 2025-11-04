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
 * \file kernel_sparse_matmul.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_SPARSE_MATMUL_H
#define MATMUL_KERNEL_KERNEL_SPARSE_MATMUL_H

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
#include "../block/sparse_block_mmad_multi_block_on_kaxis_with_layout.h"
#include "../../epilogue/block_epilogue_empty.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_iterateK.h"

namespace Act {
namespace Gemm {
namespace Kernel {
/**
 * @class KernelSparseMatmul
 * @brief A class template declaration for sparse matrix multiplication
 * 
 * This class is a template class that implements sparse matrix multiplication based on different template parameters
 * 
 * @param [in] ProblemShape_: the shape of the matrix multiplication problem
 * @param [in] BlockMmad_: block MMAD (Matrix Multiply Add) operations
 * @param [in] BlockEpilogue_: the epilogue operation for matrix multiplication
 * @param [in] BlockScheduler_: the block scheduler for managing block processing in matrix multiplication
 * @param [in] Enable_: the enable condition for template specialization
 */
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelSparseMatmul {
    static_assert(AscendC::Std::always_false_v<BlockEpilogue_>,
                  "KernelSparseMatmul is not implemented for this BlockEpilogue");
};

/**
 * @class KernelSparseMatmul
 * @brief Specialized implementation for BlockEpilogueEmpty
 * 
 * This specialization is used when BlockEpilogue_ is Block::BlockEpilogueEmpty
 * 
 * @param [in] ProblemShape_: the shape of the matrix multiplication problem
 * @param [in] BlockMmad_: block MMAD (Matrix Multiply Add) operations
 * @param [in] BlockEpilogue_: the epilogue operation for matrix multiplication
 * @param [in] BlockScheduler_: the block scheduler for managing block processing in matrix multiplication
 */
template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class KernelSparseMatmul<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    __aicore__ inline KernelSparseMatmul() {}
    __aicore__ inline ~KernelSparseMatmul() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool TRANS_A = BlockMmad::AType::isTrans;
    static constexpr bool TRANS_B = BlockMmad::BType::isTrans;
    static constexpr int64_t L1_M = GetIntegralConstant<MNK_M, typename BlockMmad::L1Shape>();
    static constexpr int64_t L1_N = GetIntegralConstant<MNK_N, typename BlockMmad::L1Shape>();
    static constexpr int64_t L1_K = GetIntegralConstant<MNK_K, typename BlockMmad::L1Shape>();
    static constexpr int64_t L0_M = GetIntegralConstant<MNK_M, typename BlockMmad::L0Shape>();
    static constexpr int64_t L0_N = GetIntegralConstant<MNK_N, typename BlockMmad::L0Shape>();
    static constexpr int64_t L0_K = GetIntegralConstant<MNK_K, typename BlockMmad::L0Shape>();
    static constexpr int64_t DENSE_MATRIX_B_OFFSET = 2;
    static constexpr int64_t INDEX_MATRIX_OFFSET = 8;

    /**
     * @struct BlockMmadArguments
     * @brief Kernel arguments for the host side
     */
    struct BlockMmadArguments {
        GM_ADDR aGmAddr{nullptr};       ///< The global memory address of matrix A
        GM_ADDR bGmAddr{nullptr};       ///< The global memory address of matrix B
        GM_ADDR cGmAddr{nullptr};       ///< The global memory address of matrix C
        GM_ADDR biasGmAddr{nullptr};    ///< The global memory address of bias
        GM_ADDR indexGmAddr{nullptr};   ///< The global memory address of index
    };

    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmad::L1Shape,
                                               typename BlockMmad::L0Shape, BlockScheduler, TRANS_A,
                                               TRANS_B>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = BlockMmad;
    using BlockMmadArguments = BlockMmadArguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = BlockMmadArguments;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmad::AType::T;
    using BType = typename BlockMmad::BType::T;
    using CType = typename BlockMmad::CType::T;
    using IndexType = uint8_t;
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
    // NZ layout for index
    using NZLayout = AscendC::Layout<
        AscendC::Shape<AscendC::Shape<_16, int64_t>, AscendC::Shape<_8, int64_t>>, // 8 = 32 / (sizeof(uint8) * 4)
        AscendC::Stride<AscendC::Stride<_8, _128>, AscendC::Stride<_1, int64_t>>   // 128 = 16 * 8
        >;
    using IndexTensorTrait = AscendC::TensorTrait<IndexType, AscendC::TPosition::GM, NZLayout>;
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

    /**
     * @struct Arguments
     * @brief Structure to hold arguments for the problem
     */
    struct Arguments {
        ProblemShape problemShape;              ///< Problem shape
        BlockMmadArguments mmadArgs;            ///< MMAD parameters
        BlockEpilogueArguments epilogueArgs;    ///< Epilogue parameters
        Arguments() = default;                  ///< Default constructor
    };

    /**
     * @struct Params
     * @brief Structure to hold parameters for the problem
     */
    struct Params {
        ProblemShape problemShape;              ///< Problem shape
        BlockMmadParams mmadParams;             ///< MMAD parameters
        BlockEpilogueParams epilogueParams;     ///< Epilogue parameters
        Params() = default;                     ///< Default constructor
    };

    /**
     * @brief Convert ProblemShape to TupleShape
     * @param [in] shape: ProblemShape to be converted
     * @return TupleShape representation of the input ProblemShape
     */
    __aicore__ inline static TupleShape ToShapeTuple(ProblemShape const& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    /**
     * @brief Initialize the parameters for the problem
     * @param [in] params: parameters to be initialized
     */
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

    /**
     * @brief Get the offset for the block
     * @param [in] BlockCoord: type of the block coordinate
     * @param [in] blockCoord: block coordinate
     * @return Tuple of offsets for A, B, C, and index
     */
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

    /**
     * @brief Check the shape of the problem
     * @param [in] shape: problem shape to be checked
     * @return Status of the check
     */
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

    /**
     * @brief Check if the problem can be implemented
     * @param [in] args: arguments for the problem
     * @return Status of the check
     */
    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        Status BlockMmadCanImplement;
        if (L0_M * L0_K * sizeof(AType) * DOUBLE_BUFFER_COUNT > L0A_SIZE ||
            L0_N * L0_K * sizeof(BType) * DOUBLE_BUFFER_COUNT > L0B_SIZE || L0_M * L0_N * sizeof(CType) > L0C_SIZE ||
            (L1_M * L1_K * sizeof(AType) + L1_K * L1_N * sizeof(BType)) * DOUBLE_BUFFER_COUNT > L1_SIZE) {
            BlockMmadCanImplement = Status::tileShapeErrorExceedsLimit;
        } else {
            BlockMmadCanImplement = Status::success;
		}
        CHECK_AND_RETURN(BlockMmadCanImplement);
        // Check args for block scheduler
        CHECK_AND_RETURN(BlockSchedulerOp::CanImplement(args.problemShape));
        return Status::success;
    }

    /**
     * @brief Get the workspace size for the problem
     * @param [in] shape: problem shape
     * @param [in] blockNum: number of blocks
     * @return Workspace size
     */
    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += 0; // workspace size for mmad is zero
        // Calculate extra workspace size for block scheduler
        workSpaceSize += BlockSchedulerOp::GetWorkspaceSize(shape);
        return workSpaceSize;
    }

    /**
     * @brief Initialize the parameters for the problem
     * @param [in] args: arguments for the problem
     * @param [out] workspace: the address of the work space
     * @return Initialized parameters
     */
    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = {args.mmadArgs.aGmAddr, args.mmadArgs.bGmAddr,
                                      args.mmadArgs.cGmAddr, args.mmadArgs.biasGmAddr, args.mmadArgs.indexGmAddr};
        // mmad params with epiligue takes workspaceGm as output
        Params params = {args.problemShape, mmadParams, {}};
        return params;
    }

    /**
     * @brief Get the number of blocks for the problem
     * @param [in] shape: problem shape
     * @return Number of blocks
     */
    static int64_t GetBlockNum(ProblemShape shape)
    {
        return BlockSchedulerOp::GetBlockNum(shape);
    }

    /**
     * @brief Overloaded operator() for the problem
     * @param [in] params: parameters for the problem
     */
    __aicore__ inline void operator()(Params const& params)
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
};

} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif
