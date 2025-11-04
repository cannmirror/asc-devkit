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
 * \file block_sparse_matmul_builder.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_SPARSE_MATMUL_BUILDER_H
#define MATMUL_BLOCK_BLOCK_SPARSE_MATMUL_BUILDER_H

#define ASCENDC_CUBE_ONLY
#include "../../utils/host_utils.h"
#include "kernel_operator.h"
#include "../matmul_intf.h"

#include "../../utils/common_utils.h"
#include "../../utils/layout_utils.h"
#include "../../utils/status_utils.h"
#include "../../utils/tuple_utils.h"

#include "../block/block_mmad.h"
#include "../policy/dispatch_policy.h"

namespace Act {
namespace Gemm {
namespace Block {
/**
 * @class BlockSparseMatmulBuilder 
 * @brief A template class for building block sparse matrix multiplication
 * @param [in] AType_: the type of matrix A
 * @param [in] LayoutA_: the layout of matrix A
 * @param [in] BType_: the type of matrix B
 * @param [in] LayoutB_: the layout of matrix B
 * @param [in] CType_: the type of matrix C
 * @param [in] LayoutC_: the layout of matrix C
 * @param [in] BiasType_: the type of bias
 * @param [in] LayoutBias_: the layout of bias
 * @param [in] L1TileShape_: the shape of L1 tiles
 * @param [in] L0TileShape_: the shape of L0 tiles
 * @param [in] BlockScheduler_: the block scheduler
 * @param [in] BlockMatmulPolicy_: the block matrix multiplication policy
 * @param [in] TileCopyParam_: the parameters for tile copy
 * @param [in] Enable_: the enable condition
 */
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_,
          class BlockMatmulPolicy_ = MatmulMultiBlockBias<>, class TileCopyParam_ = void, typename Enable_ = void>
class BlockSparseMatmulBuilder {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>,
                  "BlockSparseMatmulBuilder is not implemented for this BlockMatmulPolicy");
};

/**
 * @class BlockSparseMatmulBuilder 
 * @brief A specialized template class for block sparse matrix multiplication with specific policies
 * @param [in] AType_: the type of matrix A
 * @param [in] LayoutA_: the layout of matrix A
 * @param [in] BType_: the type of matrix B
 * @param [in] LayoutB_: the layout of matrix B
 * @param [in] CType_: the type of matrix C
 * @param [in] LayoutC_: the layout of matrix C
 * @param [in] BiasType_: the type of bias
 * @param [in] LayoutBias_: the layout of bias
 * @param [in] L1TileShape_: the shape of L1 tiles
 * @param [in] L0TileShape_: the shape of L0 tiles
 * @param [in] BlockScheduler_: the block scheduler
 * @param [in] BlockMatmulPolicy_: the block matrix multiplication policy
 * @param [in] TileCopyParam_: the parameters for tile copy
 */
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_,
          class TileCopyParam_>
class BlockSparseMatmulBuilder<
    AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_, L1TileShape_, L0TileShape_,
    BlockScheduler_, BlockMatmulPolicy_, TileCopyParam_,
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<SparseMatmulMultiBlockOnKAxisWithLayout<>,
        BlockMatmulPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using IndexType = uint8_t;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BlockMatmulPolicy = BlockMatmulPolicy_;
    using TileCopyParam = TileCopyParam_;
    // transA and transB are deduced from LayoutA and LayoutB
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatA = TagToFormat<LayoutA>::format;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr CubeFormat formatC = TagToFormat<LayoutC>::format;

    using AMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatA, AType, transA>;
    using BMatmulType = AscendC::SparseMatmulType<AscendC::TPosition::GM, AscendC::TPosition::GM, formatB, BType, transB>;
    using CMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatC, CType>;
    using BiasMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatC, CType>;

    using BlockMmadOp = Block::BlockMmad<BlockMatmulPolicy, L1TileShape, L0TileShape, AMatmulType, BMatmulType,
                                         CMatmulType, BiasMatmulType, TileCopyParam>;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape>();

    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape>();

    /**
     * @struct Arguments
     * @brief Kernel arguments for the host side
     */
    struct Arguments {
        GM_ADDR aGmAddr{nullptr};       ///< The global memory address of matrix A
        GM_ADDR bGmAddr{nullptr};       ///< The global memory address of matrix B
        GM_ADDR cGmAddr{nullptr};       ///< The global memory address of matrix C
        GM_ADDR biasGmAddr{nullptr};    ///< The global memory address of bias
        GM_ADDR indexGmAddr{nullptr};   ///< The global memory address of index
    };

    // params
    using Params = Arguments;

    __aicore__ inline BlockSparseMatmulBuilder() {}

    __aicore__ inline ~BlockSparseMatmulBuilder() {}

    /**
     * @brief Get the size of the workspace
     * @return The size of the workspace
     */
    __host_aicore__ static size_t GetWorkspaceSize()
    {
        return 0;
    }

    /**
     * @brief Check if the current configuration can be implemented
     * @param [in] args: the kernel arguments
     * @return The status indicating whether the configuration is valid
     */
    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        if (l0M * l0K * sizeof(AType) * DOUBLE_BUFFER_COUNT > L0A_SIZE ||
            l0N * l0K * sizeof(BType) * DOUBLE_BUFFER_COUNT > L0B_SIZE || l0M * l0N * sizeof(CType) > L0C_SIZE ||
            (l1M * l1K * sizeof(AType) + l1K * l1N * sizeof(BType)) * DOUBLE_BUFFER_COUNT > L1_SIZE) {
            return Status::tileShapeErrorExceedsLimit;
        }
        return Status::success;
    }

    /**
     * @brief Initialize the parmeters
     * @param [in] args: the kernel arguments
     * @return The initialized parmeters
     */
    __host_aicore__ static Params InitParams(Arguments args)
    {
        Params params = {args.aGmAddr, args.bGmAddr, args.cGmAddr, args.biasGmAddr, args.indexGmAddr};
        return params;
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Act

#endif