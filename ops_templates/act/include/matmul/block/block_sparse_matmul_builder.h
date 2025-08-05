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
 * \file block_sparse_matmul_builder.h
 * \brief
 */

#ifndef ACT_BLOCK_SPARSE_MATMUL_BUILDER_H
#define ACT_BLOCK_SPARSE_MATMUL_BUILDER_H

#define ASCENDC_CUBE_ONLY
#include "include/utils/host_utils.h"
#include "kernel_operator.h"
#include "include/matmul/matmul_intf.h"

#include "include/utils/common_utils.h"
#include "include/utils/layout_utils.h"
#include "include/utils/status_utils.h"
#include "include/utils/tuple_utils.h"

#include "include/matmul/block/block_mmad.h"
#include "include/matmul/policy/dispatch_policy.h"

using namespace AscendC;

namespace Act {
namespace Gemm {
namespace Block {
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_,
          class BlockMatmulPolicy_ = MatmulMultiBlockBias<>, class TileCopyParam_ = void, typename Enable_ = void>
class BlockSparseMatmulBuilder {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>,
                  "BlockSparseMatmulBuilder is not implemented for this BlockMatmulPolicy");
};

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_,
          class TileCopyParam_>
class BlockSparseMatmulBuilder<
    AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_, L1TileShape_, L0TileShape_,
    BlockScheduler_, BlockMatmulPolicy_, TileCopyParam_,
    std::enable_if_t<std::is_base_of_v<SparseMatmulMultiBlockOnKAxisWithLayout<>, BlockMatmulPolicy_>>> {
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
    static constexpr CubeFormat FormatA = TagToFormat<LayoutA>::format;
    static constexpr CubeFormat FormatB = TagToFormat<LayoutB>::format;
    static constexpr CubeFormat FormatC = TagToFormat<LayoutC>::format;

    using AMatmulType = MatmulType<AscendC::TPosition::GM, FormatA, AType, transA>;
    using BMatmulType = SparseMatmulType<AscendC::TPosition::GM, AscendC::TPosition::GM, FormatB, BType, transB>;
    using CMatmulType = MatmulType<AscendC::TPosition::GM, FormatC, CType>;
    using BiasMatmulType = MatmulType<AscendC::TPosition::GM, FormatC, CType>;

    using BlockMmadOp = Block::BlockMmad<BlockMatmulPolicy, L1TileShape, L0TileShape, AMatmulType, BMatmulType,
                                         CMatmulType, BiasMatmulType, TileCopyParam>;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape>();

    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape>();

    // host side kernel arguments
    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR indexGmAddr{nullptr};
    };

    // params
    using Params = Arguments;

    __aicore__ inline BlockSparseMatmulBuilder() {}

    __aicore__ inline ~BlockSparseMatmulBuilder() {}

    __host_aicore__ static size_t GetWorkSpaceSize()
    {
        return 0;
    }

    __host_aicore__ static Status CheckArgs(Arguments const& args)
    {
        if (l0M * l0K * sizeof(AType) * DOUBLE_BUFFER_COUNT > L0A_SIZE ||
            l0N * l0K * sizeof(BType) * DOUBLE_BUFFER_COUNT > L0B_SIZE || l0M * l0N * sizeof(CType) > L0C_SIZE ||
            (l1M * l1K * sizeof(AType) + l1K * l1N * sizeof(BType)) * DOUBLE_BUFFER_COUNT > L1_SIZE) {
            return Status::tileShapeErrorExceedsLimit;
        }
        return Status::success;
    }

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