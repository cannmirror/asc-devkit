/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file block_conv_builder.h
 * \brief
 */
#ifndef CONV_BLOCK_BLOCK_CONV_BUILDER_H
#define CONV_BLOCK_BLOCK_CONV_BUILDER_H

#include "block_conv_forward.h"

using namespace AscendC;
namespace Act {
namespace Conv {
namespace Block {
struct DirectConvPolicy {
    constexpr static bool enableInputDataLenCheck = false;
};
struct Img2ColConvMModePolicy {
    constexpr static bool enableInputDataLenCheck = false;
};
struct Img2ColConvHWModePolicy {
    constexpr static bool enableInputDataLenCheck = false;
};

template <class ProblemShape_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,
          class BiasType_, class LayoutBias_, class L1TileShape_, class L0TileShape_, class TileAttr_,
          class BlockConvPolicy_, class TileCopyParam_ = void, typename Enable_ = void>
class BlockConvBuilder {
    static_assert(AscendC::Std::always_false_v<BlockConvPolicy_>,
                  "BlockConvBuilder is not implemented for this BlockConvPolicy");
};

template <class ProblemShape_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,
          class BiasType_, class LayoutBias_, class L1TileShape_, class L0TileShape_, class TileAttr_,
          class BlockConvPolicy_, class TileCopyParam_>
class BlockConvBuilder<ProblemShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_,
                       L1TileShape_, L0TileShape_, TileAttr_, BlockConvPolicy_, TileCopyParam_,
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<Img2ColConvMModePolicy, BlockConvPolicy_> ||
                              AscendC::Std::is_base_of_v<Img2ColConvHWModePolicy, BlockConvPolicy_>>> {
public:
    using ProblemShape = ProblemShape_;
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using TileAttr = TileAttr_;
    using BlockConvPolicy = BlockConvPolicy_;
    using TileCopyParam = TileCopyParam_;

    using AGlobalTensorType = AscendC::GlobalTensor<AType>;
    using BGlobalTensorType = AscendC::GlobalTensor<BType>;
    using CGlobalTensorType = AscendC::GlobalTensor<CType>;
    using BiasGlobalTensorType = AscendC::GlobalTensor<BiasType>;

    using BlockConvOp = Block::BlockConv<ProblemShape, BlockConvPolicy, L1TileShape, L0TileShape,
                AType, BType, CType, BiasType,
                AGlobalTensorType, BGlobalTensorType, CGlobalTensorType, BiasGlobalTensorType, TileAttr, TileCopyParam>;

    // host side kernel arguments
    struct BlockConvArguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
    };

    __aicore__ inline BlockConvBuilder() {}

    __aicore__ inline ~BlockConvBuilder() {}
};
} // namespace Block
} // namespace Conv
} // namespace Act
#endif