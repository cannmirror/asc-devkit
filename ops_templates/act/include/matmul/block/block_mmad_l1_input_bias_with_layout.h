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
 * \file block_mmad_l1_input_bias_with_layout.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_L1_INPUT_BIAS_WITH_LAYOUT_H
#define MATMUL_BLOCK_BLOCK_MMAD_L1_INPUT_BIAS_WITH_LAYOUT_H

#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"
#include "lib/matmul/constant_tiling.h"

#include "../policy/dispatch_policy.h"
#include "./matmul_impl_traits.h"
#include "./block_mmad_with_layout.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class L1TileShape, class L0TileShape, class AT, class BT, class CT, class BiasT, class TileCopy>
class BlockMmad<MatmulL1InputBiasWithLayout<>, L1TileShape, L0TileShape, AT, BT, CT, BiasT, TileCopy,
    AscendC::Std::enable_if_t<IsMatmulLayoutTypeV<AT>>>
    : public BlockMmad<MatmulL1InputBiasWithLayout<>, L1TileShape, L0TileShape,
        ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy> {
    using Base = BlockMmad<MatmulL1InputBiasWithLayout<>, L1TileShape, L0TileShape,
                           ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy>;
    using Base::Base;
};

/**
* @class BlockMmad
* @brief Template class BlockMmad for performing matrix multiplication operations with specific layouts
*
* This class is specialized base on MatmulL1InputBiasWithLayout<> and TileCopy<Ascend910_95, CopyNoGmIn>
*/
template <class L1Shape, class L0Shape, class AType, class BType, class CType, class BiasType>
class BlockMmad<MatmulL1InputBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType,
    Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>, AscendC::Std::enable_if_t<!IsMatmulLayoutTypeV<AType>>>
    : public BlockMmadWithLayout<
        BlockMmad<MatmulL1InputBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType,
                  Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>>,
        MatmulL1InputBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType,
        Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>
    > {
public:
    using DispatchPolicy = MatmulL1InputBiasWithLayout<>;
    using TileCopy = Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>;
    using Self = BlockMmad<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    using Base = BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    friend class BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    static_assert(AscendC::PhyPosIsL1(AType::pos) && AscendC::PhyPosIsL1(BType::pos), "Only support L1 input");
    static_assert(IsF16OrBf16AB<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>(), "Unsupported dtype");
    static_assert(IsNz<AType>() && IsNz<BType>() && IsNDOrAlign<CType>(), "L1Load only support Nz input and ND output");

public:
    /**
    * @brief Sets the bias
    * @param BiasTrait: bias trait
    * @param biasL1: local tensor of bias
    */
    template <class BiasTrait>
    __aicore__ inline void SetBias(const AscendC::LocalTensor<BiasTrait>& biasL1)
    {
        AscendC::LocalTensor<typename BiasType::T> biasLocal;
        biasLocal.SetAddr(biasL1.address_);

        matmul_.SetBias(biasLocal);
    }

private:
    MatmulImplTraitsT<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy> matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
