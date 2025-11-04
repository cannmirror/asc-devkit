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
 * \file block_mmad_multi_block_bias_with_layout.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_BIAS_WITH_LAYOUT_H
#define MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_BIAS_WITH_LAYOUT_H

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
class BlockMmad<MatmulMultiBlockBiasWithLayout<>, L1TileShape, L0TileShape, AT, BT, CT, BiasT, TileCopy,
    AscendC::Std::enable_if_t<IsMatmulLayoutTypeV<AT>>>
    : public BlockMmad<MatmulMultiBlockBiasWithLayout<>, L1TileShape, L0TileShape,
        ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy> {
    using Base = BlockMmad<MatmulMultiBlockBiasWithLayout<>, L1TileShape, L0TileShape,
                           ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy>;
    using Base::Base;
};

/**
* @class BlockMmad
* @brief A template class BlockMmad for performing multi-block matrix multiplication operations
*
* The class is specialized base on MatmulMultiBlockBiasWithLayout<>
*/
template <class L1Shape, class L0Shape, class AType, class BType, class CType, class BiasType, class TileCopy>
class BlockMmad<MatmulMultiBlockBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy,
    AscendC::Std::enable_if_t<
        !AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitMWithParams>> &&
        !AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitNWithParams>> &&
        !IsMatmulLayoutTypeV<AType>
    >> : public BlockMmadWithLayout<
        BlockMmad<MatmulMultiBlockBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>,
        MatmulMultiBlockBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy
    > {
public:
    using DispatchPolicy = MatmulMultiBlockBiasWithLayout<>;
    using Self = BlockMmad<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    using Base = BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    friend class BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    static_assert(IsNDOrAlign<AType>() && IsNDOrAlign<CType>(), "Only support ND format");
    static_assert(AscendC::PhyPosIsGM(AType::pos) && AscendC::PhyPosIsGM(BType::pos), "Only support GM input");
    static_assert(
        IsF16OrBf16AB<AType, BType, CType>() || IsI8I8I32<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>(),
        "Unsupported dtype"
    );

public:
    /**
    * @brief Set the bias term
    * @param [in] biasGm: global bias tensor
    */
    template <class BiasTrait>
    __aicore__ inline void SetBias(const AscendC::GlobalTensor<BiasTrait>& biasGm)
    {
        AscendC::GlobalTensor<typename BiasType::T> biasGlobal;
        biasGlobal.address_ = biasGm.address_;

        matmul_.SetBias(biasGlobal);
    }

private:
    MatmulImplTraitsT<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType> matmul_;
};

/**
* @class BlockMmad
* @brief A template class BlockMmad for performing multi-block matrix multiplication operations
*
* The class is specialized base on MatmulMultiBlockBiasWithLayout<>, specifically for splitM/spltN scenarios
*/
template <class L1Shape, class L0Shape, class AType, class BType, class CType, class BiasType, class TileCopy>
class BlockMmad<MatmulMultiBlockBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy,
    AscendC::Std::enable_if_t<
        (AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitMWithParams>> ||
        AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitNWithParams>>) &&
        !IsMatmulLayoutTypeV<AType>
    >> : public BlockMmadWithLayout<
        BlockMmad<MatmulMultiBlockBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>,
        MatmulMultiBlockBiasWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy
    > {
public:
    using DispatchPolicy = MatmulMultiBlockBiasWithLayout<>;
    using Self = BlockMmad<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    using Base = BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    friend class BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    static_assert(
        AscendC::PhyPosIsGM(AType::pos) && AscendC::PhyPosIsGM(BType::pos) && AscendC::PhyPosIsUB(CType::pos),
        "Only support GM input UB output"
    );
    static_assert(
        IsF16F16F32<AType, BType, CType>() || IsBf16Bf16F32<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>(),
        "Unsupported dtype"
    );
    static_assert(IsNDOrAlign<AType>() && IsNDOrAlign<CType>(), "Only support ND format");

public:
    /**
    * @brief Set tensor bias for matrix multiplication
    * @param [in] bias: global tensor for matrix bias
    */
    template <class BiasTrait>
    __aicore__ inline void SetBias(const AscendC::GlobalTensor<BiasTrait>& bias)
    {
        AscendC::GlobalTensor<typename BiasType::T> biasGlobal;
        biasGlobal.address_ = bias.address_;

        matmul_.SetBias(biasGlobal);
    }

private:
    MatmulImplTraitsT<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy> matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
