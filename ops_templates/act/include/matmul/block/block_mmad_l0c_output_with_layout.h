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
 * \file block_mmad_l0c_output_with_layout.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_MMAD_L0C_OUTPUT_WITH_LAYOUT_H
#define MATMUL_BLOCK_BLOCK_MMAD_L0C_OUTPUT_WITH_LAYOUT_H

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
class BlockMmad<MatmulL0COutputWithLayout<>, L1TileShape, L0TileShape, AT, BT, CT, BiasT, TileCopy,
    AscendC::Std::enable_if_t<IsMatmulLayoutTypeV<AT>>>
    : public BlockMmad<MatmulL0COutputWithLayout<>, L1TileShape, L0TileShape,
        ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy> {
    using Base = BlockMmad<MatmulL0COutputWithLayout<>, L1TileShape, L0TileShape,
                           ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy>;
    using Base::Base;
};

/**
* @class BlockMmad
* @brief Block matrix class for matrix multiplication operations
*
* This class is specialized base on MamtulL0COutputWithLayout<>
*/
template <class L1Shape, class L0Shape, class AType, class BType, class CType, class BiasType, class TileCopy>
class BlockMmad<MatmulL0COutputWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy,
    AscendC::Std::enable_if_t<!IsMatmulLayoutTypeV<AType>>>
    : public BlockMmadWithLayout<
        BlockMmad<MatmulL0COutputWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>,
        MatmulL0COutputWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy
    > {
public:
    using DispatchPolicy = MatmulL0COutputWithLayout<>;
    using Self = BlockMmad<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    using Base = BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    friend class BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    using TileCopy_ = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<TileCopy, void>,
        Tile::TileCopy<Arch::Ascend910_95, Tile::CopyWithLayout>,
        TileCopy
    >;

    static_assert(
        AscendC::PhyPosIsGM(AType::pos) && AscendC::PhyPosIsGM(BType::pos) && AscendC::PhyPosIsL0C(CType::pos),
        "Only support GM input L0C output"
    );
    static_assert(IsF16F16F32<AType, BType, CType>() || IsBf16Bf16F32<AType, BType, CType>(), "Unsupported dtype");
    static_assert(IsNDOrAlign<AType>(), "Input A only support ND");

public:
    /**
    * @brief GetTensorC function to get matrix C tensor
    * @param [out] DstTensor: destination tensor type
    * @param [in] SrcTensor: source tensor type
    * @param [in] Coord: coordinate type
    * @param [out] ub: destination tensor
    * @param [in] l0c: source tensor
    * @param [in] coord: coordinate
    * @param [in] subIdx: sub-index
    */
    template <class DstTensor, class SrcTensor, class Coord>
    __aicore__ inline void GetTensorC(DstTensor& ub, SrcTensor& l0c, const Coord& coord, uint8_t subIdx)
    {
        using DstTrait = typename AscendC::tensor_trait<DstTensor>::trait_type;
        using SrcTrait = typename AscendC::tensor_trait<SrcTensor>::trait_type;
        using DstType = AscendC::MatmulType<
            DstTrait::tPos,
            LayoutToFormatV<typename DstTrait::LiteType, typename DstTrait::LiteLayoutType>,
            typename DstTrait::LiteType
        >;
        static_assert(AscendC::is_local_tensor_v<DstTensor> && AscendC::is_local_tensor_v<SrcTensor>,
                      "Only support local tensor.");
        static_assert(PosIsL0C<SrcTrait::tPos>() && PosIsUB<DstTrait::tPos>(),
                      "Only support l0c to ub.");
        static_assert(IsNDOrAlign<DstType>() || IsNz<DstType>(), "Dst format only support ND or Nz.");

        typename TileCopy_::template CopyCo1ToOut<DstType, DstTrait, SrcTrait> copyCo1ToOut;
        copyCo1ToOut(ub, l0c, coord, subIdx);
    }

private:
    MatmulImplTraitsT<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType> matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
