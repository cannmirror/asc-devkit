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
 * \file block_mmad_l1_input_bias.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_L1_INPUT_BIAS_H
#define MATMUL_BLOCK_BLOCK_MMAD_L1_INPUT_BIAS_H

#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"
#include "lib/matmul/constant_tiling.h"

#include "../policy/dispatch_policy.h"
#include "./matmul_impl_traits.h"
#include "./block_mmad_with_params.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class L1TileShape, class L0TileShape, class AT, class BT, class CT, class BiasT, class TileCopy>
class BlockMmad<MatmulL1InputBias<>, L1TileShape, L0TileShape, AT, BT, CT, BiasT, TileCopy,
    AscendC::Std::enable_if_t<IsMatmulLayoutTypeV<AT>>>
    : public BlockMmad<MatmulL1InputBias<>, L1TileShape, L0TileShape,
        ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy> {
    using Base = BlockMmad<MatmulL1InputBias<>, L1TileShape, L0TileShape,
                           ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy>;
    using Base::Base;
};

/**
* @class BlockMmad
* @brief Block matrix multiplication class template with L1 input bias
* This class is specialized base on MatmulL1InputBias<> and TileCopy<Ascend910_95, CopyNoGmIn>
*/
template <class L1Shape, class L0Shape, class AType, class BType, class CType, class BiasType>
class BlockMmad<MatmulL1InputBias<>, L1Shape, L0Shape, AType, BType, CType, BiasType,
    Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>, AscendC::Std::enable_if_t<!IsMatmulLayoutTypeV<AType>>>
    : public BlockMmadWithParams<
        BlockMmad<MatmulL1InputBias<>, L1Shape, L0Shape, AType, BType, CType, BiasType,
                  Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>>,
        MatmulL1InputBias<>, L1Shape, L0Shape, AType, BType, CType, BiasType,
        Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>
    > {
public:
    using DispatchPolicy = MatmulL1InputBias<>;
    using TileCopy = Tile::TileCopy<Arch::Ascend910_95, Tile::CopyNoGmIn>;
    using Self = BlockMmad<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    using Base = BlockMmadWithParams<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    friend class BlockMmadWithParams<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    static_assert(
        IsF16OrBf16AB<AType, BType, CType>() || IsI8I8I32<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>(),
        "Unsupported dtype"
    );
    static_assert(IsNz<AType>() && IsNz<BType>() && IsNDOrAlign<CType>(), "L1Load only support Nz input and ND output");

public:
    /**
    * @brief Set tensor bias for matrix multiplication
    * @param [in] biasLocal: local tensor for matrix bias
    */
    __aicore__ inline void SetBias(const AscendC::LocalTensor<typename BiasType::T>& biasLocal)
    {
        matmul_.SetBias(biasLocal);
    }
    /**
    * @brief Set tensor A for matrix multiplication
    * @param [in] aLocal: local tensor for matrix A
    * @param [in] isTransposeA: whether to transpose matrix A, default is false
    */
    __aicore__ inline void SetTensorA(const AscendC::LocalTensor<typename AType::T>& aLocal, bool isTransposeA = false)
    {
        matmul_.SetTensorA(aLocal, isTransposeA);
    }
    /**
    * @brief Set tensor B for matrix multiplication
    * @param [in] bLocal: local tensor for matrix B
    * @param [in] isTransposeB: whether to transpose matrix B, default is false
    */
    __aicore__ inline void SetTensorB(const AscendC::LocalTensor<typename BType::T>& bLocal, bool isTransposeB = false)
    {
        matmul_.SetTensorB(bLocal, isTransposeB);
    }
    /**
    * @brief Iterate over all elements and store the result in global memory
    * @param [in] gm: global memory tensor
    * @param [in] enAtomic: whether to enable atomic operations
    */
    __aicore__ inline void IterateAll(const AscendC::GlobalTensor<typename CType::T>& gm, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(gm, enAtomic);
    }
    /**
    * @brief Iterate over all elements and store the result in local memory
    * @param [in] ubCmatrix: local memory tensor
    * @param [in] enAtomic: whether to enable atomic operations
    */
    __aicore__ inline void IterateAll(const AscendC::LocalTensor<typename CType::T>& ubCmatrix, uint8_t enAtomic = 0)
    {
        matmul_.IterateAll(ubCmatrix, enAtomic);
    }

private:
    MatmulImplTraitsT<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy> matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
