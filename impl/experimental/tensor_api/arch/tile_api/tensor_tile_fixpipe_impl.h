/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file tensor_tile_fixpipe_impl.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TENSOR_TILE_FIXPIPE_IMPL_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TENSOR_TILE_FIXPIPE_IMPL_H

namespace AscendC {
struct FixpipeTrait {
    __aicore__ constexpr FixpipeTrait() {}

    __aicore__ constexpr FixpipeTrait(const QuantMode_t quantPreIn) : quantPre(quantPreIn) {}

    __aicore__ constexpr FixpipeTrait(
        const QuantMode_t quantPreIn,
        const bool enableReluIn,
        const bool enableChannleSplitIn,
        const uint8_t unitFlagIn,
        const uint8_t dualDstCtlIn
    ) :
        quantPre(quantPreIn),
        enableRelu(enableReluIn),
        enableChannleSplit(enableChannleSplitIn),
        unitFlag(unitFlagIn),
        dualDstCtl(dualDstCtlIn)
    {}

    QuantMode_t quantPre = QuantMode_t::NoQuant;
    bool enableRelu = false;
    bool enableChannleSplit = false;
    uint8_t unitFlag = false;
    uint8_t dualDstCtl = false;
};
constexpr FixpipeTrait DEFAULT_FIXPIPE_TRAIT;
}

#include "impl/experimental/tensor_api/arch/tile_api/fixpipe/tensor_tile_fixpipe_routing.h"

namespace AscendC {

template <const FixpipeTrait& trait, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplate<T, U>, void>::type
Fixpipe(const T& dst, const U& src)
{
    constexpr Hardware dstTPos = TileInternal::GetHardPos<T>();
    constexpr Hardware srcTPos = TileInternal::GetHardPos<U>();
    using Tensor2Tensor = typename TileInternal::FixpipeTensor2Tensor<dstTPos, srcTPos,
        TileInternal::CURRENT_ARCH_VERSION,TileInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<T, U, trait>(dst, src);
}

template <const FixpipeTrait& trait, typename T, typename U, typename V>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplate<T, U, V>, void>::type
Fixpipe(const T& dst, const U& src, const V& quant)
{
    constexpr Hardware dstTPos = TileInternal::GetHardPos<T>();
    constexpr Hardware srcTPos = TileInternal::GetHardPos<U>();
    using Tensor2Tensor = typename TileInternal::FixpipeQuantTensor2Tensor<dstTPos, srcTPos,
        TileInternal::CURRENT_ARCH_VERSION, TileInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<T, U, V, trait>(dst, src, quant);
}

template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplateWithCoord<T, U, Coord>, void>::type
Fixpipe(const T& dst, const U& src, const Coord& coord)
{
    constexpr Hardware dstTPos = TileInternal::GetHardPos<T>();
    constexpr Hardware srcTPos = TileInternal::GetHardPos<U>();
    using Tensor2Tensor = typename TileInternal::FixpipeTensor2Tensor<dstTPos, srcTPos,
        TileInternal::CURRENT_ARCH_VERSION, TileInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<T, U, trait, Coord>(dst, src, coord);
}

template <const FixpipeTrait& trait, typename T, typename U, typename V,  typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplateWithCoord<T, U, V, Coord>, void>::type
Fixpipe(const T& dst, const U& src, const V& quant, const Coord& coord)
{
    constexpr Hardware dstTPos = TileInternal::GetHardPos<T>();
    constexpr Hardware srcTPos = TileInternal::GetHardPos<U>();
    using Tensor2Tensor = typename TileInternal::FixpipeQuantTensor2Tensor<dstTPos, srcTPos,
        TileInternal::CURRENT_ARCH_VERSION, TileInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<T, U, V, trait, Coord>(dst, src, quant, coord);
}
} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TENSOR_TILE_FIXPIPE_IMPL_H