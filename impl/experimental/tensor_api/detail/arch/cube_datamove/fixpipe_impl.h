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
 * \file fixpipe_impl.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_IMPL_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_IMPL_H

namespace AscendC {
struct FixpipeTrait {
    __aicore__ constexpr FixpipeTrait() {}
    __aicore__ constexpr FixpipeTrait(
        QuantMode_t quantPreIn,
        bool enableReluIn,
        bool enableChannleSplitIn,
        uint8_t unitFlagIn,
        uint8_t dualDstCtlIn
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

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_routing.h"

namespace AscendC {

template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplate<T, U>, void>::type
Fixpipe(const T& dst, const U& src)
{
    constexpr Hardware dstPos = TensorInternal::GetHardPos<T>();
    constexpr Hardware srcPos = TensorInternal::GetHardPos<U>();
    constexpr Hardware quantPos = Hardware::MAX;
    auto coordZero = MakeCoord(Std::Int<0>{}, Std::Int<0>{});
    using Tensor2Tensor = typename TensorInternal::FixpipeTensor2Tensor<dstPos, srcPos, quantPos,
        TensorInternal::CURRENT_ARCH_VERSION, TensorInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<trait>(dst, src, coordZero);
}

template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename S>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplate<T, U, S>, void>::type
Fixpipe(const T& dst, const U& src, const S& quant)
{
    constexpr Hardware dstPos = TensorInternal::GetHardPos<T>();
    constexpr Hardware srcPos = TensorInternal::GetHardPos<U>();
    constexpr Hardware quantPos = Hardware::L1;
    auto coordZero = MakeCoord(Std::Int<0>{}, Std::Int<0>{});
    using Tensor2Tensor = typename TensorInternal::FixpipeTensor2Tensor<dstPos, srcPos, quantPos,
        TensorInternal::CURRENT_ARCH_VERSION, TensorInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<trait>(dst, src, quant, coordZero);
}

template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplateWithCoord<T, U, Coord>, void>::type
Fixpipe(const T& dst, const U& src, const Coord& coord)
{
    constexpr Hardware dstPos = TensorInternal::GetHardPos<T>();
    constexpr Hardware srcPos = TensorInternal::GetHardPos<U>();
    constexpr Hardware quantPos = Hardware::MAX;
    using Tensor2Tensor = typename TensorInternal::FixpipeTensor2Tensor<dstPos, srcPos, quantPos,
        TensorInternal::CURRENT_ARCH_VERSION, TensorInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<trait>(dst, src, coord);
}

template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename S,  typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplateWithCoord<T, U, S, Coord>, void>::type
Fixpipe(const T& dst, const U& src, const S& quant, const Coord& coord)
{
    constexpr Hardware dstPos = TensorInternal::GetHardPos<T>();
    constexpr Hardware srcPos = TensorInternal::GetHardPos<U>();
    constexpr Hardware quantPos = Hardware::L1;
    using Tensor2Tensor = typename TensorInternal::FixpipeTensor2Tensor<dstPos, srcPos, quantPos,
        TensorInternal::CURRENT_ARCH_VERSION, TensorInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<trait>(dst, src, quant, coord);
}
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_IMPL_H