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
 * \file tensor_tile_load_data_impl.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_TENSOR_TILE_LOAD_DATA_IMPL_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_TENSOR_TILE_LOAD_DATA_IMPL_H

namespace AscendC {
struct LoadDataTrait {
    __aicore__ constexpr LoadDataTrait() {}

    __aicore__ constexpr LoadDataTrait(const bool transposedIn) : transposed(transposedIn) {}

    bool transposed = false;
};
constexpr LoadDataTrait DEFAULT_LOAD_DATA_TRAIT{};
}

#include "impl/experimental/tensor_api/arch/tile_api/load_data/tensor_tile_load_data_routing.h"

namespace AscendC {
template<const LoadDataTrait& trait = DEFAULT_LOAD_DATA_TRAIT, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingLoadDataTemplate<T, U>, void>::type 
LoadData(const T& dst, const U& src)
{
    auto coord = AscendC::MakeCoord(
        AscendC::MakeCoord(Std::Int<0>{}, Std::Int<0>{}),
        AscendC::MakeCoord(Std::Int<0>{}, Std::Int<0>{})
    );
    LoadData<trait, T, U, decltype(coord)>(dst, src, coord);
}

template<const LoadDataTrait& trait = DEFAULT_LOAD_DATA_TRAIT, typename T, typename U, class Coord>
__aicore__ inline typename Std::enable_if<VerifyingLoadDataTemplateWithCoord<T, U, Coord>, void>::type 
LoadData(const T& dst, const U& src, const Coord& coord)
{
    constexpr Hardware dstPos = TileInternal::GetHardPos<T>();
    constexpr Hardware srcPos = TileInternal::GetHardPos<U>();
    using Tensor2Tensor = typename TileInternal::LoadDataTensor2Tensor<dstPos, srcPos, 
        TileInternal::CURRENT_ARCH_VERSION, TileInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<T, U, trait, Coord>(dst, src, coord);
}
} // namespace AscendC
#endif // IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_TENSOR_TILE_LOAD_DATA_IMPL_H