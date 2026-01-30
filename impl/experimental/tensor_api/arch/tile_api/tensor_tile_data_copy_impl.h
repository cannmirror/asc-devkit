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
 * \file tensor_tile_data_copy_impl.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_TENSOR_TILE_DATA_COPY_IMPL_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_TENSOR_TILE_DATA_COPY_IMPL_H

namespace AscendC {
struct DataCopyTrait {
};
constexpr DataCopyTrait DEFAULT_DATA_COPY_TRAIT;
}

#include "impl/experimental/tensor_api/arch/tile_api/data_copy/tensor_tile_data_copy_routing.h"

namespace AscendC {
template <const DataCopyTrait& trait = DEFAULT_DATA_COPY_TRAIT, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingDataCopyTemplate<T, U>, void>::type
DataCopy(const T& dst, const U& src)
{
    constexpr Hardware dstTPos = TileInternal::GetHardPos<T>();
    constexpr Hardware srcTPos = TileInternal::GetHardPos<U>();
    using Tensor2Tensor = typename
        TileInternal::DataCopyTensor2Tensor<dstTPos, srcTPos, TileInternal::CURRENT_ARCH_VERSION, TileInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<T, U, trait>(dst, src);
}

template <const DataCopyTrait& trait = DEFAULT_DATA_COPY_TRAIT, typename T, typename U, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingDataCopyTemplateWithCoord<T, U, Coord>, void>::type
DataCopy(const T& dst, const U& src, const Coord& coord)
{
    DataCopy<trait, T, U>(dst, src);
}

} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_TENSOR_TILE_DATA_COPY_IMPL_H