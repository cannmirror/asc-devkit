/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_tensor_tile_data_copy_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_IMPL_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_IMPL_H

#include "datacopy/kernel_tensor_tile_data_copy_routing.h"

namespace AscendC {

template <typename T>
__aicore__ inline decltype(auto) MakeNZLayout(size_t row, size_t column) {
    return TileInternal::MakeLayoutForNZ<T>(row, column);
}

template <>
__aicore__ inline decltype(auto) MakeNZLayout<Std::ignore_t>(size_t row, size_t column) {
    return TileInternal::MakeLayoutForNZ<uint16_t>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeRowMajorLayout(size_t row, size_t column) {
    return TileInternal::MakeLayoutForND<T>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeColumnMajorLayout(size_t row, size_t column) {
    return TileInternal::MakeLayoutForDN<T>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeZNLayout(size_t row, size_t column) {
    return TileInternal::MakeLayoutForZN<T>(row, column);
}

template <const DataCopyTrait& trait, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingDataCopyTemplate<T, U>, void>::type
DataCopy(const T& dst, const U& src)
{
    constexpr TPosition dstTPos = TileInternal::GetTensorTraitType<T>::tPos;
    constexpr TPosition srcTPos = TileInternal::GetTensorTraitType<U>::tPos;
    using Tensor2Tensor = typename
        TileInternal::DataCopyTensor2Tensor<dstTPos, srcTPos, TileInternal::CURRENT_ARCH_VERSION, TileInternal::FOUR_DIM_DATA>::type;
    Tensor2Tensor{}.template Run<T, U, trait>(dst, src);
}

template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
__aicore__ inline typename Std::enable_if<VerifyingDataCopyTemplateWithCoord<T, U, Coord>, void>::type
DataCopy(const T& dst, const U& src, const Coord& coord)
{
    auto index = Crd2Idx(coord, src.GetTensorTrait().GetLayout());
    if constexpr (TileInternal::IsIntegralConstantV<decltype(index)>) {
        DataCopy<T, U, trait>(dst, src[decltype(index)()]);
    } else {
        DataCopy<T, U, trait>(dst, src[index]);
    }
}

} // namespace AscendC

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_IMPL_H