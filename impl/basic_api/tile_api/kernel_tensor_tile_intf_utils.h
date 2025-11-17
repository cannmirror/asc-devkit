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
 * \file kernel_tensor_tile_intf_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_INTF_UTILS_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_INTF_UTILS_H

#include "kernel_tensor_tile_utils.h"

namespace AscendC {

template <typename T, size_t row, size_t column, typename Enable = void>
struct NZLayoutFormat;

template <typename T, size_t row, size_t column>
struct NZLayoutFormat<T, row, column, typename Std::enable_if<!Std::is_same_v<T, Std::ignore_t>>::type> {
    using type = Layout<TileInternal::NZShapeFormat<T, row, column>, TileInternal::NZStrideFormat<T, row, column>>;
};

template <typename T, size_t row, size_t column>
struct NZLayoutFormat<T, row, column, typename Std::enable_if<Std::is_same_v<T, Std::ignore_t>>::type> {
    using type = Layout<TileInternal::NZShapeFormat<uint16_t, row, column>, TileInternal::NZStrideFormat<uint16_t, row, column>>;
};

template <typename T, size_t row, size_t column>
using NDLayoutFormat = Layout<TileInternal::NDShapeFormat<T, row, column>, TileInternal::NDStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using DNLayoutFormat = Layout<TileInternal::DNShapeFormat<T, row, column>, TileInternal::DNStrideFormat<T, row, column>>;

template <typename T, typename U>
constexpr bool VerifyingDataCopyTemplate =
    ((TileInternal::IsGlobalTensorTraitV<U> && TileInternal::IsLocalTensorTraitV<T>) ||
    (TileInternal::IsLocalTensorTraitV<U> && TileInternal::IsLocalTensorTraitV<T>));

template <typename T, typename U, typename Coord>
constexpr bool VerifyingDataCopyTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingDataCopyTemplate<T, U>;

template <typename T, typename U>
constexpr bool VerifyingLoadDataTemplate = TileInternal::IsLocalTensorTraitV<U> && TileInternal::IsLocalTensorTraitV<T>;

template <typename T, typename U, typename Coord>
constexpr bool VerifyingLoadDataTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingLoadDataTemplate<T, U>;

template <typename T, typename U>
static constexpr bool VerifyingFixpipeTemplate = (TileInternal::IsGlobalTensorTraitV<T> && TileInternal::IsLocalTensorTraitV<U>);

template <typename T, typename U, typename V>
static constexpr bool VerifyingFixpipeQuantTemplate = (TileInternal::IsGlobalTensorTraitV<T> && TileInternal::IsLocalTensorTraitV<U>
    && (TileInternal::IsLocalTensorTraitV<V> || Std::is_same_v<V, uint64_t>)); 

template <typename T, typename U, typename Coord>
constexpr bool VerifyingFixpipeTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingFixpipeTemplate<T, U>;

} // namespace AscendC

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_INTF_UTILS_H