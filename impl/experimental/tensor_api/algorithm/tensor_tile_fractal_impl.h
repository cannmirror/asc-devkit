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
 * \file tensor_tile_fractal_impl.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_TILE_FRACTAL_IMPL_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_TILE_FRACTAL_IMPL_H

#include "impl/experimental/tensor_api/utils/tensor_tile_utils.h"
#include "impl/experimental/tensor_api/struct/tensor_tile_struct.h"

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

template <typename T>
__aicore__ inline decltype(auto) MakeZZLayout(size_t row, size_t column) {
    return TileInternal::MakeLayoutForZZ<T>(row, column);
}

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

template <typename T, size_t row, size_t column>
using ZNLayoutFormat = Layout<TileInternal::ZNShapeFormat<T, row, column>, TileInternal::ZNStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using ZZLayoutFormat = Layout<TileInternal::ZZShapeFormat<T, row, column>, TileInternal::ZZStrideFormat<T, row, column>>;

} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_TILE_FRACTAL_IMPL_H
