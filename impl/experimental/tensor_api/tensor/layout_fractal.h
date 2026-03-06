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
* \file layout_fractal.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H

#include "impl/experimental/tensor_api/utils/utils_impl.h"
#include "impl/experimental/tensor_api/tensor/layout_dispatch.h"

namespace AscendC {
namespace Te {

// NZ
template <typename T, size_t row, size_t column>
using NZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<row / FRACTAL_FIXED>>,
    Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<column / (C0_SIZE / sizeof(T))>>>;

template <typename T, size_t row, size_t column>
using NZStrideFormat = Stride<Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>,
    Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * row>>>;

// ND
template <typename T, size_t row, size_t column>
using NDShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using NDStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<column>>, Stride<Std::Int<0>, Std::Int<1>>>;

// DN
template <typename T, size_t row, size_t column>
using DNShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using DNStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<1>>, Stride<Std::Int<0>, Std::Int<row>>>;

// ZN
template <typename T, size_t  row, size_t  column>
using ZNShapeFormat = Shape<Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<row / (C0_SIZE / sizeof(T))>>,
    Shape<Std::Int<FRACTAL_FIXED>, Std::Int<column / FRACTAL_FIXED>>>;
template <typename T, size_t  row, size_t  column>
using ZNStrideFormat = Stride<Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * column>>,
    Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>>;

// ZZ
template <typename T, size_t row, size_t column>
using ZZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<row / FRACTAL_FIXED>>,
    Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<column / (C0_SIZE / sizeof(T))>>>;
template <typename T, size_t row, size_t column>
using ZZStrideFormat = Stride<Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<FRACTAL_FIXED * column>>,
    Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>>;

template <typename T>
__aicore__ inline decltype(auto) MakeNzLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::NZ, T>::apply(row, column);
}

template <>
__aicore__ inline decltype(auto) MakeNzLayout<Std::ignore_t>(size_t row, size_t column) {
    return MakeNzLayout<uint16_t>(row, column);
}

__aicore__ inline decltype(auto) MakeL0CLayout(size_t row, size_t column) {
    return MakeNzLayout<uint16_t>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeNDLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::ND, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeDNLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::DN, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeZnLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::ZN, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeZzLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::ZZ, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeNnLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::NN, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeScaleANDLayout(size_t row, size_t column) { // 不转置(m, scaleK)
    return LayoutDispatcher<LayoutFormat::ND, Std::ignore_t>::apply(row, column); // (m, scaleK)
}

template <typename T>
__aicore__ inline decltype(auto) MakeScaleADNLayout(size_t row, size_t column) { // 转置(m, scaleK)
    return LayoutDispatcher<LayoutFormat::DN, T>::apply(row, column); // 转置(m, scaleK)
}

template <typename T>
__aicore__ inline decltype(auto) MakeScaleBNDLayout(size_t row, size_t column) { // 不转置(scaleK, n)
    return LayoutDispatcher<LayoutFormat::ND, T>::apply(row, column); // (scaleK, n)
}

template <typename T>
__aicore__ inline decltype(auto) MakeScaleBDNLayout(size_t row, size_t column) { // 转置(scaleK, n)
    return LayoutDispatcher<LayoutFormat::DN, Std::ignore_t>::apply(row, column); // (scaleK, n)
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_FRACTAL_H