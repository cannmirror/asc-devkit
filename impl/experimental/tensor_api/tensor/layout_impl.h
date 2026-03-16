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
 * \file layout_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_IMPL_H

#include "impl/experimental/tensor_api/utils/utils_impl.h"
#include "impl/experimental/tensor_api/tensor/layout_method.h"
#include "impl/experimental/tensor_api/tensor/coord_index.h"
#include "impl/experimental/tensor_api/tensor/layout_fractal.h"

namespace AscendC {
namespace Te {

template <typename T>
struct LocalTensor;

template <typename Coord, typename LayoutType, typename ShapeType>
__aicore__ inline decltype(auto) MakeTileLayout(const Coord& coord, const LayoutType& layout, const ShapeType& tileShape) 
{
    static_assert(nesting_depth_v<ShapeType> == LayoutType::depth, "tile shape depth is not equal layout.shape");
    static_assert(Std::tuple_size_v<ShapeType> == LayoutType::rank, "tile shape rank is not equal layout.shape");
    return MakeLayout(tileShape, layout.Stride());
}

template <typename Coord, typename LayoutType, typename TensorType>
__aicore__ inline decltype(auto) MakeTileLayout(const Coord& coord, const LayoutType& layout, const LocalTensor<TensorType>& tileTensor) 
{
    static_assert(tileTensor.rank == LayoutType::rank, "tensor rank is not equal layout");

    auto innerRow = Std::get<0>(Std::get<0>(layout.Shape()));
    auto innerCol = Std::get<0>(Std::get<1>(layout.Shape()));

    auto srcRow = innerRow * Std::get<1>(Std::get<0>(layout.Shape())) - Std::get<0>(coord);
    auto srcCol = innerCol * Std::get<1>(Std::get<1>(layout.Shape())) - Std::get<1>(coord);

    auto dstRow = Std::get<0>(Std::get<0>(tileTensor.Shape())) * Std::get<1>(Std::get<0>(tileTensor.Shape()));
    auto dstCol = Std::get<0>(Std::get<1>(tileTensor.Shape())) * Std::get<1>(Std::get<1>(tileTensor.Shape()));

    auto realRow = Min(srcRow, dstRow);
    auto realCol = Min(srcCol, dstCol);

    auto row = MakeShape(innerRow, CeilDivision(realRow, innerRow));
    auto col = MakeShape(innerCol, CeilDivision(realCol, innerCol));

    return MakeTileLayout(coord, layout, MakeShape(row, col));
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H
