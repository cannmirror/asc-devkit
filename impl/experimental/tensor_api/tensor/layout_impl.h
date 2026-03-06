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

class DimConversion {
public:
    template<typename T>
    __aicore__ inline auto Run(const T& tensor) {
        return ConvertTwoDim2FourDim(tensor);
    }
    
private:
    template<typename T>
    __aicore__ inline auto ConvertTwoDim2FourDim(const T& tensor) {
        auto layout = tensor.Layout();
        auto shape4Dim = MakeShape(
            MakeShape(Std::Int<1>{}, GetShape<0>(layout)),
            MakeShape(Std::Int<1>{}, GetShape<1>(layout))
        );
        auto stride4Dim = MakeStride(
            MakeStride(Std::Int<0>{}, GetStride<0>(layout)),
            MakeStride(Std::Int<0>{}, GetStride<1>(layout))
        );
        auto fourDimLayout = MakeLayout(shape4Dim, stride4Dim);
        return MakeTensorImpl(tensor.Engine().Begin(), fourDimLayout);
    }
};

template <typename T>
__aicore__ inline auto PreProcess(const T& tensor) {
    using  tensorShape = Std::remove_cvref_t<decltype(tensor.Shape())>;
    if constexpr (nesting_depth_v<tensorShape> == TWO_DIM_DATA) {
        return DimConversion{}.Run(tensor);
    } else {
        static_assert(nesting_depth_v<tensorShape> == FOUR_DIM_DATA, "Only support two or four dim LayoutType");
        return tensor;
    }
}

template <typename Layout, typename TileShape>
__aicore__ inline decltype(auto) MakeTileLayout(const Layout& layout, const TileShape& tileShape) {
    static_assert(Std::is_tuple_v<TileShape>);

    using OriginShape = Std::remove_cvref_t<decltype(layout.Shape())>;
    if constexpr (nesting_depth_v<TileShape> == nesting_depth_v<OriginShape>
                  && Std::tuple_size_v<TileShape> == Std::tuple_size_v<OriginShape>) {
        return MakeLayout(tileShape, layout.Stride());
    } else {
        static_assert(Std::tuple_size_v<TileShape> == TWO_DIM_DATA);

        const uint32_t rows = Std::get<0>(tileShape);
        const uint32_t cols = Std::get<1>(tileShape);

        const auto& innerRow = Std::get<0>(Std::get<0>(layout.Shape()));
        const auto& innerCol = Std::get<0>(Std::get<1>(layout.Shape()));

        using InnerRowType = Std::remove_cvref_t<decltype(innerRow)>;
        using InnerColType = Std::remove_cvref_t<decltype(innerCol)>;

        if constexpr (IsIntegralConstantV<InnerRowType> && IsIntegralConstantV<InnerColType>) {
            return MakeLayout(
                MakeShape(MakeShape(Std::Int<InnerRowType::value>{}, CeilDivision(rows, InnerRowType::value)),
                          MakeShape(Std::Int<InnerColType::value>{}, CeilDivision(cols, InnerColType::value))),
                layout.Stride());
        } else {
            return MakeLayout(
                MakeShape(MakeShape(innerRow, CeilDivision(rows, innerRow)),
                                        MakeShape(innerCol, CeilDivision(cols, innerCol))),
                layout.Stride());
        }
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H
