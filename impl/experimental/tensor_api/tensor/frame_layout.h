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
* \file frame_layout.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_FRAME_LAYOUT_H
#define IMPL_TENSOR_API_TENSOR_FRAME_LAYOUT_H

#include "impl/experimental/tensor_api/tensor/layout_method.h"

namespace AscendC {
namespace Te {

struct MakeNzFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        constexpr auto c0Ele = TraitType::C0_element;
        auto shape = MakeShape(MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED)),
                               MakeShape(c0Ele, Std::ceil_division(column, c0Ele)));
        auto stride = MakeStride(MakeStride(c0Ele, c0Ele * Std::Int<FRACTAL_FIXED>{}),
                                 MakeStride(Std::Int<1>{}, c0Ele * Std::ceil_align(row, FRACTAL_FIXED)));
        using LayoutT = Layout<decltype(shape), decltype(stride), NzLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeL0CFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        constexpr auto c0Ele = LayoutTrait<uint16_t>::C0_element;
        auto shape = MakeShape(MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED)),
                               MakeShape(c0Ele, Std::ceil_division(column, c0Ele)));
        auto stride = MakeStride(MakeStride(c0Ele, c0Ele * Std::Int<FRACTAL_FIXED>{}),
                                 MakeStride(Std::Int<1>{}, c0Ele * Std::ceil_align(row, FRACTAL_FIXED)));
        using LayoutT = Layout<decltype(shape), decltype(stride), L0CLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeNDFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, column), MakeStride(Std::Int<0>{}, Std::Int<1>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), NDLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeZnFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        constexpr auto c0Ele = TraitType::C0_element;
        auto shape = MakeShape(MakeShape(c0Ele, Std::ceil_division(row, c0Ele)),
                               MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(column, FRACTAL_FIXED)));
        auto stride = MakeStride(MakeStride(Std::Int<1>{}, c0Ele * Std::ceil_align(column, FRACTAL_FIXED)),
                                 MakeStride(c0Ele, c0Ele * Std::Int<FRACTAL_FIXED>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), ZnLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeDNFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, Std::Int<1>{}), MakeStride(Std::Int<0>{}, row));
        using LayoutT = Layout<decltype(shape), decltype(stride), DNLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeZzFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        constexpr auto c0Ele = TraitType::C0_element;
        auto shape = MakeShape(MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED)),
                               MakeShape(c0Ele, Std::ceil_division(column, c0Ele)));
        auto stride = MakeStride(MakeStride(c0Ele, FRACTAL_FIXED * Std::ceil_align(column, c0Ele)),
                                 MakeStride(Std::Int<1>{}, c0Ele * Std::Int<FRACTAL_FIXED>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), ZzLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeNnFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t>,
            "NnLayoutPattern only supports fp8_e8m0_t.");
        auto shape = MakeShape(MakeShape(Std::Int<MX_SCALE_K0>{}, row / MX_SCALE_K0),
                               MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(column, FRACTAL_FIXED)));
        auto stride = MakeStride(MakeStride(Std::Int<1>{}, Std::Int<C0_SIZE<fp8_e8m0_t>>{}),
                                 MakeStride(Std::Int<MX_SCALE_K0>{}, row * FRACTAL_FIXED));
        using LayoutT = Layout<decltype(shape), decltype(stride), NnLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleANDFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t>,
            "ScaleANDLayoutPattern only supports fp8_e8m0_t.");
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, column), MakeStride(Std::Int<0>{}, Std::Int<1>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), ScaleANDLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleADNFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t>,
            "ScaleADNLayoutPattern only supports fp8_e8m0_t.");
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<MX_SCALE_K0>{}, column / MX_SCALE_K0));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, Std::Int<MX_SCALE_K0>{}),
                                 MakeStride(Std::Int<1>{}, MX_SCALE_K0 * row));
        using LayoutT = Layout<decltype(shape), decltype(stride), ScaleADNLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleBNDFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t>,
            "ScaleBNDLayoutPattern only supports fp8_e8m0_t.");
        auto shape = MakeShape(MakeShape(Std::Int<MX_SCALE_K0>{}, row / MX_SCALE_K0), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<1>{}, MX_SCALE_K0 * column),
                                 MakeStride(Std::Int<0>{}, Std::Int<MX_SCALE_K0>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), ScaleBNDLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleBDNFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t>,
            "ScaleBDNLayoutPattern only supports fp8_e8m0_t.");
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, Std::Int<1>{}), MakeStride(Std::Int<0>{}, row));
        using LayoutT = Layout<decltype(shape), decltype(stride), ScaleBDNLayoutPattern>;
        return LayoutT(shape, stride);
    }
};

using LayoutFormatSet = TupleMap<
    Std::tuple<NzLayoutPattern, MakeNzFrameLayout>,
    Std::tuple<L0CLayoutPattern, MakeL0CFrameLayout>,
    Std::tuple<NDLayoutPattern, MakeNDFrameLayout>,
    Std::tuple<DNLayoutPattern, MakeDNFrameLayout>,
    Std::tuple<NnLayoutPattern, MakeNnFrameLayout>,
    Std::tuple<ZzLayoutPattern, MakeZzFrameLayout>,
    Std::tuple<ZnLayoutPattern, MakeZnFrameLayout>,
    Std::tuple<ScaleANDLayoutPattern, MakeScaleANDFrameLayout>,
    Std::tuple<ScaleADNLayoutPattern, MakeScaleADNFrameLayout>,
    Std::tuple<ScaleBNDLayoutPattern, MakeScaleBNDFrameLayout>,
    Std::tuple<ScaleBDNLayoutPattern, MakeScaleBDNFrameLayout>>;

template <typename LayoutPattern, typename TraitType, typename... Args>
__aicore__ inline decltype(auto) MakeFrameLayout(const Args&... args) {
    using GetLayoutMakeFun = typename LayoutFormatSet::template Get<LayoutPattern>;
    static_assert(!Std::is_same_v<GetLayoutMakeFun, EmptyValue>, "Unsupported layout pattern.");
    return GetLayoutMakeFun::template Make<TraitType>(args...);
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_FRAME_LAYOUT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
