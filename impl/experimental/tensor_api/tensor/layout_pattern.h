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
* \file layout_pattern.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_PATTERN_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_PATTERN_H

#include "impl/experimental/tensor_api/tensor/layout_method.h"

namespace AscendC {
namespace Te {

struct ZnLayoutPattern {};
struct FP4ZnLayoutPattern {};
struct ZzLayoutPattern {};
struct NnLayoutPattern {};
struct NzLayoutPattern {};
struct L0CLayoutPattern {};
struct NDLayoutPattern {};
struct DNLayoutPattern {};
struct RowMajorLayoutPattern {};
struct ColumnMajorLayoutPattern {};
struct ScaleZzLayoutPattern {};
struct ScaleANDLayoutPattern {};
struct ScaleADNLayoutPattern {};
struct ScaleBNDLayoutPattern {};
struct ScaleBDNLayoutPattern {};
struct DefaultPattern {};

struct MakeNzFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        auto shape = MakeShape(MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED)),
                               MakeShape(c0Ele, Std::ceil_division(column, c0Ele)));
        auto stride = MakeStride(MakeStride(c0Ele, c0Ele * Std::Int<FRACTAL_FIXED>{}),
                                 MakeStride(Std::Int<1>{}, c0Ele * Std::ceil_align(row, FRACTAL_FIXED)));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<NzLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeL0CFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        static_assert(c0Ele == 16, "L0CLayoutPattern only supports ShapeColumn0 as 16.");
        auto shape = MakeShape(MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED)),
                               MakeShape(c0Ele, Std::ceil_division(column, c0Ele)));
        auto stride = MakeStride(MakeStride(c0Ele, c0Ele * Std::Int<FRACTAL_FIXED>{}),
                                 MakeStride(Std::Int<1>{}, c0Ele * Std::ceil_align(row, FRACTAL_FIXED)));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<L0CLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeNDFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, column), MakeStride(Std::Int<0>{}, Std::Int<1>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<NDLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeRowMajorFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, column), MakeStride(Std::Int<0>{}, Std::Int<1>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<RowMajorLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeZnFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        auto shape = MakeShape(MakeShape(c0Ele, Std::ceil_division(row, c0Ele)),
                               MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(column, FRACTAL_FIXED)));
        auto stride = MakeStride(MakeStride(Std::Int<1>{}, c0Ele * Std::ceil_align(column, FRACTAL_FIXED)),
                                 MakeStride(c0Ele, c0Ele * Std::Int<FRACTAL_FIXED>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<ZnLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeZnFP4FrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        static_assert(is_b4_type<DataType> && c0Ele == 64,
            "ZnLayoutPatternFP4 only supports fp8_e8m0_t and ShapeColumn0 as 64.");
        auto shape = MakeShape(MakeShape(c0Ele, Std::ceil_division(row, c0Ele)),
                               MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(column, FRACTAL_FIXED)));
        auto stride = MakeStride(MakeStride(Std::Int<1>{}, c0Ele * Std::ceil_align(column, FRACTAL_FIXED)),
                                 MakeStride(c0Ele, c0Ele * Std::Int<FRACTAL_FIXED>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<FP4ZnLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeDNFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, Std::Int<1>{}), MakeStride(Std::Int<0>{}, row));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<DNLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeColumnMajorFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, Std::Int<1>{}), MakeStride(Std::Int<0>{}, row));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<ColumnMajorLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeZzFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        auto shape = MakeShape(MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED)),
                               MakeShape(c0Ele, Std::ceil_division(column, c0Ele)));
        auto stride = MakeStride(MakeStride(c0Ele, FRACTAL_FIXED * Std::ceil_align(column, c0Ele)),
                                 MakeStride(Std::Int<1>{}, c0Ele * Std::Int<FRACTAL_FIXED>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<ZzLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleZzFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t> && c0Ele == 2,
            "ScaleZzLayoutPattern only supports fp8_e8m0_t and ShapeColumn0 as 2.");
        auto shape = MakeShape(MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(row, FRACTAL_FIXED)),
                               MakeShape(c0Ele, Std::ceil_division(column, c0Ele)));
        auto stride = MakeStride(MakeStride(c0Ele, FRACTAL_FIXED * Std::ceil_align(column, c0Ele)),
                                 MakeStride(Std::Int<1>{}, c0Ele * Std::Int<FRACTAL_FIXED>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<ScaleZzLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeNnFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t> && c0Ele == 2,
            "NnLayoutPattern only supports fp8_e8m0_t and ShapeColumn0 as 2.");
        auto shape = MakeShape(MakeShape(c0Ele, row / c0Ele), MakeShape(Std::Int<FRACTAL_FIXED>{}, Std::ceil_division(column, FRACTAL_FIXED)));
        auto stride = MakeStride(MakeStride(Std::Int<1>{}, c0Ele * Std::Int<FRACTAL_FIXED>{}),
                                 MakeStride(c0Ele, row * FRACTAL_FIXED));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<NnLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleANDFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t> , "ScaleANDLayoutPattern only supports fp8_e8m0_t.");
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, column), MakeStride(Std::Int<0>{}, Std::Int<1>{}));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<ScaleANDLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleADNFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t> && c0Ele == 2,
            "ScaleADNLayoutPattern only supports fp8_e8m0_t and ShapeColumn0 as 2.");
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(c0Ele, column / c0Ele));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, c0Ele),
                                 MakeStride(Std::Int<1>{}, c0Ele * row));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<ScaleADNLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleBNDFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        constexpr auto c0Ele = TraitType::C0_ELEMENT;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t> && c0Ele == 2,
            "ScaleBNDLayoutPattern only supports fp8_e8m0_t and ShapeColumn0 as 2.");
        auto shape = MakeShape(MakeShape(c0Ele, row / c0Ele), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<1>{}, c0Ele * column),
                                 MakeStride(Std::Int<0>{}, c0Ele));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<ScaleBNDLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};

struct MakeScaleBDNFrameLayout {
    template <typename TraitType, typename T, typename U>
    __aicore__ inline static auto Make(T row, U column) {
        using DataType = typename TraitType::type;
        static_assert(Std::is_same_v<DataType, fp8_e8m0_t> , "ScaleBDNLayoutPattern only supports fp8_e8m0_t.");
        auto shape = MakeShape(MakeShape(Std::Int<1>{}, row), MakeShape(Std::Int<1>{}, column));
        auto stride = MakeStride(MakeStride(Std::Int<0>{}, Std::Int<1>{}), MakeStride(Std::Int<0>{}, row));
        using LayoutT = Layout<decltype(shape), decltype(stride), Std::tuple<ScaleBDNLayoutPattern, TraitType>>;
        return LayoutT(shape, stride);
    }
};


} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_PATTERN_H
