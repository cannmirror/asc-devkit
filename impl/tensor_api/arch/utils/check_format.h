/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/tensor_api/arch/utils/check_format.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file check_format.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_UTILS_CHECK_FORMAT_H
#define IMPL_TENSOR_API_ARCH_UTILS_CHECK_FORMAT_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/tensor/pointer_impl.h"
#include "impl/tensor_api/tensor/local_tensor_impl.h"
#include "impl/tensor_api/arch/utils/is_format.h"

namespace AscendC {
namespace Te {
struct CheckNzLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        constexpr auto C0_ELEMENT = TraitType::C0_ELEMENT;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
                      "Layout->Shape->Row->ZeroDim must be Int<16>!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_ELEMENT>>,
                      "Layout->Shape->Column->ZeroDim is differnt from C0_ELEMENT!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_ELEMENT>>,
                      "Layout->Stride->Row->ZeroDim is differnt from C0_ELEMENT!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
                      "Layout->Stride->Column->ZeroDim must be Int<1>!");
        static_assert(
            Std::is_same_v<StrideRow1, Std::Int<C0_ELEMENT * FRACTAL_FIXED>>,
            "Layout->Stride->Column->ZeroDim is differnt from C0_ELEMENT * FRACTAL_FIXED!");
    }
};

struct CheckNDLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Layout->Shape->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Layout->Shape->Column->ZeroDim must be Int<1>!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Layout->Stride->Row->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Layout->Stride->Column->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "Layout->Stride->Column->OneDim must be Int<1>!");
    }
};

struct CheckDNLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Src->Layout->Shape->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Src->Layout->Shape->Column->ZeroDim must be Int<1>!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Src->Layout->Stride->Row->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>, "Src->Layout->Stride->Row->OneDim must be Int<1>!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Src->Layout->Stride->Column->ZeroDim must be Int<0>!");
    }
};

struct CheckNnLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<2>>, "Src->Layout->Shape->Row->ZeroDim must be Int<2>!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<16>>, "Src->Layout->Shape->Column->ZeroDim must be Int<16>!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>, "Src->Layout->Stride->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<32>>, "Src->Layout->Stride->Row->OneDim must be Int<32>!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<2>>, "Src->Layout->Stride->Column->ZeroDim must be Int<2>!");
    }
};

struct CheckZzLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        constexpr auto C0_ELEMENT = TraitType::C0_ELEMENT;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_ELEMENT>>,
                      "Layout->Shape->Column->ZeroDim must is differnt from C0_ELEMENT!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
                      "Layout->Shape->Row->ZeroDim must be Int<16>!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Layout->Stride->Column-ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_ELEMENT>>,
                      "Layout->Stride->Row->ZeroDim is differnt from C0_ELEMENT!");
    }
};

struct CheckZnLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        constexpr auto C0_ELEMENT = TraitType::C0_ELEMENT;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
                      "Layout->Shape->Column->ZeroDim must be Int<16>!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<C0_ELEMENT>>,
                      "Layout->Shape->Row->ZeroDim is differnt from C0_ELEMENT!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<C0_ELEMENT>>,
                      "Layout->Stride->Column-ZeroDim is differnt from C0_ELEMENT!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>,
                      "Layout->Stride->Row->ZeroDim must be Int<1>!");
    }
};

struct CheckScaleANDLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Layout->Shape->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Layout->Shape->Column->ZeroDim must be Int<1>!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Layout->Stride->Row->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Layout->Stride->Column-ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "Layout->Stride->Column-OneDim must be Int<1>!");
    }
};

struct CheckScaleADNLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Layout->Shape->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<2>>, "Layout->Shape->Column->ZeroDim must be Int<2>!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Layout->Stride->Row->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<2>>, "Layout->Stride->Row->OneDim must be Int<2>!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Layout->Stride->Column-OneDim must be Int<1>!");
    }
};

struct CheckScaleBNDLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<2>>, "Layout->Shape->Row->ZeroDim must be Int<2>!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Layout->Shape->Column->ZeroDim must be Int<1>!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>, "Layout->Stride->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Layout->Stride->Column-ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<2>>, "Layout->Stride->Column-OneDim must be Int<2>!");
    }
};

struct CheckScaleBDNLayoutPattern {
    template <typename T, typename TraitType>
    __aicore__ inline static constexpr void Check() {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Layout->Shape->Row->ZeroDim must be Int<1>!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Layout->Shape->Column->ZeroDim must be Int<1>!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Layout->Stride->Row->ZeroDim must be Int<0>!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>, "Layout->Stride->Row->OneDim must be Int<1>!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Layout->Stride->Column-OneDim must be Int<0>!");
    }
};

using LayoutPatternCheckSet = TupleMap<
    Std::tuple<ZnLayoutPtn, CheckZnLayoutPattern>,
    Std::tuple<ZzLayoutPtn, CheckZzLayoutPattern>,
    Std::tuple<NnLayoutPtn, CheckNnLayoutPattern>,
    Std::tuple<NzLayoutPtn, CheckNzLayoutPattern>,
    Std::tuple<NDLayoutPtn, CheckNDLayoutPattern>,
    Std::tuple<DNLayoutPtn, CheckDNLayoutPattern>,
    Std::tuple<ScaleANDLayoutPtn, CheckScaleANDLayoutPattern>,
    Std::tuple<ScaleADNLayoutPtn, CheckScaleADNLayoutPattern>,
    Std::tuple<ScaleBNDLayoutPtn, CheckScaleBNDLayoutPattern>,
    Std::tuple<ScaleBDNLayoutPtn, CheckScaleBDNLayoutPattern>>;

template <typename T>
__aicore__ inline decltype(auto) CheckLayoutPattern() {
    using Layout = typename T::layoutType;
    using LayoutPattern = GetLayoutPattern<Layout>;
    using TraitType = GetLayoutTrait<Layout>;
    using PatternCheck = typename LayoutPatternCheckSet::template Get<LayoutPattern>;
    static_assert(!Std::is_same_v<PatternCheck, EmptyValue>, "Unsupported layout pattern.");
    PatternCheck::template Check<T, TraitType>();
}

class CheckFormat {
public:
    template <typename T>
    __aicore__ inline static constexpr void CheckZZTemplate()
    {
        using dataType = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_ELEMENT<dataType>>>,
                      "Layout->Shape->Column->ZeroDim must be 32/sizeof(dataType)!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
                      "Layout->Shape->Row->ZeroDim must be 16!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Layout->Stride->Column-ZeroDim must be 1!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_ELEMENT<dataType>>>,
                      "Layout->Stride->Row->ZeroDim must be 32/sizeof(dataType)!");
    }

    template <typename T>
    __aicore__ inline static constexpr void CheckZNTemplate()
    {
        using dataType = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
                      "Filter Layout->Shape->Column->ZeroDim must be 16!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<C0_ELEMENT<dataType>>>,
                      "Filter Layout->Shape->Row->ZeroDim must be (is_b4_type<dataType> ? 64 : 32 / sizeof(dataType))!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<C0_ELEMENT<dataType>>>,
                      "Filter Layout->Stride->Column-ZeroDim must be (is_b4_type<dataType> ? 64 : 32 / sizeof(dataType))!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>,
                      "Filter Layout->Stride->Row->ZeroDim must be 1!");
    }

    template <typename T>
    __aicore__ inline static constexpr void CheckNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
                      "Layout->Shape->Row->ZeroDim must be 16!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_ELEMENT<type>>>,
                      "Layout->Shape->Column->ZeroDim must be (is_b4_type<dataType> ? 64 : 32 / sizeof(dataType))!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_ELEMENT<type>>>,
                      "Layout->Stride->Row->ZeroDim must be (is_b4_type<dataType> ? 64 : 32 / sizeof(dataType))!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
                      "Layout->Stride->Column->ZeroDim must be 1!");
        static_assert(
            Std::is_same_v<StrideRow1, Std::Int<C0_ELEMENT<type> * FRACTAL_FIXED>>,
            "Layout->Stride->Column->ZeroDimmust be (is_b4_type<dataType> ? 64 : 32 / sizeof(dataType) * 16)!");
    }

    template <typename T, bool enableChannelSplit>
    __aicore__ inline static constexpr void CheckFixpipeNZTemplate()
    {
        using Dtype = typename T::elementType;

        constexpr bool isB32 = is_one_of_attr_v<Dtype, float, int32_t>;
        if constexpr (enableChannelSplit) {
            static_assert(isB32, "When enable channel split, data type must be B32");
        }

        using type = typename Std::conditional<!isB32 || enableChannelSplit, Dtype, half>::type;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;

        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Layout->Shape->Row->ZeroDim must be 16!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_ELEMENT<type>>>,
                      "Layout->Shape->Column->ZeroDim must be (is_b4_type<dataType> ? 64 : 32 / sizeof(dataType))!");

        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_ELEMENT<type>>>,
                      "Layout->Stride->Row->ZeroDim must be (is_b4_type<dataType> ? 64 : 32 / sizeof(dataType))!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Layout->Stride->Column->ZeroDim must be 1!");
        static_assert(
            Std::is_same_v<StrideRow1, Std::Int<C0_ELEMENT<type> * FRACTAL_FIXED>>,
            "Layout->Stride->Column->ZeroDimmust be (is_b4_type<dataType> ? 64 : 32 / sizeof(dataType) * 16)!");
    }

    template <typename T>
    __aicore__ inline static constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Layout->Shape->Row->ZeroDim must be 1!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Layout->Shape->Column->ZeroDim must be 1!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Layout->Stride->Row->ZeroDim must be 0!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Layout->Stride->Column->ZeroDim must be 0!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "Layout->Stride->Column->OneDim must be 1!");
    }

    template <typename T>
    __aicore__ inline static constexpr void CheckNDFp8Template()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<2>>, "CopyCbufToFB Layout->Shape->Row->ZeroDim must be 2!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
                      "CopyCbufToFB Layout->Shape->Column->ZeroDim must be 1!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>, "CopyCbufToFB Layout->Stride->Row->ZeroDim must be 1!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
                      "CopyCbufToFB Layout->Stride->Column->ZeroDim must be 0!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<MX_SCALE_K0>>,
                      "CopyCbufToFB Layout->Stride->Column->OneDim must be 2!");
    }

    template <typename T>
    __aicore__ inline static constexpr void CheckL0CNZTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
                      "Layout->Shape->Row->ZeroDim must be 16!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
                      "Layout->Shape->Column->ZeroDim must be 16!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<FRACTAL_FIXED>>,
                      "Layout->Stride->Row->ZeroDim must be 16!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Layout->Stride->Column->ZeroDim must be 1!");
    }

    template <typename T>
    __aicore__ inline static constexpr void CheckDNTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Src->Layout->Shape->Row->ZeroDim must be 1!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Src->Layout->Shape->Column->ZeroDim must be 1!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Src->Layout->Stride->Row->ZeroDim must be 0!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>, "Src->Layout->Stride->Row->OneDim must be 1!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Src->Layout->Stride->Column->ZeroDim must be 0!");
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_UTILS_CHECK_FORMAT_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif