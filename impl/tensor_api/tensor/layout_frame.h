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
    "impl/tensor_api/tensor/layout_frame.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file layout_frame.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H

#include "impl/tensor_api/tensor/layout_pattern.h"

namespace AscendC {
namespace Te {

using LayoutFormatSet = TupleMap<
    Std::tuple<NZLayoutPtn, MakeNzFrameLayout>,
    Std::tuple<NDLayoutPtn, MakeNDFrameLayout>,
    Std::tuple<DNLayoutPtn, MakeDNFrameLayout>,
    Std::tuple<NDExtLayoutPtn, MakeNDExtFrameLayout>,
    Std::tuple<DNExtLayoutPtn, MakeDNExtFrameLayout>,
    Std::tuple<NNLayoutPtn, MakeNnFrameLayout>,
    Std::tuple<ZZLayoutPtn, MakeZzFrameLayout>,
    Std::tuple<ZNLayoutPtn, MakeZnFrameLayout>,
    Std::tuple<ScaleANDLayoutPtn, MakeScaleANDFrameLayout>,
    Std::tuple<ScaleADNLayoutPtn, MakeScaleADNFrameLayout>,
    Std::tuple<ScaleBNDLayoutPtn, MakeScaleBNDFrameLayout>,
    Std::tuple<ScaleBDNLayoutPtn, MakeScaleBDNFrameLayout>>;

template <typename T, size_t C0>
struct LayoutTrait {
    using type = T;
    static constexpr auto C0_ELEMENT = Std::Int<C0>{};
};

template <typename T = uint16_t, size_t C0 = C0_ELEMENT<T>>
struct LayoutTraitDefault : LayoutTrait<T, C0> {};

struct LayoutTraitScale : LayoutTraitDefault<fp8_e8m0_t, 2 / sizeof(fp8_e8m0_t)> {};

struct LayoutTraitFP4 : LayoutTraitDefault<fp4x2_e2m1_t, C0_ELEMENT<fp4x2_e2m1_t>> {};

using FormatTraitSet = TupleMap<
    Std::tuple<NZLayoutPtn, LayoutTraitDefault<>>,
    Std::tuple<NDLayoutPtn, LayoutTraitDefault<>>,
    Std::tuple<DNLayoutPtn, LayoutTraitDefault<>>,
    Std::tuple<NDExtLayoutPtn, LayoutTraitDefault<>>,
    Std::tuple<DNExtLayoutPtn, LayoutTraitDefault<>>,
    Std::tuple<NNLayoutPtn, LayoutTraitDefault<>>,
    Std::tuple<ZZLayoutPtn, LayoutTraitDefault<>>,
    Std::tuple<ZNLayoutPtn, LayoutTraitDefault<>>,
    Std::tuple<ScaleANDLayoutPtn, LayoutTraitScale>,
    Std::tuple<ScaleADNLayoutPtn, LayoutTraitScale>,
    Std::tuple<ScaleBNDLayoutPtn, LayoutTraitScale>,
    Std::tuple<ScaleBDNLayoutPtn, LayoutTraitScale>>;

template <typename T, typename = void>
struct IsFrameLayoutTrait : Std::false_type {};

template <typename T>
struct IsFrameLayoutTrait<T, void_t<typename T::type, decltype(T::C0_ELEMENT)>> : Std::true_type {};

template <typename T>
constexpr bool IsFrameLayoutTraitV = IsFrameLayoutTrait<T>::value;

template <typename T, bool = IsIntegralConstantV<Std::remove_cvref_t<T>>>
struct FrameLayoutTrait {
    using type = LayoutTraitDefault<Std::remove_cvref_t<T>>;
};

template <typename T>
struct FrameLayoutTrait<T, true> {
    using type = LayoutTraitDefault<uint16_t, Std::remove_cvref_t<T>::value>;
};

template <typename T>
using FrameLayoutTraitT = typename FrameLayoutTrait<T>::type;

template <typename LayoutPattern, typename TraitType>
struct GetTrait {
    using type = TraitType;
};

template <typename LayoutPattern>
struct GetTrait<LayoutPattern, LayoutTraitDefault<>> {
    using type = typename FormatTraitSet::template Get<LayoutPattern>;
};

template <typename LayoutPattern, typename TraitType = LayoutTraitDefault<>,
    Std::enable_if_t<IsFrameLayoutTraitV<TraitType>, int> = 0, typename... Args>
__aicore__ inline constexpr decltype(auto) MakeFrameLayout(const Args&... args) {
    using Trait = typename GetTrait<LayoutPattern, TraitType>::type;
    using LayoutMaker = typename LayoutFormatSet::template Get<LayoutPattern>;
    static_assert(!Std::is_same_v<LayoutMaker, EmptyValue>, "Unsupported layout pattern.");
    return LayoutMaker::template Make<Trait>(args...);
}

template <typename LayoutPattern, typename IntTypeOrDataType,
    Std::enable_if_t<!IsFrameLayoutTraitV<IntTypeOrDataType>, int> = 0, typename... Args>
__aicore__ inline constexpr decltype(auto) MakeFrameLayout(const Args&... args) {
    using TraitType = FrameLayoutTraitT<IntTypeOrDataType>;
    return MakeFrameLayout<LayoutPattern, TraitType>(args...);
}

template <typename LayoutPattern, size_t C0Element, typename... Args>
__aicore__ inline constexpr decltype(auto) MakeFrameLayout(const Args&... args) {
    return MakeFrameLayout<LayoutPattern, Std::Int<C0Element>>(args...);
}

template <typename LayoutPattern, typename TraitType = LayoutTraitDefault<>>
struct FrameLayoutFormat {
    template <typename... Args>
    __aicore__ inline constexpr decltype(auto) operator()(const Args&... args) {
        return MakeFrameLayout<LayoutPattern, TraitType>(args...);
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_FRAME_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
