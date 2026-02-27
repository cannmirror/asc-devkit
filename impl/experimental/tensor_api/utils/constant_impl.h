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
* \file constant_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H
#define IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H

#include <cstdint>
#include <utility>
#include <type_traits>
#include "impl/experimental/tensor_api/utils/macro_impl.h"
#include "include/utils/std/tuple.h"
#include "include/utils/std/type_traits.h"
#include "include/utils/std/utility.h"
#include "include/utils/std/algorithm.h"

namespace AscendC {
namespace Te {
constexpr size_t TWO_DIM_DATA = 2;
constexpr size_t FOUR_DIM_DATA = 4;
constexpr size_t C0_SIZE = 32;
constexpr size_t FRACTAL_FIXED = 16;
constexpr size_t DISABLE_COORD = 0;
constexpr size_t ENABLE_COORD = 1;
constexpr size_t SHIFT_LEFT_16 = 0x00010000;
constexpr size_t L2_CACHE_OFFSET = 60;
constexpr size_t MX_SCALE_K0 = 2;

enum class LayoutFormat : uint8_t { NZ, ZN, ZZ, DN, ND, NN};

template <QuantMode_t Value, QuantMode_t... Args>
struct is_one_of_value : Std::false_type {};

template <typename TupleType>
using tuple_sequence = Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<TupleType>>>;

template <typename T, typename U>
__aicore__ inline constexpr auto max(const T src0, const U src1)
{
    static_assert(Std::is_same_v<T, U> || Std::is_integral_v<T>, "Only support compare with same type");
    return (src0 > src1) ? src0 : src1;
}

template <QuantMode_t Value, QuantMode_t Head, QuantMode_t... Tail>
struct is_one_of_value<Value, Head, Tail...>
    : Std::bool_constant<(Value == Head) || is_one_of_value<Value, Tail...>::value> {};

template <QuantMode_t Value, QuantMode_t... Args>
inline constexpr bool is_one_of_value_v = is_one_of_value<Value, Args...>::value;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define SCALAR_QUANT_MODE QuantMode_t::DEQF16, QuantMode_t::QF322B8_PRE, QuantMode_t::REQ8,\
    QuantMode_t::QS322BF16_PRE, QuantMode_t::QF322F16_PRE, QuantMode_t::QF322BF16_PRE, QuantMode_t::QF322FP8_PRE,\
    QuantMode_t::QF322HIF8_PRE, QuantMode_t::QF322HIF8_PRE_HYBRID, QuantMode_t::QF322F32_PRE
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define SCALAR_QUANT_MODE QuantMode_t::DEQF16, QuantMode_t::QF322B8_PRE, QuantMode_t::REQ8
#else
#define SCALAR_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsScalarQuantMode = is_one_of_value<quantPre, SCALAR_QUANT_MODE>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::VDEQF16, QuantMode_t::VQF322B8_PRE, QuantMode_t::VREQ8,\
    QuantMode_t::VQS322BF16_PRE, QuantMode_t::VQF322F16_PRE, QuantMode_t::VQF322BF16_PRE, QuantMode_t::VQF322FP8_PRE,\
    QuantMode_t::VQF322HIF8_PRE, QuantMode_t::VQF322HIF8_PRE_HYBRID, QuantMode_t::VQF322F32_PRE
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::VDEQF16, QuantMode_t::VQF322B8_PRE, QuantMode_t::VREQ8
#else
#define TILE_OP_INTERNAL_TENSOR_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsVectorQuantMode = is_one_of_value<quantPre, TILE_OP_INTERNAL_TENSOR_QUANT_MODE>;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::F322F16, QuantMode_t::F322BF16
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::F322F16, QuantMode_t::F322BF16
#else
#define TILE_OP_INTERNAL_DIRECT_QUANT_MODE QuantMode_t::NoQuant
#endif

template <QuantMode_t quantPre>
using IsDirectQuantMode = is_one_of_value<quantPre, TILE_OP_INTERNAL_DIRECT_QUANT_MODE>;

using ZeroCoord2DType = AscendC::Std::tuple<Std::Int<0>, Std::Int<0>>;

using EmptyShape = AscendC::Std::tuple<AscendC::Std::tuple<Std::Int<0>, Std::Int<0>>, 
    AscendC::Std::tuple<Std::Int<0>, Std::Int<0>>>;

// IsIntegralConstant
template <typename T>
struct IsIntegralConstant : Std::false_type {};

template <size_t Value>
struct IsIntegralConstant<Std::Int<Value>> : Std::true_type {};

template <typename T>
constexpr bool IsIntegralConstantV = IsIntegralConstant<T>::value;

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_UTILS_CONSTANT_IMPL_H