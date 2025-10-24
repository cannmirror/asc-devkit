/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file conv_integral_constant.h
 * \brief
 */
#ifndef UTILS_CONV_INTEGRAL_CONSTANT_H
#define UTILS_CONV_INTEGRAL_CONSTANT_H
namespace AscendC {
namespace Std {
template <typename...>
struct always_false : public false_type {};

template <typename... Tp>
constexpr bool always_false_v = always_false<Tp...>::value;
} // namespace Std
} // namespace AscendC

namespace Act {
namespace Conv {
template <int32_t t>
using Int = AscendC::Std::integral_constant<int32_t, t>;

// Unary operator
template <auto t>
__host_aicore__ inline constexpr Int<+t> operator+(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<-t> operator-(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<~t> operator~(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<!t> operator!(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<*t> operator*(Int<t>)
{
    return {};
}

// Binary operator
template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t + u)> operator+(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t - u)> operator-(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t * u)> operator*(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t / u)> operator/(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t % u)> operator%(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t & u)> operator&(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t | u)> operator|(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t ^ u)> operator^(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t << u)> operator<<(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t >> u)> operator>>(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t && u)> operator&&(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t || u)> operator||(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t == u)> operator==(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t != u)> operator!=(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t > u)> operator>(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t < u)> operator<(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t >= u)> operator>=(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t <= u)> operator<=(Int<t>, Int<u>)
{
    return {};
}
} // namespace Conv
} // namespace Act
#endif