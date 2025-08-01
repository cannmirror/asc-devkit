//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file conditional.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_CONDITIONAL_H
#define AICORE_UTILS_STD_TYPE_TRAITS_CONDITIONAL_H

namespace AscendC {
namespace Std {

namespace conditional_impl {

template <bool>
struct IfImpl;

template <>
struct IfImpl<true> {
    template <typename IfRes, typename ElseRes>
    using Select = IfRes;
};

template <>
struct IfImpl<false> {
    template <typename IfRes, typename ElseRes>
    using Select = ElseRes;
};

template <bool Cond, typename IfRes, typename ElseRes>
using If = typename IfImpl<Cond>::template Select<IfRes, ElseRes>;

} // namespace conditional_impl

template <bool Bp, typename If, typename Then>
struct conditional {
    using type = If;
};

template <typename If, typename Then>
struct conditional<false, If, Then> {
    using type = Then;
};

template <bool Bp, typename If, typename Then>
using conditional_t = typename conditional<Bp, If, Then>::type;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_CONDITIONAL_H
