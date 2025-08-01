//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file enable_if.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_ENABLE_IF_H
#define AICORE_UTILS_STD_TYPE_TRAITS_ENABLE_IF_H

namespace AscendC {
namespace Std {

template <bool, typename Tp = void>
struct enable_if {};

template <typename Tp>
struct enable_if<true, Tp> {
    using type = Tp;
};

template <bool Bp, typename Tp = void>
using enable_if_t = typename enable_if<Bp, Tp>::type;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_ENABLE_IF_H
