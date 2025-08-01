//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file remove_volatile.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_VOLATILE_H
#define AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_VOLATILE_H

namespace AscendC {
namespace Std {

template <typename Tp>
struct remove_volatile {
    using type = Tp;
};

template <typename Tp>
struct remove_volatile<volatile Tp> {
    using type = Tp;
};

template <typename Tp>
using remove_volatile_t = typename remove_volatile<Tp>::type;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_VOLATILE_H
