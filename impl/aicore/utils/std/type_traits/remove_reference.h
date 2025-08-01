//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file remove_reference.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_REFERENCE_H
#define AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_REFERENCE_H

namespace AscendC {
namespace Std {

template <typename Tp>
struct remove_reference {
    using type = Tp;
};

template <typename Tp>
struct remove_reference<Tp&> {
    using type = Tp;
};

template <typename Tp>
struct remove_reference<Tp&&> {
    using type = Tp;
};

template <typename Tp>
using remove_reference_t = typename remove_reference<Tp>::type;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_REFERENCE_H
