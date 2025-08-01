//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file add_const.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_ADD_CONST_H
#define AICORE_UTILS_STD_TYPE_TRAITS_ADD_CONST_H

namespace AscendC {
namespace Std {

template <typename Tp>
struct add_const {
    using type = const Tp;
};

template <typename Tp>
using add_const_t = typename add_const<Tp>::type;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_ADD_CONST_H
