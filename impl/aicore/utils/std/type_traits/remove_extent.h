//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file remove_extent.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_EXTENT_H
#define AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_EXTENT_H

namespace AscendC {
namespace Std {

template <typename Tp>
struct remove_extent {
    using type = Tp;
};

template <typename Tp>
struct remove_extent<Tp[]> {
    using type = Tp;
};

template <typename Tp, size_t Np>
struct remove_extent<Tp[Np]> {
    using type = Tp;
};

template <typename Tp>
using remove_extent_t = typename remove_extent<Tp>::type;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_EXTENT_H
