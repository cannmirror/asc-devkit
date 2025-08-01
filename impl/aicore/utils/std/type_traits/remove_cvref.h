//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file remove_cvref.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_CVREF_H
#define AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_CVREF_H

#include "is_same.h"
#include "remove_cv.h"
#include "remove_reference.h"

namespace AscendC {
namespace Std {

template <typename Tp>
using remove_cvref_t = remove_cv_t<remove_reference_t<Tp>>;

template <typename Tp>
struct remove_cvref {
    using type = remove_cvref_t<Tp>;
};

template <typename Tp, typename Up>
struct is_same_uncvref : IsSame<remove_cvref_t<Tp>, remove_cvref_t<Up>> {};

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_CVREF_H
