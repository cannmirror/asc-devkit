//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file remove_cv.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_CV_H
#define AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_CV_H

#include "remove_const.h"
#include "remove_volatile.h"

namespace AscendC {
namespace Std {

template <typename Tp>
struct remove_cv {
    using type = remove_volatile_t<remove_const_t<Tp>>;
};

template <typename Tp>
using remove_cv_t = remove_volatile_t<remove_const_t<Tp>>;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_REMOVE_CV_H
