//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file add_pointer.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_ADD_POINTER_H
#define AICORE_UTILS_STD_TYPE_TRAITS_ADD_POINTER_H

#include "is_referenceable.h"
#include "is_same.h"
#include "is_void.h"
#include "remove_cv.h"
#include "remove_reference.h"

namespace AscendC {
namespace Std {

template <typename Tp, bool = is_referenceable<Tp>::value || is_void<Tp>::value>
struct AddPointerImpl {
    using type = remove_reference_t<Tp>*;
};

template <typename Tp>
struct AddPointerImpl<Tp, false> {
    using type = Tp;
};

template <typename Tp>
using add_pointer_t = typename AddPointerImpl<Tp>::type;

template <typename Tp>
struct add_pointer {
    using type = add_pointer_t<Tp>;
};

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_ADD_POINTER_H
