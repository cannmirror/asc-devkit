//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file type_traits.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_H
#define AICORE_UTILS_STD_TYPE_TRAITS_H

#include "utils/std/type_traits/add_const.h"
#include "utils/std/type_traits/add_cv.h"
#include "utils/std/type_traits/add_lvalue_reference.h"
#include "utils/std/type_traits/add_pointer.h"
#include "utils/std/type_traits/add_rvalue_reference.h"
#include "utils/std/type_traits/add_volatile.h"
#include "utils/std/type_traits/conditional.h"
#include "utils/std/type_traits/decay.h"
#include "utils/std/type_traits/enable_if.h"
#include "utils/std/type_traits/integral_constant.h"
#include "utils/std/type_traits/is_array.h"
#include "utils/std/type_traits/is_base_of.h"
#include "utils/std/type_traits/is_class.h"
#include "utils/std/type_traits/is_constant.h"
#include "utils/std/type_traits/is_const.h"
#include "utils/std/type_traits/is_convertible.h"
#include "utils/std/type_traits/is_floating_point.h"
#include "utils/std/type_traits/is_function.h"
#include "utils/std/type_traits/is_integral.h"
#include "utils/std/type_traits/is_pointer.h"
#include "utils/std/type_traits/is_referenceable.h"
#include "utils/std/type_traits/is_reference.h"
#include "utils/std/type_traits/is_same.h"
#include "utils/std/type_traits/is_tuple.h"
#include "utils/std/type_traits/is_union.h"
#include "utils/std/type_traits/is_void.h"
#include "utils/std/type_traits/remove_const.h"
#include "utils/std/type_traits/remove_cv.h"
#include "utils/std/type_traits/remove_cvref.h"
#include "utils/std/type_traits/remove_extent.h"
#include "utils/std/type_traits/remove_pointer.h"
#include "utils/std/type_traits/remove_reference.h"
#include "utils/std/type_traits/remove_volatile.h"

namespace AscendC {
namespace Std {

// enable_if
template <bool, typename Tp>
struct enable_if;

// conditional
template <bool Bp, typename If, typename Then>
struct conditional;

// is_convertible
template <typename From, typename To>
struct is_convertible;

// is_base_of
template <typename Base, typename Derived>
struct is_base_of;

// is_same
template <typename Tp, typename Up>
struct is_same;

// integral_constant
template <typename Tp, Tp v>
struct integral_constant;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_H
