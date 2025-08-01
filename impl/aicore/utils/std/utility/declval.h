//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file declval.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_UTILITY_DECLVAL_H
#define AICORE_UTILS_STD_UTILITY_DECLVAL_H

#include <type_traits>
#include "../type_traits/add_rvalue_reference.h"

namespace AscendC {
namespace Std {

template <typename T>
ASCENDC_HOST_AICORE typename add_rvalue_reference<T>::type declval() noexcept
{
    static_assert(!std::is_abstract<T>::value || std::is_polymorphic<T>::value,
        "Std::declval() cannot be used with polymorphic and abstract types !");
    return static_cast<typename add_rvalue_reference<T>::type>(*static_cast<T*>(nullptr));
}

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_UTILITY_DECLVAL_H
