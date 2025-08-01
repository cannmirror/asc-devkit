//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file add_rvalue_reference.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_ADD_RVALUE_REFERENCE_H
#define AICORE_UTILS_STD_TYPE_TRAITS_ADD_RVALUE_REFERENCE_H

#include "is_referenceable.h"

namespace AscendC {
namespace Std {

template <typename Tp, bool = is_referenceable<Tp>::value>
struct AddRvalueReferenceImpl {
    using type = Tp;
};

template <typename Tp>
struct AddRvalueReferenceImpl<Tp, true> {
    using type = Tp&&;
};

template <typename Tp>
using add_rvalue_reference_t = typename AddRvalueReferenceImpl<Tp>::type;

template <typename Tp>
struct add_rvalue_reference {
    using type = add_rvalue_reference_t<Tp>;
};

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_ADD_RVALUE_REFERENCE_H
