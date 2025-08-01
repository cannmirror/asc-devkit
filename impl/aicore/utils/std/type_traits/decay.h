//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file decay.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_DECAY_H
#define AICORE_UTILS_STD_TYPE_TRAITS_DECAY_H

#include "add_pointer.h"
#include "conditional.h"
#include "is_array.h"
#include "is_function.h"
#include "is_referenceable.h"
#include "remove_cv.h"
#include "remove_extent.h"
#include "remove_reference.h"

namespace AscendC {
namespace Std {

template <typename Up, bool>
struct DecayImpl {
    using type = remove_cv_t<Up>;
};

template <typename Up>
struct DecayImpl<Up, true> {
public:
    using type = conditional_t<is_array<Up>::value, remove_extent_t<Up>*,
        conditional_t<is_function<Up>::value, add_pointer_t<Up>, remove_cv_t<Up>>>;
};

template <typename Tp>
struct decay {
private:
    using Up = remove_reference_t<Tp>;

public:
    using type = typename DecayImpl<Up, is_referenceable<Up>::value>::type;
};

template <typename Tp>
using decay_t = typename decay<Tp>::type;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_DECAY_H
