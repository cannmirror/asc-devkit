//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file utility.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_UTILITY_H
#define AICORE_UTILS_STD_UTILITY_H

#include "utils/std/utility/declval.h"
#include "utils/std/utility/forward.h"
#include "utils/std/utility/move.h"
#include "utils/std/utility/integer_sequence.h"

namespace AscendC {
namespace Std {
template <size_t... Idx>
using index_sequence = IntegerSequence<size_t, Idx...>;
template <size_t N>
using make_index_sequence = MakeIntegerSequence<size_t, N>;
} // namespace Std
} // namespace AscendC
#endif // AICORE_UTILS_STD_UTILITY_H
