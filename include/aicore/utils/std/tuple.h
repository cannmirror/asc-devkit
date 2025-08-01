//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file tuple.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TUPLE_H
#define AICORE_UTILS_STD_TUPLE_H

#include "utils/std/tuple/tuple_impl.h"

namespace AscendC {
namespace Std {

// tuple
template <typename... Tps>
class tuple;

// tuple_size
template <typename... Tps>
struct tuple_size;

// tuple_element
template <size_t N, typename... Tps>
struct tuple_element;

// make_tuple
template <typename... Tps>
ASCENDC_HOST_AICORE inline constexpr tuple<unwrap_decay_t<Tps>...> make_tuple(Tps&&... args);

// tie
template <typename... Tps>
ASCENDC_HOST_AICORE inline constexpr tuple<Tps&...> tie(Tps&... args) noexcept;

// get
template <size_t N, typename... Tps>
ASCENDC_HOST_AICORE inline typename tuple_element<N, tuple<Tps...>>::type& get(tuple<Tps...>& t) noexcept;

template <size_t N, typename... Tps>
ASCENDC_HOST_AICORE inline const typename tuple_element<N, tuple<Tps...>>::type& get(const tuple<Tps...>& t) noexcept;

template <size_t N, typename... Tps>
ASCENDC_HOST_AICORE inline typename tuple_element<N, tuple<Tps...>>::type&& get(tuple<Tps...>&& t) noexcept;

template <size_t N, typename... Tps>
ASCENDC_HOST_AICORE inline const typename tuple_element<N, tuple<Tps...>>::type&& get(const tuple<Tps...>&& t) noexcept;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TUPLE_H
