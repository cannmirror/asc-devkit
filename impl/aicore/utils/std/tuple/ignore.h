//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file ignore.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TUPLE_IGNORE_H
#define AICORE_UTILS_STD_TUPLE_IGNORE_H

namespace AscendC {
namespace Std {

struct ignore_t {
    ASCENDC_HOST_AICORE inline ignore_t() = default;

    template <typename Tp, typename... Ts>
    ASCENDC_HOST_AICORE inline constexpr ignore_t(const Tp&, const Ts&...) noexcept
    {}

    template <typename Tp>
    ASCENDC_HOST_AICORE inline constexpr const ignore_t& operator=(const Tp&) const noexcept
    {
        return *this;
    }
};

constexpr ignore_t ignore{};

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TUPLE_IGNORE_H
