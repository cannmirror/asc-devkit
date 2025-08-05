/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file forward.h
 * \brief
 */
#ifndef ATVC_FORWARD_H
#define ATVC_FORWARD_H

#include <type_traits>

namespace ATVC {
template<typename T>
__aicore__ inline constexpr T&& Forward(std::remove_reference_t<T>& param) noexcept {
    return static_cast<T&&>(param);
}

template<typename T>
__aicore__ inline constexpr T&& Forward(std::remove_reference_t<T>&& param) noexcept {
    static_assert(!std::is_lvalue_reference<T>::value, "Invalid Forward of an rvalue as an lvalue");
    return static_cast<T&&>(param);
}

}
#endif
