/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file declval.h
 * \brief
 */
#ifndef IMPL_STD_ASCENDC_STD_DECLVAL_IMPL__H
#define IMPL_STD_ASCENDC_STD_DECLVAL_IMPL__H

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

}
}

#endif // IMPL_STD_ASCENDC_STD_DECLVAL_IMPL__H
