/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file utility.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_UTILITY_H
#define AICORE_UTILS_STD_UTILITY_H

#include "impl/utils/std/utility/declval.h"
#include "impl/utils/std/utility/forward.h"
#include "impl/utils/std/utility/move.h"
#include "impl/utils/std/utility/integer_sequence.h"

namespace AscendC {
namespace Std {
template <size_t... Idx>
using index_sequence = IntegerSequence<size_t, Idx...>;
template <size_t N>
using make_index_sequence = MakeIntegerSequence<size_t, N>;
}
}
#endif // AICORE_UTILS_STD_UTILITY_H
