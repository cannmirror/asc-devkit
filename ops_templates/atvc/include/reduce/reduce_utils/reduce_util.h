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
 * \file reduce_util.h
 * \brief reduce template struct
 */

#ifndef ATVC_REDUCE_REDUCE_UTIL_H
#define ATVC_REDUCE_REDUCE_UTIL_H
#include "common/const_def.h"
#include "reduce/common/patterns.h"

namespace ATVC {
namespace KernelUtils {
template <int32_t dim>
struct Shape {
    int64_t value[dim];
    int64_t oriBurstLen;
};

template <typename T>
__aicore__ inline constexpr int32_t GetCopyInCount()
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, half>::value) {
        return CONST2;
    } else {
        return CONST3;
    }
}

template <typename T>
__aicore__ inline constexpr int32_t GetComputeCount()
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, half>::value) {
        return CONST2;
    } else {
        return CONST0;
    }
}
} // namespace KernelUtils
} // namespace ATVC
#endif