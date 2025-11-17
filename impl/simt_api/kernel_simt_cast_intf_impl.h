/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_CAST_INTERFACE_IMPL_H
#define ASCENDC_MODULE_SIMT_CAST_INTERFACE_IMPL_H

#include "kernel_simt_common_intf_impl.h"

namespace AscendC {
namespace Simt {

template <typename T, typename U, RoundMode roundMode>
__aicore__ inline T Cast(U x)
{
    if constexpr (roundMode == RoundMode::CAST_EVEN || roundMode == RoundMode::CAST_ZERO || roundMode ==
        RoundMode::CAST_FLOOR || roundMode == RoundMode::CAST_CEIL) {
        static_assert(
            SupportType<Tuple<U, T>, Tuple<float, int>, Tuple<int, float>, Tuple<float, int64_t>,
                Tuple<int64_t, float>, Tuple<float, half>, Tuple<float, bfloat16_t>>(),
            "Input type only supports"
            "[(float, int), (int, float), (float, int64), (int64, float), (float, half), (float, bfloat16)]");
        } else if constexpr (roundMode == RoundMode::CAST_NONE) {
            static_assert(SupportType<Tuple<U, T>, Tuple<half, float>, Tuple<bfloat16_t, float>>(),
                "Input type only supports [(half, float), (bfloat16, float)]");
        } else {
            static_assert(roundMode == RoundMode::CAST_EVEN || roundMode == RoundMode::CAST_ZERO || roundMode ==
                RoundMode::CAST_FLOOR || roundMode == RoundMode::CAST_CEIL || roundMode == RoundMode::CAST_NONE,
                    "RoundMode only supports [CAST_EVEN, CAST_ZERO, CAST_FLOOR, CAST_CEIL, CAST_NONE]");
        }
    return CastImpl<T, U, roundMode>(x);
}

template <typename T>
__aicore__ inline T Round(T x)
{
    static_assert(SupportType<T, float, half, bfloat16_t>(), "Input type only supports float, half, bfloat16.");
    return RoundImpl(x);
}

template <typename T>
__aicore__ inline T Rint(T x)
{
    static_assert(SupportType<T, float, half, bfloat16_t>(), "Input type only supports float, half, bfloat16.");
    return RintImpl(x);
}

template <typename T>
__aicore__ inline T Floor(T x)
{
    static_assert(SupportType<T, float, half, bfloat16_t>(), "Input type only supports float, half, bfloat16.");
    return FloorImpl(x);
}

template <typename T>
__aicore__ inline T Ceil(T x)
{
    static_assert(SupportType<T, float, half, bfloat16_t>(), "Input type only supports float, half, bfloat16.");
    return CeilImpl(x);
}

template <typename T>
__aicore__ inline T Trunc(T x)
{
    static_assert(SupportType<T, float, half, bfloat16_t>(), "Input type only supports float, half, bfloat16.");
    return TruncImpl(x);
}

}  // namespace Simt
}  // namespace AscendC
#endif  // ASCENDC_MODULE_SIMT_CAST_INTERFACE_IMPL_H
