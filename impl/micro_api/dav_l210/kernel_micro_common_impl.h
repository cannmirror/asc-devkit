/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_micro_common_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_COMMON_IMPL_H
#define ASCENDC_MODULE_MICRO_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "micro_api/kernel_micro_utils.h"
#include "micro_api/kernel_micro_common_intf.h"

namespace AscendC {
namespace MicroAPI {

#define GetLoopBoundB8(x) get_vloopn_bound_b8(x)

template <typename T, StoreDist dist> __aicore__ inline constexpr StoreDist GetStoreDist()
{
    if constexpr (dist == StoreDist::DIST_NORM) {
        static_assert(SupportBytes<T, 1, 2, 4>(), "StoreDist DIST_NORM only support type b8/b16/b32 on current device");
        if constexpr (sizeof(T) == 1) {
            return StoreDist::DIST_NORM_B8;
        } else if constexpr (sizeof(T) == 2) {
            return StoreDist::DIST_NORM_B16;
        } else if constexpr (sizeof(T) == 4) {
            return StoreDist::DIST_NORM_B32;
        } else if constexpr (sizeof(T) == 8) {
            return StoreDist::DIST_NORM_B32;
        }
    }
    return dist;
}

template <typename RegT, const RegTrait &otherTrait = RegTraitNumOne> constexpr __aicore__ inline bool CheckRegTrait()
{
    constexpr RegTrait regTrait = RegT::trait;
    return regTrait.REG_NUM == otherTrait.REG_NUM;
}

template <RoundMode mode> __aicore__ inline constexpr ::ROUND GetRound()
{
// To avoid naming conflicts of ROUND member variables in cpu debug,
// the names of the returned member variables are changed to be different from those of
// the ROUND enumeration class of the compiler.
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    if constexpr (mode == RoundMode::CAST_RINT) {
        return ::ROUND::CAST_RINT;
    } else if constexpr (mode == RoundMode::CAST_ROUND) {
        return ::ROUND::CAST_ROUND;
    } else if constexpr (mode == RoundMode::CAST_FLOOR) {
        return ::ROUND::CAST_FLOOR;
    } else if constexpr (mode == RoundMode::CAST_CEIL) {
        return ::ROUND::CAST_CEIL;
    } else if constexpr (mode == RoundMode::CAST_TRUNC) {
        return ::ROUND::CAST_TRUNC;
    } else if constexpr (mode == RoundMode::CAST_ODD) {
        return ::ROUND::CAST_ODD;
    } else {
        return ::ROUND::CAST_HYBRID;
    }
#else
    if constexpr (mode == RoundMode::CAST_RINT) {
        return ::ROUND::R;
    } else if constexpr (mode == RoundMode::CAST_ROUND) {
        return ::ROUND::A;
    } else if constexpr (mode == RoundMode::CAST_FLOOR) {
        return ::ROUND::F;
    } else if constexpr (mode == RoundMode::CAST_CEIL) {
        return ::ROUND::C;
    } else if constexpr (mode == RoundMode::CAST_TRUNC) {
        return ::ROUND::Z;
    } else if constexpr (mode == RoundMode::CAST_ODD) {
        return ::ROUND::O;
    } else {
        return ::ROUND::H;
    }
#endif
}

template <MaskMergeMode mode> __aicore__ inline constexpr auto GetMaskMergeMode()
{
// To avoid naming conflicts of mode struct in cpu debug.
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    return std::integral_constant<::CpuMode, static_cast<::CpuMode>(mode)>();
#else
    return std::integral_constant<::Mode, static_cast<::Mode>(mode)>();
#endif
}

template <MemType src, MemType dst> __aicore__ inline void LocalMemBarImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LocalMemBar is not supported on current device!"); });
}

} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_COMMON_IMPL_H