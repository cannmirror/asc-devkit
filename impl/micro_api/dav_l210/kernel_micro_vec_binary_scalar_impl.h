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
 * \file kernel_micro_vec_binary_scalar_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_BINARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_BINARY_SCALAR_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {

#define BINARY_OP_VEC_SCALAR_IMPL_SAME_TYPE(INSTRUCTION, REG_T, SCALAR_T, MASK_MODE, ...)                          \
    using ActualT = typename REG_T::ActualT;                                                                       \
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");         \
    static_assert(SupportType<ActualT, ##__VA_ARGS__>(), "current data type is not supported on current device!"); \
    static_assert(                                                                                                 \
        SupportType<SCALAR_T, ##__VA_ARGS__>(), "current scalar data type is not supported on current device!");   \
    static_assert(SupportEnum<MASK_MODE, MaskMergeMode::MERGING>(),                                                \
        "current api only supported Mode MERGING on current device!");                                             \
    constexpr auto modeValue = GetMaskMergeMode<mode>();                                                           \
    INSTRUCTION(dstReg, srcReg0, scalar, mask, modeValue)

#define BINARY_OP_VEC_SCALAR_IMPL_FIX_SCALAR_TYPE(INSTRUCTION, REG_T, SCALAR_T, SCALAR_ALLOW_T, MASK_MODE, ...)      \
    using ActualT = typename REG_T::ActualT;                                                                       \
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");         \
    static_assert(SupportType<ActualT, ##__VA_ARGS__>(), "current data type is not supported on current device!"); \
    static_assert(                                                                                                 \
        SupportType<SCALAR_T, SCALAR_ALLOW_T>(), "current scalar data type is not supported on current device!");  \
    static_assert(SupportEnum<MASK_MODE, MaskMergeMode::MERGING>(),                                                \
        "current api only supported Mode MERGING on current device!");                                             \
    constexpr auto modeValue = GetMaskMergeMode<mode>();                                                           \
    INSTRUCTION(dstReg, srcReg0, scalar, mask, modeValue)

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void AddsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalar, MaskReg &mask)
{
    BINARY_OP_VEC_SCALAR_IMPL_SAME_TYPE(vadds, RegT, ScalarT, mode, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MulsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalar, MaskReg &mask)
{
    BINARY_OP_VEC_SCALAR_IMPL_SAME_TYPE(vmuls, RegT, ScalarT, mode, uint8_t, int8_t, uint16_t, int16_t, int32_t, half, float);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MaxsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalar, MaskReg &mask)
{
    BINARY_OP_VEC_SCALAR_IMPL_SAME_TYPE(vmaxs, RegT, ScalarT, mode, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MinsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalar, MaskReg &mask)
{
    BINARY_OP_VEC_SCALAR_IMPL_SAME_TYPE(vmins, RegT, ScalarT, mode, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ShiftLeftsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalar, MaskReg &mask)
{
    BINARY_OP_VEC_SCALAR_IMPL_FIX_SCALAR_TYPE(vshls, RegT, ScalarT, int16_t, mode, uint8_t, uint16_t, int16_t, uint32_t, int32_t);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ShiftRightsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalar, MaskReg &mask)
{
    BINARY_OP_VEC_SCALAR_IMPL_FIX_SCALAR_TYPE(vshrs, RegT, ScalarT, int16_t, mode, uint8_t, uint16_t, int16_t, uint32_t, int32_t);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void RoundsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalar, MaskReg &mask)
{
    BINARY_OP_VEC_SCALAR_IMPL_FIX_SCALAR_TYPE(vrnds, RegT, ScalarT, uint16_t, mode, int16_t, int32_t);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void LeakyReluImpl(RegT &dstReg, RegT &srcReg0, T scalar, MaskReg &mask)
{
    BINARY_OP_VEC_SCALAR_IMPL_SAME_TYPE(vlrelu, RegT, T, mode, half, float);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_BINARY_SCALAR_IMPL_H
