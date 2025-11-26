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
 * \file kernel_micro_vec_cmpsel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_CMPSEL_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_CMPSEL_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {
template <CMPMODE mode = CMPMODE::EQ, typename RegT>
__aicore__ inline void CompareUint64Impl(MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CompareUint64 is not supported on current device!"); });
}

template <CMPMODE mode = CMPMODE::EQ, typename RegT>
__aicore__ inline void CompareInt64Impl(MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CompareInt64 is not supported on current device!"); });
}

template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename RegT>
__aicore__ inline void CompareImpl(MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    if constexpr (mode == CMPMODE::EQ) {
        vcmp_eq(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::NE) {
        vcmp_ne(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::GT) {
        vcmp_gt(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::GE) {
        vcmp_ge(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::LT) {
        vcmp_lt(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::LE) {
        vcmp_le(dstMask, srcReg0, srcReg1, mask);
    }
}

template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename RegT, typename ScalarT>
__aicore__ inline void CompareScalarImpl(MaskReg &dstMask, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(
        SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    if constexpr (mode == CMPMODE::EQ) {
        vcmps_eq(dstMask, srcReg0, scalarValue, mask);
    } else if constexpr (mode == CMPMODE::NE) {
        vcmps_ne(dstMask, srcReg0, scalarValue, mask);
    } else if constexpr (mode == CMPMODE::GT) {
        vcmps_gt(dstMask, srcReg0, scalarValue, mask);
    } else if constexpr (mode == CMPMODE::GE) {
        vcmps_ge(dstMask, srcReg0, scalarValue, mask);
    } else if constexpr (mode == CMPMODE::LT) {
        vcmps_lt(dstMask, srcReg0, scalarValue, mask);
    } else if constexpr (mode == CMPMODE::LE) {
        vcmps_le(dstMask, srcReg0, scalarValue, mask);
    }
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void SelectImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<T, 1, 2, 4>(),
        "current data type is not supported on current device!");
    if constexpr (sizeof(ActualT) == 1) {
        vsel((RegTensor<uint8_t> &)dstReg, (RegTensor<uint8_t> &)srcReg0, (RegTensor<uint8_t> &)srcReg1, mask);
    } else if constexpr (sizeof(ActualT) == 2) {
        vsel((RegTensor<uint16_t> &)dstReg, (RegTensor<uint16_t> &)srcReg0, (RegTensor<uint16_t> &)srcReg1, mask);
    } else if constexpr (sizeof(ActualT) == 4) {
        vsel((RegTensor<uint32_t> &)dstReg, (RegTensor<uint32_t> &)srcReg0, (RegTensor<uint32_t> &)srcReg1, mask);
    }
}

} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_CMPSEL_IMPL_H