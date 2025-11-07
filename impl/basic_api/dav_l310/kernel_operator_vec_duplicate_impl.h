/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_vec_duplicate_impl.h
 * \brief AscendC l310 support vector duplicate api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#include <type_traits>
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckDuplicateSupportedType()
{
    static_assert(std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value ||
        std::is_same<T, half>::value || std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, float>::value,
        "Duplicate instr only support uint8_t/int8_t/half/int16_t/uint16_t/int32_t/uint32_t/float type");
}

// level 2
template <typename T>
typename std::enable_if_t<
!std::is_same<T, uint8_t>::value &&
!std::is_same<T, int8_t>::value &&
!std::is_same<T, uint16_t>::value &&
!std::is_same<T, int16_t>::value &&
!std::is_same<T, half>::value &&
!std::is_same<T, uint32_t>::value &&
!std::is_same<T, int32_t>::value &&
!std::is_same<T, float>::value
>
__aicore__ inline DuplicateImpl(__ubuf__ T* dst, const T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
typename std::enable_if_t<
std::is_same<T, uint8_t>::value ||
std::is_same<T, int8_t>::value ||
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value ||
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline DuplicateImpl(__ubuf__ T* dst, const T scalarValue, const int32_t& count)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vDst;
        uint32_t sreg = (uint32_t)count;
        MaskReg preg;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTime = CeilDivision(count, sregLower);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            preg = CreatePredicate<T>(sreg);
            Duplicate(vDst, scalarValue, preg);
            DataCopy(dst, vDst, i * sregLower, preg);
        }
    }
}

// level 0, continuous mode
template <typename T, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint8_t>::value ||
std::is_same<T, int8_t>::value ||
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value ||
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline DuplicateImpl(__ubuf__ T* dst, const T scalarValue, uint64_t mask,
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vDst;
        uint32_t sreg = (uint32_t)mask;
        MaskReg preg = CreatePredicate<T>(sreg);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            Duplicate(vDst, scalarValue, preg);
            DataCopy(dst, vDst, dstBlockStride, i * dstRepeatStride, preg);
        }
    }
}

// level 0, bit mode
template <typename T, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, int8_t>::value ||
std::is_same<T, uint8_t>::value
>
__aicore__ inline DuplicateImpl(__ubuf__ T* dst, const T scalarValue, uint64_t mask[2],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    SetVectorMask<uint16_t>(mask[1], mask[0]);
    __VEC_SCOPE__
    {
        RegTensor<T> vDst;
        MaskReg preg = movp_b16();
        MaskReg preg1;
        PredicatePack(preg1, preg);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            Duplicate(vDst, scalarValue, preg1);
            DataCopy(dst, vDst, dstBlockStride, i * dstRepeatStride, preg1);
        }
    }
}

template <typename T, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline DuplicateImpl(__ubuf__ T* dst, const T scalarValue, uint64_t mask[2],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    SetVectorMask<uint16_t>(mask[1], mask[0]);
    __VEC_SCOPE__
    {
        RegTensor<T> vDst;
        MaskReg preg = movp_b16();
        for (uint16_t i = 0; i < repeatTime; ++i) {
            Duplicate(vDst, scalarValue, preg);
            DataCopy(dst, vDst, dstBlockStride, i * dstRepeatStride, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline DuplicateImpl(__ubuf__ T* dst, const T scalarValue, uint64_t mask[2],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    SetVectorMask<uint16_t>(mask[1], mask[0]);
    __VEC_SCOPE__
    {
        RegTensor<T> vDst;
        MaskReg preg = movp_b32();
        for (uint16_t i = 0; i < repeatTime; ++i) {
            Duplicate(vDst, scalarValue, preg);
            DataCopy(dst, vDst, dstBlockStride, i * dstRepeatStride, preg);
        }
    }
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
