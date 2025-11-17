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
 * \file kernel_operator_vec_cmpsel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H

#include "kernel_utils.h"
#include "kernel_struct_unary.h"

namespace AscendC {
/* ***************************************************************************************
 * ************************************** Compare ****************************************
 * ************************************************************************************** */
#define CONTINUOUS_MODE_B16_VCMPV_VF(cmpMode)                                                   \
    ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                        \
                    " vector calculate is not support");                                        \
        })

#define CONTINUOUS_MODE_B32_VCMPV_VF(cmpMode)                                                   \
    ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                        \
                    " vector calculate is not support");                                        \
        })


#define BITS_MODE_B16_VCMPV_VF(cmpMode)                                                         \
    ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                        \
                    " vector calculate is not support");                                        \
        })

#define BITS_MODE_B32_VCMPV_VF(cmpMode)                                                         \
    ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                        \
                    " vector calculate is not support");                                        \
        })

#define COUNTER_MODE_B16_VCMPV_VF(cmpMode)                                                      \
    ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                        \
                    " vector calculate is not support");                                        \
        })


#define COUNTER_MODE_B32_VCMPV_VF(cmpMode)                                                      \
    ASCENDC_ASSERT((false), {                                                                   \
                KERNEL_LOG(KERNEL_ERROR,                                                        \
                    " vector calculate is not support");                                        \
        })
// Compare::Level 0 - mask bit mode
template <typename T = half, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    CMPMODE cmpMode, const uint64_t mask[2], uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    switch (cmpMode) {
        case CMPMODE::LT: {
            BITS_MODE_B16_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            BITS_MODE_B16_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            BITS_MODE_B16_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            BITS_MODE_B16_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            BITS_MODE_B16_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            BITS_MODE_B16_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    CMPMODE cmpMode, const uint64_t mask[2], uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    switch (cmpMode) {
        case CMPMODE::LT: {
            BITS_MODE_B32_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            BITS_MODE_B32_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            BITS_MODE_B32_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            BITS_MODE_B32_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            BITS_MODE_B32_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            BITS_MODE_B32_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

// Compare::Level 0 - mask normaL mode
template <typename T = half, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B16_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B16_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B16_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
}

template <typename T = float, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B32_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B32_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B32_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
}

// Compare::Level 2
template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B16_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B16_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B16_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B16_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B16_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B16_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B32_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B32_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B32_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B32_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B32_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B32_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
}

// Compare written to CMPMASK
template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask[2], const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

/* ***************************************************************************************
 * *********************************** CompareScalar *************************************
 * ************************************************************************************** */
// CompareScalar::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

// CompareScalar::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

// CompareScalar::Level 2
template <typename T, typename U>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

/* ***************************************************************************************
 * *************************************** Select ****************************************
 * ************************************************************************************** */
// ============ select mode: 0/2 ============
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ T* sel, __ubuf__ half* src0, __ubuf__ half* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    uint32_t blockElm = ONE_BLOCK_SIZE / sizeof(half);
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ T* sel, __ubuf__ float* src0, __ubuf__ float* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    uint32_t blockElm = ONE_BLOCK_SIZE / sizeof(float);
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

// select mode: 1
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* selMask, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ T* sel, __ubuf__ half* src0, half src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ T* sel, __ubuf__ float* src0, float src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

// select mode: 0/2
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* selMask, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ T* sel, __ubuf__ half* src0, __ubuf__ half* src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ T* sel, __ubuf__ float* src0, __ubuf__ float* src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

// select mode: 1
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* selMask, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ T* sel, __ubuf__ half* src0, half src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ T* sel, __ubuf__ float* src0, float src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T, SELMODE selMode>
__aicore__ inline void SelectCal(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, int32_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
__aicore__ inline void SelectCal(
    __ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, int32_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    T src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ T* sel, __ubuf__ half* src0,
    half src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ T* sel, __ubuf__ float* src0,
    float src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    __ubuf__ T* src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ T* sel, __ubuf__ half* src0,
    __ubuf__ half* src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ T* sel, __ubuf__ float* src0,
    __ubuf__ float* src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T>
__aicore__ inline void GetCmpMaskImpl(__ubuf__ T* dst)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}

template <typename T>
__aicore__ inline void SetCmpMaskImpl(__ubuf__ T* src)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "vector calculate is not support"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H