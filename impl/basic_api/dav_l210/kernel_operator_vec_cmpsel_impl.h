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
 * \file kernel_operator_vec_cmpsel_impl.h
 * \brief AscendC l210 support vector compare and select memory base api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H

#include "kernel_utils.h"

namespace AscendC {
/* ***************************************************************************************
 * ************************************** Compare ****************************************
 * ************************************************************************************** */
#define COUNTER_MODE_B8_VCMPV_VF(cmpMode, vregType, scalarType)                                                         \
    uint32_t offsetBit32 = VECTOR_REG_WIDTH / B32_BIT_SIZE;                                           \
    __VEC_SCOPE__ {                                                                                \
        for (uint16_t i = 0; i <= get_vloopn_bound_b8(count); ++i) {                           \
            vector_##vregType vreg0;                                                                       \
            vector_##vregType vreg1;                                                                       \
            vector_bool preg1;                                                                     \
            vector_address srcOffset = vag_b8(VECTOR_REG_WIDTH);                  \
            vector_address dstOffset = vag_b32(offsetBit32);   \
            vector_bool preg = vpd_b8();                                                          \
            vld(vreg0, src0, srcOffset, NORM);                                                     \
            vld(vreg1, src1, srcOffset, NORM);                                                     \
            vcmp_##cmpMode(preg1, vreg0, vreg1, preg);                                             \
            pst(preg1, (__ubuf__ uint32_t *)dst, dstOffset, NORM);                                   \
        }                                                                                          \
    }

#define COUNTER_MODE_B16_VCMPV_VF(cmpMode, vregType, scalarType)                                                         \
    uint32_t offset16 = VECTOR_REG_WIDTH / B16_BYTE_SIZE;                                           \
    uint32_t offsetBit32 = VECTOR_REG_WIDTH / B16_BYTE_SIZE / B32_BIT_SIZE;                                           \
    __VEC_SCOPE__ {                                                                                \
        for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); ++i) {                           \
            vector_##vregType vreg0;                                                                      \
            vector_##vregType vreg1;                                                                      \
            vector_bool preg1;                                                                     \
            vector_address srcOffset = vag_b16(offset16);                  \
            vector_address dstOffset = vag_b32(offsetBit32);   \
            vector_bool preg = vpd_b16();                                                          \
            vld(vreg0, src0, srcOffset, NORM);                                                     \
            vld(vreg1, src1, srcOffset, NORM);                                                     \
            vcmp_##cmpMode(preg1, vreg0, vreg1, preg);                                             \
            pst(preg1, (__ubuf__ uint32_t *)dst, dstOffset, PK);                                   \
        }                                                                                          \
    }

#define COUNTER_MODE_B32_VCMPV_VF(cmpMode, vregType, scalarType)                                   \
    uint32_t repeatElm = VECTOR_REG_WIDTH / B32_BYTE_SIZE;                                         \
    uint16_t repeatTime = CeilDivision(count, repeatElm);                                      \
    uint16_t halfElm = (repeatTime / 2)* VECTOR_REG_WIDTH / B32_BYTE_SIZE;                        \
    if (halfElm > 0) {                                                                             \
        uint32_t offset32 = 2 * VECTOR_REG_WIDTH / B32_BYTE_SIZE;                                           \
        uint32_t offsetBit32 = VECTOR_REG_WIDTH * 2 / B32_BYTE_SIZE / B32_BIT_SIZE;                                           \
        __VEC_SCOPE__ {                                                                                \
            for (uint16_t i = 0; i <= get_vloopn_bound_b32(halfElm); ++i) { \
                vector_##vregType vreg00, vreg01;                                                             \
                vector_##vregType vreg10, vreg11;                                                             \
                vector_bool preg0, preg1, preg2, preg3;                                                \
                vector_address srcOffset = vag_b32(offset32);              \
                vector_address dstOffset = vag_b32(offsetBit32);   \
                vector_bool preg = vpd_b32();                                                          \
                vld(vreg00, src0, srcOffset, NORM);                                                    \
                vld(vreg10, src1, srcOffset, NORM);                                                    \
                vld(vreg01, src0 + VECTOR_REG_WIDTH / B32_BYTE_SIZE, srcOffset, NORM);                 \
                vld(vreg11, src1 + VECTOR_REG_WIDTH / B32_BYTE_SIZE, srcOffset, NORM);                 \
                vcmp_##cmpMode(preg0, vreg00, vreg10, preg);                                           \
                vcmp_##cmpMode(preg1, vreg01, vreg11, preg);                                           \
                pdintlv_b8(preg2, preg3, preg0, preg1);                                                \
                pst(preg2, (__ubuf__ uint32_t *)dst, dstOffset, PK);                                   \
            }                                                                                          \
        }                                                                                              \
    }                                                                                              \
    uint16_t tailElm = count - halfElm * 2;                                                     \
    if (tailElm > 0) {                                                                             \
        __ubuf__ scalarType* src0Tail = src0 + halfElm * 2;                                                 \
        __ubuf__ scalarType* src1Tail = src1 + halfElm * 2;                                                 \
        __ubuf__ uint32_t* dstTail = (__ubuf__ uint32_t *)dst + (repeatTime / 2) * VECTOR_REG_WIDTH * 2 / B32_BYTE_SIZE / B32_BIT_SIZE;            \
        __VEC_SCOPE__ {                                                                                \
            for (uint16_t i = 0; i <= get_vloopn_bound_b32(tailElm); ++i) {                            \
                vector_##vregType vreg0, vreg1;                                                               \
                vector_bool preg5, preg6;                                                              \
                vector_address srcTailOffset = vag_b32(0);                                             \
                vector_address dstTailOffset = vag_b32(0);                                             \
                vector_bool preg = vpd_b32();                                                          \
                vld(vreg0, src0Tail, srcTailOffset, NORM);                                             \
                vld(vreg1, src1Tail, srcTailOffset, NORM);                                             \
                vcmp_##cmpMode(preg5, vreg0, vreg1, preg);                                             \
                ppack(preg6, preg5, LOWER);                                                            \
                pst(preg6, dstTail, dstTailOffset, PK);                                                \
            }                                                                                          \
        }                                                                                              \
    }

#define COUNTER_MODE_B8_CMP(vregType, scalarType)                                             \
    switch (cmpMode) {                                                                         \
        case CMPMODE::LT: {                                                                   \
            COUNTER_MODE_B8_VCMPV_VF(lt, vregType, scalarType);                               \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::GT: {                                                                   \
            COUNTER_MODE_B8_VCMPV_VF(gt, vregType, scalarType);                               \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::EQ: {                                                                   \
            COUNTER_MODE_B8_VCMPV_VF(eq, vregType, scalarType);                               \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::LE: {                                                                   \
            COUNTER_MODE_B8_VCMPV_VF(le, vregType, scalarType);                               \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::GE: {                                                                   \
            COUNTER_MODE_B8_VCMPV_VF(ge, vregType, scalarType);                               \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::NE: {                                                                   \
            COUNTER_MODE_B8_VCMPV_VF(ne, vregType, scalarType);                               \
            break;                                                                            \
        }                                                                                     \
        default:                                                                              \
            break;                                                                            \
    }                                                                                         \

#define COUNTER_MODE_B16_CMP(vregType, scalarType)                                            \
    switch (cmpMode) {                                                                         \
        case CMPMODE::LT: {                                                                   \
            COUNTER_MODE_B16_VCMPV_VF(lt, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::GT: {                                                                   \
            COUNTER_MODE_B16_VCMPV_VF(gt, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::EQ: {                                                                   \
            COUNTER_MODE_B16_VCMPV_VF(eq, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::LE: {                                                                   \
            COUNTER_MODE_B16_VCMPV_VF(le, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::GE: {                                                                   \
            COUNTER_MODE_B16_VCMPV_VF(ge, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::NE: {                                                                   \
            COUNTER_MODE_B16_VCMPV_VF(ne, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        default:                                                                              \
            break;                                                                            \
    }                                                                                         \

#define COUNTER_MODE_B32_CMP(vregType, scalarType)                                             \
    switch (cmpMode) {                                                                         \
        case CMPMODE::LT: {                                                                   \
            COUNTER_MODE_B32_VCMPV_VF(lt, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::GT: {                                                                   \
            COUNTER_MODE_B32_VCMPV_VF(gt, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::EQ: {                                                                   \
            COUNTER_MODE_B32_VCMPV_VF(eq, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::LE: {                                                                   \
            COUNTER_MODE_B32_VCMPV_VF(le, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::GE: {                                                                   \
            COUNTER_MODE_B32_VCMPV_VF(ge, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        case CMPMODE::NE: {                                                                   \
            COUNTER_MODE_B32_VCMPV_VF(ne, vregType, scalarType);                              \
            break;                                                                            \
        }                                                                                     \
        default:                                                                              \
            break;                                                                            \
    }                                                                                         \

// Compare::Level 2
template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ uint8_t* src0, __ubuf__ uint8_t* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B8_CMP(u8, uint8_t);
}

template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ int8_t* src0, __ubuf__ int8_t* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B8_CMP(s8, int8_t);
}

template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B16_CMP(f16, half);
}

template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B16_CMP(s16, int16_t);
}

template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B16_CMP(u16, uint16_t);
}

template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B32_CMP(f32, float);
}

template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B32_CMP(s32, int32_t);
}

template <typename T>
__aicore__ inline void VcmpvImpl(__ubuf__ T* dst, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B32_CMP(u32, uint32_t);
}

/* ***************************************************************************************
 * *********************************** CompareScalar *************************************
 * ************************************************************************************** */
#define COUNTER_MODE_B8_VCMPVS_VF(cmpMode, vregType, scalarType)                                   \
    uint32_t offsetBit32 = VECTOR_REG_WIDTH / B32_BIT_SIZE;                                           \
    __VEC_SCOPE__ {                                                                                \
        for (uint16_t i = 0; i <= get_vloopn_bound_b8(count); ++i) {                            \
            vector_##vregType vreg0;                                                                       \
            vector_bool preg1;                                                                     \
            vector_address srcOffset = vag_b8(VECTOR_REG_WIDTH);                                   \
            vector_address dstOffset = vag_b32(offsetBit32);                   \
            vector_bool preg = vpd_b8();                                                          \
            vld(vreg0, src0, srcOffset, NORM);                                                     \
            vcmps_##cmpMode(preg1, vreg0, (scalarType)scalar, preg);                               \
            pst(preg1, (__ubuf__ uint32_t *)dst, dstOffset, NORM);                                   \
        }                                                                                          \
    }

#define COUNTER_MODE_B16_VCMPVS_VF(cmpMode, vregType, scalarType)                                  \
    uint32_t offset16 = VECTOR_REG_WIDTH / B16_BYTE_SIZE;                        \
    uint32_t offsetBit32 = VECTOR_REG_WIDTH / B16_BYTE_SIZE / B32_BIT_SIZE;                        \
    __VEC_SCOPE__ {                                                                                \
        for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); ++i) {                           \
            vector_##vregType vreg0;                                                               \
            vector_bool preg1;                                                                     \
            vector_address srcOffset = vag_b16(offset16);                  \
            vector_address dstOffset = vag_b32(offsetBit32);   \
            vector_bool preg = vpd_b16();                                                          \
            vld(vreg0, src0, srcOffset, NORM);                                                     \
            vcmps_##cmpMode(preg1, vreg0, (scalarType)scalar, preg);                               \
            pst(preg1, (__ubuf__ uint32_t *)dst, dstOffset, PK);                                   \
        }                                                                                          \
    }

#define COUNTER_MODE_B32_VCMPVS_VF(cmpMode, vregType, scalarType)                                  \
    uint32_t repeatElm = VECTOR_REG_WIDTH / B32_BYTE_SIZE;                                         \
    uint16_t repeatTime = CeilDivision(count, repeatElm);                                      \
    uint16_t halfElm = (repeatTime / 2)* VECTOR_REG_WIDTH / B32_BYTE_SIZE;                        \
    if (halfElm > 0) {                                                                               \
        uint32_t offset32 = 2 * VECTOR_REG_WIDTH / B32_BYTE_SIZE;                        \
        uint32_t offsetBit32 = VECTOR_REG_WIDTH * 2 / B32_BYTE_SIZE / B32_BIT_SIZE;                        \
        __VEC_SCOPE__ {                                                                                \
            for (uint16_t i = 0; i <= get_vloopn_bound_b32(halfElm); ++i) { \
                vector_##vregType vreg00, vreg01;                                                      \
                vector_bool preg0, preg1, preg2, preg3;                                                \
                vector_address srcOffset = vag_b32(offset32);              \
                vector_address dstOffset = vag_b32(offsetBit32);   \
                vector_bool preg = vpd_b32();                                                          \
                vld(vreg00, src0, srcOffset, NORM);                                                    \
                vld(vreg01, src0 + VECTOR_REG_WIDTH / B32_BYTE_SIZE, srcOffset, NORM);                 \
                vcmps_##cmpMode(preg0, vreg00, (scalarType)scalar, preg);                                    \
                vcmps_##cmpMode(preg1, vreg01, (scalarType)scalar, preg);                                    \
                pdintlv_b8(preg2, preg3, preg0, preg1);                                                \
                pst(preg2, (__ubuf__ uint32_t *)dst, dstOffset, PK);                                   \
            }                                                                                          \
        }                                                                                              \
    }                                                                                              \
    uint16_t tailElm = count - halfElm * 2;                                                     \
    if (tailElm > 0) {                                                                             \
        __ubuf__ scalarType* src0Tail = src0 + halfElm * 2;                                             \
        __ubuf__ uint32_t* dstTail = (__ubuf__ uint32_t *)dst + (repeatTime / 2) * VECTOR_REG_WIDTH * 2 / B32_BYTE_SIZE / B32_BIT_SIZE;            \
        __VEC_SCOPE__ {                                                                            \
            for (uint16_t i = 0; i <= get_vloopn_bound_b32(tailElm); ++i) {                        \
                vector_##vregType vreg0;                                                                  \
                vector_bool preg5, preg6;                                                          \
                vector_address srcTailOffset = vag_b32(0);                                         \
                vector_address dstTailOffset = vag_b32(0);                                         \
                vector_bool preg = vpd_b32();                                                      \
                vld(vreg0, src0Tail, srcTailOffset, NORM);                                         \
                vcmps_##cmpMode(preg5, vreg0, (scalarType)scalar, preg);                                \
                ppack(preg6, preg5, LOWER);                                                        \
                pst(preg6, dstTail, dstTailOffset, PK);                                            \
            }                                                                                      \
        }                                                                                          \
    }

#define COUNTER_MODE_B8_CMPS(vregType, scalarType)                                  \
    switch (cmpMode) {                                                              \
        case CMPMODE::LT: {                                                         \
            COUNTER_MODE_B8_VCMPVS_VF(lt, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::GT: {                                                         \
            COUNTER_MODE_B8_VCMPVS_VF(gt, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::EQ: {                                                         \
            COUNTER_MODE_B8_VCMPVS_VF(eq, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::LE: {                                                         \
            COUNTER_MODE_B8_VCMPVS_VF(le, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::GE: {                                                         \
            COUNTER_MODE_B8_VCMPVS_VF(ge, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::NE: {                                                         \
            COUNTER_MODE_B8_VCMPVS_VF(ne, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        default:                                                                    \
            break;                                                                  \
    }

#define COUNTER_MODE_B16_CMPS(vregType, scalarType)                                  \
    switch (cmpMode) {                                                              \
        case CMPMODE::LT: {                                                         \
            COUNTER_MODE_B16_VCMPVS_VF(lt, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::GT: {                                                         \
            COUNTER_MODE_B16_VCMPVS_VF(gt, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::EQ: {                                                         \
            COUNTER_MODE_B16_VCMPVS_VF(eq, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::LE: {                                                         \
            COUNTER_MODE_B16_VCMPVS_VF(le, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::GE: {                                                         \
            COUNTER_MODE_B16_VCMPVS_VF(ge, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::NE: {                                                         \
            COUNTER_MODE_B16_VCMPVS_VF(ne, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        default:                                                                    \
            break;                                                                  \
    }

#define COUNTER_MODE_B32_CMPS(vregType, scalarType)                                  \
    switch (cmpMode) {                                                              \
        case CMPMODE::LT: {                                                         \
            COUNTER_MODE_B32_VCMPVS_VF(lt, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::GT: {                                                         \
            COUNTER_MODE_B32_VCMPVS_VF(gt, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::EQ: {                                                         \
            COUNTER_MODE_B32_VCMPVS_VF(eq, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::LE: {                                                         \
            COUNTER_MODE_B32_VCMPVS_VF(le, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::GE: {                                                         \
            COUNTER_MODE_B32_VCMPVS_VF(ge, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        case CMPMODE::NE: {                                                         \
            COUNTER_MODE_B32_VCMPVS_VF(ne, vregType, scalarType);                    \
            break;                                                                  \
        }                                                                           \
        default:                                                                    \
            break;                                                                  \
    }

// CompareScalar::Level 2
template <typename T, typename U>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VcmpvsImpl(__ubuf__ T* dst, __ubuf__ uint8_t* src0, uint8_t scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B8_CMPS(u8, uint8_t);
}

template <typename T>
__aicore__ inline void VcmpvsImpl(__ubuf__ T* dst, __ubuf__ int8_t* src0, int8_t scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B8_CMPS(s8, int8_t);
}

template <typename T>
__aicore__ inline void VcmpvsImpl(__ubuf__ T* dst, __ubuf__ half* src0, half scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B16_CMPS(f16, half);
}

template <typename T>
__aicore__ inline void VcmpvsImpl(__ubuf__ T* dst, __ubuf__ int16_t* src0, int16_t scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B16_CMPS(s16, int16_t);
}

template <typename T>
__aicore__ inline void VcmpvsImpl(__ubuf__ T* dst, __ubuf__ uint16_t* src0, uint16_t scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B16_CMPS(u16, uint16_t);
}

template <typename T>
__aicore__ inline void VcmpvsImpl(__ubuf__ T* dst, __ubuf__ float* src0, float scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B32_CMPS(f32, float);
}

template <typename T>
__aicore__ inline void VcmpvsImpl(__ubuf__ T* dst, __ubuf__ int32_t* src0, int32_t scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B32_CMPS(s32, int32_t);
}

template <typename T>
__aicore__ inline void VcmpvsImpl(__ubuf__ T* dst, __ubuf__ uint32_t* src0, uint32_t scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    COUNTER_MODE_B32_CMPS(u32, uint32_t);
}

/* ***************************************************************************************
 * *************************************** Select ****************************************
 * ************************************************************************************** */
// select mode: 1
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    T src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ int8_t* dst, __ubuf__ T* sel, __ubuf__ int8_t* src0,
    int8_t src1, SELMODE selMode, uint32_t count)
{
    uint32_t offsetBit32 = VECTOR_REG_WIDTH / B32_BIT_SIZE;
    __VEC_SCOPE__ {
        for (uint16_t j = 0; j < 1; ++j) {
            vector_s8 vreg1;
            vbr(vreg1, src1);
            for (uint16_t i = 0; i <= get_vloopn_bound_b8(count); ++i) {
                vector_s8 vreg0;
                vector_s8 vreg2;
                vector_bool preg0 = vpd_b8();
                vector_bool preg1;
                vector_address offset0 = vag_b32(offsetBit32);
                vector_address offset1 = vag_b8(VECTOR_REG_WIDTH);
                pld(preg1, (__ubuf__ uint32_t *)sel, offset0, NORM);
                vld(vreg0, src0, offset1, NORM);
                vsel(vreg2, vreg0, vreg1, preg1);
                vst(vreg2, dst, offset1, NORM_B8, preg0);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ uint8_t* dst, __ubuf__ T* sel, __ubuf__ uint8_t* src0,
    uint8_t src1, SELMODE selMode, uint32_t count)
{
    VselImpl((__ubuf__ int8_t*) dst, sel, (__ubuf__ int8_t*) src0,
    (int8_t)src1, selMode, count);
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ T* sel, __ubuf__ half* src0,
    half src1, SELMODE selMode, uint32_t count)
{
    uint32_t offset16 = VECTOR_REG_WIDTH / B16_BYTE_SIZE;
    uint32_t offsetBit32 = VECTOR_REG_WIDTH / B16_BYTE_SIZE / B32_BIT_SIZE;
    __VEC_SCOPE__ {
        for (uint16_t j = 0; j < 1; ++j) {
            vector_f16 vreg1;
            vbr(vreg1, src1);
            for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); ++i) {
                vector_f16 vreg0;
                vector_f16 vreg2;
                vector_bool preg0 = vpd_b16();
                vector_bool preg1;
                vector_address offset0 = vag_b32(offsetBit32);
                vector_address offset1 = vag_b16(offset16);
                pld(preg1, (__ubuf__ uint32_t *)sel, offset0, US);
                vld(vreg0, src0, offset1, NORM);
                vsel(vreg2, vreg0, vreg1, preg1);
                vst(vreg2, dst, offset1, NORM_B16, preg0);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ int16_t* dst, __ubuf__ T* sel, __ubuf__ int16_t* src0,
    int16_t src1, SELMODE selMode, uint32_t count)
{
    half scalar = *(half*)(&src1);
    VselImpl((__ubuf__ half*) dst, sel, (__ubuf__ half*) src0, scalar, selMode, count);
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ uint16_t* dst, __ubuf__ T* sel, __ubuf__ uint16_t* src0,
    uint16_t src1, SELMODE selMode, uint32_t count)
{
    half scalar = *(half*)(&src1);
    VselImpl((__ubuf__ half*) dst, sel, (__ubuf__ half*) src0, scalar, selMode, count);
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ T* sel, __ubuf__ float* src0,
    float src1, SELMODE selMode, uint32_t count)
{
    uint32_t repeatElm = VECTOR_REG_WIDTH / B32_BYTE_SIZE;
    uint16_t repeatTime = CeilDivision(count, repeatElm);
    uint16_t halfElm = (repeatTime / 2)* VECTOR_REG_WIDTH / B32_BYTE_SIZE;
    if (halfElm > 0) {
        uint32_t offset32 = 2 * VECTOR_REG_WIDTH / B32_BYTE_SIZE;
        uint32_t offsetBit32 = VECTOR_REG_WIDTH / B16_BYTE_SIZE / B32_BIT_SIZE;
        __VEC_SCOPE__ {
            for (uint16_t j = 0; j < 1; ++j) {
                vector_f32 vreg1;
                vector_bool preg2 = pge_b8(PAT_ALLF);
                vbr(vreg1, src1);
                for (uint16_t i = 0; i <= get_vloopn_bound_b32(halfElm); ++i) {
                    vector_bool preg0 = vpd_b32();
                    vector_bool preg1;
                    vector_bool preg3;
                    vector_bool preg4;
                    vector_f32 vreg0;
                    vector_f32 vreg2;
                    vector_f32 vreg3;
                    vector_f32 vreg4;
                    vector_address pregOffset = vag_b32(offsetBit32);
                    vector_address offset = vag_b32(offset32);
                    pld(preg1, (__ubuf__ uint32_t *)sel, pregOffset, US);
                    vld(vreg0, src0, offset, NORM);
                    vld(vreg3, src0 + VECTOR_REG_WIDTH / B32_BYTE_SIZE, offset, NORM);
                    pintlv_b16(preg3, preg4, preg1, preg2);
                    vsel(vreg2, vreg0, vreg1, preg3);
                    vsel(vreg4, vreg3, vreg1, preg4);
                    vst(vreg2, dst, offset, NORM_B32, preg0);
                    vst(vreg4, dst + VECTOR_REG_WIDTH / B32_BYTE_SIZE, offset, NORM_B32, preg0);
                }
            }
        }
    }

    uint16_t tailElm = count - halfElm * 2;
    if (tailElm > 0) {
        __ubuf__ float* src0Tail = src0 + halfElm * 2;
        __ubuf__ uint32_t* selTail = (__ubuf__ uint32_t *)sel + (repeatTime / 2) * VECTOR_REG_WIDTH / B16_BYTE_SIZE / B32_BIT_SIZE;
        __ubuf__ float* dstTail = dst + halfElm * 2;
        __VEC_SCOPE__ {
            for (uint16_t j = 0; j < 1; ++j) {
                vector_f32 vreg1;
                vbr(vreg1, src1);
                for (uint16_t i = 0; i <= get_vloopn_bound_b32(tailElm); ++i) {
                    vector_f32 vreg5;
                    vector_f32 vreg6;
                    vector_bool preg5;
                    vector_bool preg6;
                    vector_address offset = vag_b32(0);
                    vector_bool preg0 = vpd_b32();
                    pld(preg5, (__ubuf__ uint32_t *)selTail, offset, US);
                    vld(vreg5, src0Tail, offset, NORM);
                    punpack(preg6, preg5, LOWER);
                    vsel(vreg6, vreg5, vreg1, preg6);
                    vst(vreg6, dstTail, offset, NORM_B32, preg0);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ int32_t* dst, __ubuf__ T* sel, __ubuf__ int32_t* src0,
    int32_t src1, SELMODE selMode, uint32_t count)
{
    float scalar = *(float*)(&src1);
    VselImpl((__ubuf__ float*) dst, sel, (__ubuf__ float*) src0, scalar, selMode, count);
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ uint32_t* dst, __ubuf__ T* sel, __ubuf__ uint32_t* src0,
    uint32_t src1, SELMODE selMode, uint32_t count)
{
    float scalar = *(float*)(&src1);
    VselImpl((__ubuf__ float*) dst, sel, (__ubuf__ float*) src0, scalar, selMode, count);
}

// select mode: 0/2
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    __ubuf__ T* src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ int8_t* dst, __ubuf__ T* sel, __ubuf__ int8_t* src0,
    __ubuf__ int8_t* src1, SELMODE selMode, uint32_t count)
{
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        uint16_t repeatTime = CeilDivision(count, VECTOR_REG_WIDTH);
        uint16_t halfElm = (repeatTime / 2)* VECTOR_REG_WIDTH;
        __ubuf__ uint32_t* selTail = (__ubuf__ uint32_t *)sel + VECTOR_REG_WIDTH / B32_BIT_SIZE;
        if (halfElm > 0) {
            uint32_t offset8 = 2 * VECTOR_REG_WIDTH;
            __VEC_SCOPE__ {
                for (uint16_t j = 0; j < 1; ++j) {
                    vector_bool preg0, preg1;
                    vector_address offset0 = vag_b8(0);
                    pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, NORM);
                    pld(preg1, ((__ubuf__ uint32_t *)selTail), offset0, NORM);
                    for (uint16_t i = 0; i <= get_vloopn_bound_b8(halfElm); ++i) {
                        vector_s8 vreg00, vreg01;
                        vector_s8 vreg10, vreg11;
                        vector_s8 vreg20, vreg21;
                        vector_bool preg2 = vpd_b8();
                        vector_address offset1 = vag_b8(offset8);
                        vld(vreg00, src0, offset1, NORM);
                        vld(vreg01, src0 + VECTOR_REG_WIDTH, offset1, NORM);
                        vld(vreg10, src1, offset1, NORM);
                        vld(vreg11, src1 + VECTOR_REG_WIDTH, offset1, NORM);
                        vsel(vreg20, vreg00, vreg10, preg0);
                        vsel(vreg21, vreg01, vreg11, preg1);
                        vst(vreg20, dst, offset1, NORM_B8, preg2);
                        vst(vreg21, dst + VECTOR_REG_WIDTH, offset1, NORM_B8, preg2);
                    }
                }
            }
        }
        uint16_t tailElm = count - halfElm * 2;
        if (tailElm > 0) {
            __ubuf__ int8_t* src0Tail = src0 + halfElm * 2;
            __ubuf__ int8_t* src1Tail = src1 + halfElm * 2;
            __ubuf__ int8_t* dstTail = dst + halfElm * 2;
            __VEC_SCOPE__ {
                for (uint16_t i = 0; i <= get_vloopn_bound_b8(tailElm); ++i) {
                    vector_s8 vreg0;
                    vector_s8 vreg1;
                    vector_s8 vreg2;
                    vector_bool preg0;
                    vector_address offset0 = vag_b16(0);
                    vector_bool preg2 = vpd_b8();
                    vector_address offset1 = vag_b8(0);
                    pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, NORM);
                    vld(vreg0, src0Tail, offset1, NORM);
                    vld(vreg1, src1Tail, offset1, NORM);
                    vsel(vreg2, vreg0, vreg1, preg0);
                    vst(vreg2, dstTail, offset1, NORM_B8, preg2);
                }
            }
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        uint32_t offsetBit32 = VECTOR_REG_WIDTH / B32_BIT_SIZE;
        __VEC_SCOPE__ {
            for (uint16_t i = 0; i <= get_vloopn_bound_b8(count); ++i) {
                vector_s8 vreg0;
                vector_s8 vreg1;
                vector_s8 vreg2;
                vector_bool preg0;
                vector_bool preg2 = vpd_b8();
                vector_address offset0 = vag_b32(offsetBit32);
                vector_address offset1 = vag_b8(VECTOR_REG_WIDTH);
                pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, NORM);
                vld(vreg0, src0, offset1, NORM);
                vld(vreg1, src1, offset1, NORM);
                vsel(vreg2, vreg0, vreg1, preg0);
                vst(vreg2, dst, offset1, NORM_B8, preg2);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ uint8_t* dst, __ubuf__ T* sel, __ubuf__ uint8_t* src0,
    __ubuf__ uint8_t* src1, SELMODE selMode, uint32_t count)
{
    VselImpl((__ubuf__ int8_t*)dst, sel, (__ubuf__ int8_t*)src0,
    (__ubuf__ int8_t*) src1, selMode, count);
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ T* sel, __ubuf__ half* src0,
    __ubuf__ half* src1, SELMODE selMode, uint32_t count)
{
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        uint32_t repeatElm = VECTOR_REG_WIDTH / B16_BYTE_SIZE;
        uint16_t repeatTime = CeilDivision(count, repeatElm);
        uint16_t halfElm = (repeatTime / 2)* VECTOR_REG_WIDTH / B16_BYTE_SIZE;
        __ubuf__ uint32_t* selHigh = (__ubuf__ uint32_t *)sel + VECTOR_REG_WIDTH / B16_BYTE_SIZE / B32_BIT_SIZE;
        if (halfElm > 0) {
            uint32_t offset16 = 2 * VECTOR_REG_WIDTH / B16_BYTE_SIZE;
            __VEC_SCOPE__ {
                for (uint16_t j = 0; j < 1; ++j) {
                    vector_bool preg0, preg1;
                    vector_address offset0 = vag_b16(0);
                    pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, US);
                    pld(preg1, ((__ubuf__ uint32_t *)selHigh), offset0, US);
                    for (uint16_t i = 0; i <= get_vloopn_bound_b16(halfElm); ++i) {
                        vector_f16 vreg00, vreg01;
                        vector_f16 vreg10, vreg11;
                        vector_f16 vreg20, vreg21;
                        vector_bool preg2 = vpd_b16();
                        vector_address offset1 = vag_b16(offset16);
                        vld(vreg00, src0, offset1, NORM);
                        vld(vreg01, src0 + VECTOR_REG_WIDTH / B16_BYTE_SIZE, offset1, NORM);
                        vld(vreg10, src1, offset1, NORM);
                        vld(vreg11, src1 + VECTOR_REG_WIDTH / B16_BYTE_SIZE, offset1, NORM);
                        vsel(vreg20, vreg00, vreg10, preg0);
                        vsel(vreg21, vreg01, vreg11, preg1);
                        vst(vreg20, dst, offset1, NORM_B16, preg2);
                        vst(vreg21, dst + VECTOR_REG_WIDTH / B16_BYTE_SIZE, offset1, NORM_B16, preg2);
                    }
                }
            }
        }
        uint16_t tailElm = count - halfElm * 2;
        if (tailElm > 0) {
            __ubuf__ half* src0Tail = src0 + halfElm * 2;
            __ubuf__ half* src1Tail = src1 + halfElm * 2;
            __ubuf__ half* dstTail = dst + halfElm * 2;
            __VEC_SCOPE__ {
                for (uint16_t j = 0; j < 1; ++j) {
                    vector_bool preg0;
                    vector_address offset0 = vag_b16(0);
                    pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, US);
                    for (uint16_t i = 0; i <= get_vloopn_bound_b16(tailElm); ++i) {
                        vector_f16 vreg0;
                        vector_f16 vreg1;
                        vector_f16 vreg2;
                        vector_bool preg2 = vpd_b16();
                        vector_address offset1 = vag_b16(0);
                        vld(vreg0, src0Tail, offset1, NORM);
                        vld(vreg1, src1Tail, offset1, NORM);
                        vsel(vreg2, vreg0, vreg1, preg0);
                        vst(vreg2, dstTail, offset1, NORM_B16, preg2);
                    }
                }
            }
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        uint32_t offset16 = VECTOR_REG_WIDTH / B16_BYTE_SIZE;
        uint32_t offsetBit32 = VECTOR_REG_WIDTH / B16_BYTE_SIZE / B32_BIT_SIZE;
        __VEC_SCOPE__ {
            for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); ++i) {
                vector_f16 vreg0;
                vector_f16 vreg1;
                vector_f16 vreg2;
                vector_bool preg0;
                vector_bool preg2 = vpd_b16();
                vector_address offset0 = vag_b32(offsetBit32);
                vector_address offset1 = vag_b16(offset16);
                pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, US);
                vld(vreg0, src0, offset1, NORM);
                vld(vreg1, src1, offset1, NORM);
                vsel(vreg2, vreg0, vreg1, preg0);
                vst(vreg2, dst, offset1, NORM_B16, preg2);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ uint16_t* dst, __ubuf__ T* sel, __ubuf__ uint16_t* src0,
    __ubuf__ uint16_t* src1, SELMODE selMode, uint32_t count)
{
    VselImpl((__ubuf__ half*)dst, sel, (__ubuf__ half*)src0,
    (__ubuf__ half*)src1, selMode, count);
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ int16_t* dst, __ubuf__ T* sel, __ubuf__ int16_t* src0,
    __ubuf__ int16_t* src1, SELMODE selMode, uint32_t count)
{
    VselImpl((__ubuf__ half*)dst, sel, (__ubuf__ half*)src0,
    (__ubuf__ half*)src1, selMode, count);
}

template <typename T>
__aicore__ inline void VselSprImpl(__ubuf__ float *dst, __ubuf__ T *sel, __ubuf__ float *src0,
                                   __ubuf__ float *src1, SELMODE selMode, uint32_t count)
{
    uint32_t repeatElm = VECTOR_REG_WIDTH / B32_BYTE_SIZE;
    uint16_t repeatTime = CeilDivision(count, repeatElm);
    uint16_t halfElm = (repeatTime / 2)* VECTOR_REG_WIDTH / B32_BYTE_SIZE;
    uint16_t tailElm = count - halfElm * 2;
    __ubuf__ float* src0Tail = src0 + halfElm * 2;
    __ubuf__ float* src1Tail = src1 + halfElm * 2;
    __ubuf__ float* dstTail = dst + halfElm * 2;

    if (halfElm > 0)
    {
        uint32_t offset32 = 2 * VECTOR_REG_WIDTH / B32_BYTE_SIZE;
        __VEC_SCOPE__
        {
            for (uint16_t j = 0; j < 1; j++)
            {
                vector_bool preg0, preg2, preg3;
                vector_address offset0 = vag_b32(0);
                vector_bool preg1 = pge_b8(PAT_ALLF);
                pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, US);
                for (uint16_t i = 0; i <= get_vloopn_bound_b32(halfElm); ++i)
                {
                    vector_f32 vreg00, vreg01;
                    vector_f32 vreg10, vreg11;
                    vector_f32 vreg20, vreg21;
                    vector_bool preg4 = vpd_b32();
                    vector_address offset1 = vag_b32(offset32);
                    vld(vreg00, src0, offset1, NORM);
                    vld(vreg01, src0 + VECTOR_REG_WIDTH / B32_BYTE_SIZE, offset1, NORM);
                    vld(vreg10, src1, offset1, NORM);
                    vld(vreg11, src1 + VECTOR_REG_WIDTH / B32_BYTE_SIZE, offset1, NORM);
                    pintlv_b16(preg2, preg3, preg0, preg1);
                    vsel(vreg20, vreg00, vreg10, preg2);
                    vsel(vreg21, vreg01, vreg11, preg3);
                    vst(vreg20, dst, offset1, NORM_B32, preg4);
                    vst(vreg21, dst + VECTOR_REG_WIDTH / B32_BYTE_SIZE, offset1, NORM_B32, preg4);
                }
            }
        }
    }

    if (tailElm > 0)
    {
        __VEC_SCOPE__
        {
            for (uint16_t j = 0; j < 1; j++)
            {
                vector_bool preg0, preg1;
                vector_address offset0 = vag_b32(0);
                pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, US);
                for (uint16_t i = 0; i <= get_vloopn_bound_b32(tailElm); ++i)
                {
                    vector_f32 vreg0;
                    vector_f32 vreg1;
                    vector_f32 vreg2;
                    vector_bool preg3 = vpd_b32();
                    vector_address offset1 = vag_b32(0);
                    vld(vreg0, src0Tail, offset1, NORM);
                    vld(vreg1, src1Tail, offset1, NORM);
                    punpack(preg1, preg0, LOWER);
                    vsel(vreg2, vreg0, vreg1, preg1);
                    vst(vreg2, dstTail, offset1, NORM_B32, preg3);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void VselTensorImpl(__ubuf__ float *dst, __ubuf__ T *sel, __ubuf__ float *src0,
                                   __ubuf__ float *src1, SELMODE selMode, uint32_t count)
{
    uint32_t repeatElm = VECTOR_REG_WIDTH / B32_BYTE_SIZE;
    uint16_t repeatTime = CeilDivision(count, repeatElm);
    uint16_t halfElm = (repeatTime / 2)* VECTOR_REG_WIDTH / B32_BYTE_SIZE;
    uint16_t tailElm = count - halfElm * 2;
    __ubuf__ float* src0Tail = src0 + halfElm * 2;
    __ubuf__ float* src1Tail = src1 + halfElm * 2;
    __ubuf__ float* dstTail = dst + halfElm * 2;
    __ubuf__ uint32_t* selTail= (__ubuf__ uint32_t *)sel + (repeatTime / 2) * VECTOR_REG_WIDTH / B16_BYTE_SIZE / B32_BIT_SIZE;
    if (halfElm > 0)
    {
        uint32_t offset32 = 2 * VECTOR_REG_WIDTH / B32_BYTE_SIZE;
        uint32_t offsetBit32 = 2 * VECTOR_REG_WIDTH / B32_BYTE_SIZE / B32_BIT_SIZE;
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i <= get_vloopn_bound_b32(halfElm); ++i)
            {
                vector_f32 vreg00, vreg01;
                vector_f32 vreg10, vreg11;
                vector_f32 vreg20, vreg21;
                vector_bool preg0, preg2, preg3;
                vector_bool preg1 = pge_b8(PAT_ALLF);
                vector_bool preg4 = vpd_b32();
                vector_address offset0 = vag_b32(offsetBit32);
                vector_address offset1 = vag_b32(offset32);
                pld(preg0, ((__ubuf__ uint32_t *)sel), offset0, US);
                vld(vreg00, src0, offset1, NORM);
                vld(vreg01, src0 + VECTOR_REG_WIDTH / B32_BYTE_SIZE, offset1, NORM);
                vld(vreg10, src1, offset1, NORM);
                vld(vreg11, src1 + VECTOR_REG_WIDTH / B32_BYTE_SIZE, offset1, NORM);
                pintlv_b16(preg2, preg3, preg0, preg1);
                vsel(vreg20, vreg00, vreg10, preg2);
                vsel(vreg21, vreg01, vreg11, preg3);
                vst(vreg20, dst, offset1, NORM_B32, preg4);
                vst(vreg21, dst + VECTOR_REG_WIDTH / B32_BYTE_SIZE, offset1, NORM_B32, preg4);
            }
        }
    }

    if (tailElm > 0)
    {
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i <= get_vloopn_bound_b32(tailElm); ++i)
            {
                vector_f32 vreg0;
                vector_f32 vreg1;
                vector_f32 vreg2;
                vector_bool preg0, preg1;
                vector_bool preg3 = vpd_b32();
                vector_address offset = vag_b32(0);
                pld(preg0, ((__ubuf__ uint32_t *)selTail), offset, US);
                vld(vreg0, src0Tail, offset, NORM);
                vld(vreg1, src1Tail, offset, NORM);
                punpack(preg1, preg0, LOWER);
                vsel(vreg2, vreg0, vreg1, preg1);
                vst(vreg2, dstTail, offset, NORM_B32, preg3);
            }
        }
    }
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ T* sel, __ubuf__ float* src0,
    __ubuf__ float* src1, SELMODE selMode, uint32_t count)
{
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        VselSprImpl(dst, sel, src0, src1, selMode, count);
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        VselTensorImpl(dst, sel, src0, src1, selMode, count);
    }
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ uint32_t* dst, __ubuf__ T* sel, __ubuf__ uint32_t* src0,
    __ubuf__ uint32_t* src1, SELMODE selMode, uint32_t count)
{
    VselImpl((__ubuf__ float*)dst, sel, (__ubuf__ float*)src0,
    (__ubuf__ float*)src1, selMode, count);
}

template <typename T>
__aicore__ inline void VselImpl(__ubuf__ int32_t* dst, __ubuf__ T* sel, __ubuf__ int32_t* src0,
    __ubuf__ int32_t* src1, SELMODE selMode, uint32_t count)
{
    VselImpl((__ubuf__ float*)dst, sel, (__ubuf__ float*)src0,
    (__ubuf__ float*)src1, selMode, count);
}

template <typename T, SELMODE selMode>
__aicore__ inline void SelectCal(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, int32_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SelectCal is not supported!"); });
}

template <typename T, typename U>
__aicore__ inline void SelectCal(
    __ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, int32_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SelectCal is not supported!"); });
}

template <typename T>
__aicore__ inline void GetCmpMaskImpl(__ubuf__ T* dst)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetCmpMask is not supported!"); });
}

template <typename T>
__aicore__ inline void SetCmpMaskImpl(__ubuf__ T* src)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetCmpMask is not supported!"); });
}


} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H