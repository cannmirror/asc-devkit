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
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H

#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_struct_unary.h"
namespace AscendC {
/* ***************************************************************************************
 * ************************************** Compare ****************************************
 * ************************************************************************************** */
#define COUNTER_MODE_B8_VCMPV_VF(cmpMode)                                                         \
    __VEC_SCOPE__                                                                                 \
    {                                                                                             \
        RegTensor<T> vSrc0;                                                                       \
        RegTensor<T> vSrc1;                                                                       \
        uint32_t sreg = (uint32_t)count;                                                       \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                            \
        uint16_t repeatTime = CeilDivision(count, sregLower);                                 \
        MaskReg preg;                                                                             \
        MaskReg dstReg;                                                                           \
        AddrReg dstOffset;                                                                        \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                              \
            preg = CreatePredicate<T>(sreg);                                                      \
            dstReg = CreatePredicate<T>();                                                        \
            dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);              \
            DataCopy(vSrc0, src0, i * sregLower);                                                 \
            DataCopy(vSrc1, src1, i * sregLower);                                                 \
            Compare<T, cmpMode>(dstReg, vSrc0, vSrc1, preg);                                      \
            DataCopy<uint32_t, Dist::DIST_NORM>((__ubuf__ uint32_t *)dst, dstReg, dstOffset);     \
        }                                                                                         \
    }

#define COUNTER_MODE_B16_VCMPV_VF(cmpMode)                                                      \
    __VEC_SCOPE__                                                                               \
    {                                                                                           \
        RegTensor<T> vSrc0;                                                                     \
        RegTensor<T> vSrc1;                                                                     \
        uint32_t sreg = (uint32_t)count;                                                     \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                          \
        uint16_t repeatTime = CeilDivision(count, sregLower);                               \
        MaskReg preg;                                                                           \
        MaskReg dstReg;                                                                         \
        AddrReg dstOffset;                                                                      \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                            \
            preg = CreatePredicate<T>(sreg);                                                    \
            dstReg = CreatePredicate<T>();                                                      \
            dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);            \
            DataCopy(vSrc0, src0, i * sregLower);                                               \
            DataCopy(vSrc1, src1, i * sregLower);                                               \
            Compare<T, cmpMode>(dstReg, vSrc0, vSrc1, preg);                                    \
            DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg, dstOffset);     \
        }                                                                                       \
    }

#define COUNTER_MODE_B32_VCMPV_VF(cmpMode)                                                                         \
    uint32_t sreg = (uint32_t)count;                                                                            \
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);                                                             \
    uint16_t repeatTime = CeilDivision(count, sregLower);                                                      \
    uint16_t halfRepeatTimes = repeatTime / 2;                                                                    \
    if (halfRepeatTimes > 0) {                                                                                     \
        __VEC_SCOPE__                                                                                              \
        {                                                                                                          \
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {                                                       \
                RegTensor<T> vSrc00, vSrc01;                                                                       \
                RegTensor<T> vSrc10, vSrc11;                                                                       \
                MaskReg dstReg0 = CreatePredicate<T>();                                                            \
                MaskReg dstReg1 = CreatePredicate<T>();                                                            \
                MaskReg dstReg2 = CreatePredicate<T>();                                                            \
                MaskReg dstReg3 = CreatePredicate<T>();                                                            \
                MaskReg preg = CreatePredicate<T>(sreg);                                                           \
                AddrReg dstOffset = CreateAddrReg<U>(2 * sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);               \
                DataCopy(vSrc00, src0, 2 * i * sregLower);                                                         \
                DataCopy(vSrc10, src1, 2 * i * sregLower);                                                         \
                DataCopy(vSrc01, src0 + sregLower, 2 * i * sregLower);                                             \
                DataCopy(vSrc11, src1 + sregLower, 2 * i * sregLower);                                             \
                Compare<T, cmpMode>(dstReg0, vSrc00, vSrc10, preg);                                                \
                Compare<T, cmpMode>(dstReg1, vSrc01, vSrc11, preg);                                                \
                PredicateDeInterleave<uint8_t>(dstReg2, dstReg3, dstReg0, dstReg1);                                \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg2, dstOffset);                   \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;                                                        \
    if (tailTimes > 0) {                                                                                           \
        __ubuf__ T *src0Tail = src0 + sregLower * halfRepeatTimes * 2;                                             \
        __ubuf__ T *src1Tail = src1 + sregLower * halfRepeatTimes * 2;                                             \
        __ubuf__ U *dstTail = (__ubuf__ U *)dst + halfRepeatTimes * sregLower * 2 / sizeof(U) / ONE_BYTE_BIT_SIZE; \
        __VEC_SCOPE__                                                                                              \
        {                                                                                                          \
            for (uint16_t i = 0; i < tailTimes; ++i) {                                                             \
                RegTensor<T> vSrc0, vSrc1;                                                                         \
                MaskReg dstReg0 = CreatePredicate<T>();                                                            \
                MaskReg dstReg1 = CreatePredicate<T>();                                                            \
                AddrReg dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);                   \
                MaskReg preg = CreatePredicate<T>(sreg);                                                           \
                DataCopy(vSrc0, src0Tail, i * sregLower);                                                          \
                DataCopy(vSrc1, src1Tail, i * sregLower);                                                          \
                Compare<T, cmpMode>(dstReg0, vSrc0, vSrc1, preg);                                                  \
                PredicatePack(dstReg1, dstReg0);                                                                   \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dstTail, dstReg1, dstOffset);               \
            }                                                                                                      \
        }                                                                                                          \
    }

// level 0, mask count mode
#define CONTINUOUS_MODE_B8_VCMPV_VF(cmpMode)                                                            \
    __VEC_SCOPE__                                                                                       \
    {                                                                                                   \
        RegTensor<T> vSrc0;                                                                             \
        RegTensor<T> vSrc1;                                                                             \
        uint32_t sreg = (uint32_t)mask;                                                                 \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                                  \
        uint32_t dstCalcElm = sregLower /  ONE_BYTE_BIT_SIZE;                                \
        MaskReg preg = CreatePredicate<T>(sreg);                                                        \
        MaskReg dstReg = CreatePredicate<T>();                                                          \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                                    \
            uint32_t dstOffsetUint32 = i * dstCalcElm;                                                  \
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg);    \
            DataCopy(vSrc1, src1, repeatParams.src1BlkStride, i * repeatParams.src1RepStride, preg);    \
            Compare<T, cmpMode>(dstReg, vSrc0, vSrc1, preg);                                            \
            DataCopy<uint32_t, Dist::DIST_NORM>((__ubuf__ uint32_t *)dst, dstReg, dstOffsetUint32);     \
        }                                                                                               \
    }

#define CONTINUOUS_MODE_B16_VCMPV_VF(cmpMode)                                                         \
    __VEC_SCOPE__                                                                                     \
    {                                                                                                 \
        RegTensor<T> vSrc0;                                                                           \
        RegTensor<T> vSrc1;                                                                           \
        uint32_t sreg = (uint32_t)mask;                                                               \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                                \
        uint32_t dstCalcElm = sregLower / ONE_BYTE_BIT_SIZE;                              \
        MaskReg preg = CreatePredicate<T>(sreg);                                                      \
        MaskReg dstReg = CreatePredicate<T>();                                                        \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                                  \
            uint32_t dstOffsetUint32 = i * dstCalcElm;                                                \
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg);  \
            DataCopy(vSrc1, src1, repeatParams.src1BlkStride, i * repeatParams.src1RepStride, preg);  \
            Compare<T, cmpMode>(dstReg, vSrc0, vSrc1, preg);                                          \
            DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg, dstOffsetUint32);     \
        }                                                                                             \
    }

#define CONTINUOUS_MODE_B32_VCMPV_VF(cmpMode)                                                                       \
    uint32_t sreg = (uint32_t)mask;                                                                                 \
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);                                                              \
    uint16_t halfRepeatTimes = repeatTime / 2;                                                                     \
    uint32_t dstCalcElm = 2 * sregLower / ONE_BYTE_BIT_SIZE;                                            \
    if (halfRepeatTimes > 0) {                                                                                      \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            MaskReg preg = CreatePredicate<T>(sreg);                                                                \
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {                                                        \
                RegTensor<T> vSrc00, vSrc01;                                                                        \
                RegTensor<T> vSrc10, vSrc11;                                                                        \
                MaskReg dstReg0 = CreatePredicate<T>();                                                             \
                MaskReg dstReg1 = CreatePredicate<T>();                                                             \
                MaskReg dstReg2 = CreatePredicate<T>();                                                             \
                MaskReg dstReg3 = CreatePredicate<T>();                                                             \
                uint32_t dstOffsetUint32 = i * dstCalcElm;                                                          \
                DataCopy(vSrc00, src0, repeatParams.src0BlkStride, 2 * i * repeatParams.src0RepStride, preg);       \
                DataCopy(vSrc10, src1, repeatParams.src1BlkStride, 2 * i * repeatParams.src1RepStride, preg);       \
                DataCopy(vSrc01, src0, repeatParams.src0BlkStride, (2 * i + 1) * repeatParams.src0RepStride, preg); \
                DataCopy(vSrc11, src1, repeatParams.src1BlkStride, (2 * i + 1) * repeatParams.src1RepStride, preg); \
                Compare<T, cmpMode>(dstReg0, vSrc00, vSrc10, preg);                                                 \
                Compare<T, cmpMode>(dstReg1, vSrc01, vSrc11, preg);                                                 \
                PredicateDeInterleave<uint8_t>(dstReg2, dstReg3, dstReg0, dstReg1);                                 \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg2, dstOffsetUint32);              \
            }                                                                                                       \
        }                                                                                                           \
    }                                                                                                               \
    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;                                                         \
    if (tailTimes > 0) {                                                                                            \
        __ubuf__ T *src0Tail = src0 + sregLower * halfRepeatTimes * 2 * repeatParams.src0BlkStride;                 \
        __ubuf__ T *src1Tail = src1 + sregLower * halfRepeatTimes * 2 * repeatParams.src1BlkStride;                 \
        __ubuf__ U *dstTail = (__ubuf__ U *)dst + halfRepeatTimes * dstCalcElm;                                     \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            MaskReg preg = CreatePredicate<T>(sreg);                                                                \
            for (uint16_t i = 0; i < tailTimes; ++i) {                                                              \
                RegTensor<T> vSrc0, vSrc1;                                                                          \
                MaskReg dstReg0 = CreatePredicate<T>();                                                             \
                MaskReg dstReg1 = CreatePredicate<T>();                                                             \
                AddrReg dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);                    \
                DataCopy(vSrc0, src0Tail, repeatParams.src0BlkStride, 0, preg);                                     \
                DataCopy(vSrc1, src1Tail, repeatParams.src1BlkStride, 0, preg);                                     \
                Compare<T, cmpMode>(dstReg0, vSrc0, vSrc1, preg);                                                   \
                PredicatePack(dstReg1, dstReg0);                                                                    \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dstTail, dstReg1, dstOffset);                \
            }                                                                                                       \
        }                                                                                                           \
    }

// level 0, mask bit mode
#define BITS_MODE_B16_VCMPV_VF(cmpMode)                                                               \
    __VEC_SCOPE__                                                                                     \
    {                                                                                                 \
        RegTensor<T> vSrc0;                                                                           \
        RegTensor<T> vSrc1;                                                                           \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                                \
        uint32_t dstCalcElm = sregLower / ONE_BYTE_BIT_SIZE;                              \
        MaskReg preg;                                                                                 \
        DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)tempBuf), 0);                   \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                                  \
            MaskReg dstReg = CreatePredicate<T>();                                                    \
            uint32_t dstOffsetUint32 = i * dstCalcElm;                                                \
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg);  \
            DataCopy(vSrc1, src1, repeatParams.src1BlkStride, i * repeatParams.src1RepStride, preg);  \
            Compare<T, cmpMode>(dstReg, vSrc0, vSrc1, preg);                                          \
            DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg, dstOffsetUint32);     \
        }                                                                                             \
    }

#define BITS_MODE_B32_VCMPV_VF(cmpMode)                                                                              \
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);                                                               \
    uint16_t halfRepeatTimes = repeatTime / 2;                                                                      \
    uint32_t dstCalcElm = 2 * sregLower / ONE_BYTE_BIT_SIZE;                                             \
    if (halfRepeatTimes > 0) {                                                                                       \
        __VEC_SCOPE__                                                                                                \
        {                                                                                                            \
            MaskReg preg;                                                                                            \
            MaskReg preg1;                                                                                           \
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)tempBuf), 0);                              \
            PredicateUnPack(preg1, preg);                                                                            \
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {                                                         \
                RegTensor<T> vSrc00, vSrc01;                                                                         \
                RegTensor<T> vSrc10, vSrc11;                                                                         \
                MaskReg dstReg0 = CreatePredicate<T>();                                                              \
                MaskReg dstReg1 = CreatePredicate<T>();                                                              \
                MaskReg dstReg2 = CreatePredicate<T>();                                                              \
                MaskReg dstReg3 = CreatePredicate<T>();                                                              \
                uint32_t dstOffsetUint32 = i * dstCalcElm;                                                           \
                DataCopy(vSrc00, src0, repeatParams.src0BlkStride, 2 * i * repeatParams.src0RepStride, preg1);       \
                DataCopy(vSrc10, src1, repeatParams.src1BlkStride, 2 * i * repeatParams.src1RepStride, preg1);       \
                DataCopy(vSrc01, src0, repeatParams.src0BlkStride, (2 * i + 1) * repeatParams.src0RepStride, preg1); \
                DataCopy(vSrc11, src1, repeatParams.src1BlkStride, (2 * i + 1) * repeatParams.src1RepStride, preg1); \
                Compare<T, cmpMode>(dstReg0, vSrc00, vSrc10, preg1);                                                 \
                Compare<T, cmpMode>(dstReg1, vSrc01, vSrc11, preg1);                                                 \
                PredicateDeInterleave<uint8_t>(dstReg2, dstReg3, dstReg0, dstReg1);                                  \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg2, dstOffsetUint32);               \
            }                                                                                                        \
        }                                                                                                            \
    }                                                                                                                \
    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;                                                          \
    if (tailTimes > 0) {                                                                                             \
        __ubuf__ T *src0Tail = src0 + sregLower * halfRepeatTimes * 2 * repeatParams.src0BlkStride;                  \
        __ubuf__ T *src1Tail = src1 + sregLower * halfRepeatTimes * 2 * repeatParams.src1BlkStride;                  \
        __ubuf__ U *dstTail = (__ubuf__ U *)dst + halfRepeatTimes * dstCalcElm;                                      \
        __VEC_SCOPE__                                                                                                \
        {                                                                                                            \
            MaskReg preg;                                                                                            \
            MaskReg preg1;                                                                                           \
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)tempBuf), 0);                              \
            PredicateUnPack(preg1, preg);                                                                            \
            for (uint16_t i = 0; i < tailTimes; ++i) {                                                               \
                RegTensor<T> vSrc0, vSrc1;                                                                           \
                MaskReg dstReg0 = CreatePredicate<T>();                                                              \
                MaskReg dstReg1 = CreatePredicate<T>();                                                              \
                AddrReg dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);                     \
                DataCopy(vSrc0, src0Tail, repeatParams.src0BlkStride, 0, preg1);                                     \
                DataCopy(vSrc1, src1Tail, repeatParams.src1BlkStride, 0, preg1);                                     \
                Compare<T, cmpMode>(dstReg0, vSrc0, vSrc1, preg1);                                                   \
                PredicatePack(dstReg1, dstReg0);                                                                     \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dstTail, dstReg1, dstOffset);                 \
            }                                                                                                        \
        }                                                                                                            \
    }


// Compare::Level 2
template <typename T, typename U>
typename std::enable_if_t<
!std::is_same<U, uint8_t>::value &&
!std::is_same<U, int8_t>::value &&
!std::is_same<U, uint16_t>::value &&
!std::is_same<U, int16_t>::value &&
!std::is_same<U, half>::value &&
!std::is_same<U, uint32_t>::value &&
!std::is_same<U, int32_t>::value &&
!std::is_same<U, float>::value
>
__aicore__ inline VcmpvImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
typename std::enable_if_t<std::is_same<U, uint8_t>::value || std::is_same<U, int8_t>::value>
__aicore__ inline VcmpvImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B8_VCMPV_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B8_VCMPV_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B8_VCMPV_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B8_VCMPV_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B8_VCMPV_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B8_VCMPV_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<U, uint16_t>::value ||
std::is_same<U, int16_t>::value ||
std::is_same<U, half>::value
>
__aicore__ inline VcmpvImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B16_VCMPV_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B16_VCMPV_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B16_VCMPV_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B16_VCMPV_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B16_VCMPV_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B16_VCMPV_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<U, uint32_t>::value ||
std::is_same<U, int32_t>::value ||
std::is_same<U, float>::value
>
__aicore__ inline VcmpvImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1,
    CMPMODE cmpMode, const uint32_t count)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B32_VCMPV_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B32_VCMPV_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B32_VCMPV_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B32_VCMPV_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B32_VCMPV_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B32_VCMPV_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

// Compare::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
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
__aicore__ inline VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value>
__aicore__ inline VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B8_VCMPV_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B8_VCMPV_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B8_VCMPV_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B8_VCMPV_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B8_VCMPV_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B8_VCMPV_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B16_VCMPV_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B16_VCMPV_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B16_VCMPV_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B32_VCMPV_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B32_VCMPV_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B32_VCMPV_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

// Compare::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
!std::is_same<T, uint16_t>::value &&
!std::is_same<T, int16_t>::value &&
!std::is_same<T, half>::value &&
!std::is_same<T, uint32_t>::value &&
!std::is_same<T, int32_t>::value &&
!std::is_same<T, float>::value
>
__aicore__ inline VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask[2], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask[2], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    switch (cmpMode) {
        case CMPMODE::LT: {
            BITS_MODE_B16_VCMPV_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            BITS_MODE_B16_VCMPV_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            BITS_MODE_B16_VCMPV_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            BITS_MODE_B16_VCMPV_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            BITS_MODE_B16_VCMPV_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            BITS_MODE_B16_VCMPV_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask[2], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    switch (cmpMode) {
        case CMPMODE::LT: {
            BITS_MODE_B32_VCMPV_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            BITS_MODE_B32_VCMPV_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            BITS_MODE_B32_VCMPV_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            BITS_MODE_B32_VCMPV_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            BITS_MODE_B32_VCMPV_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            BITS_MODE_B32_VCMPV_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}


/* ***************************************************************************************
 * *********************************** CompareScalar *************************************
 * ************************************************************************************** */
// CompareScalar::Level 2
#define COUNTER_MODE_B8_VCMPVS_VF(cmpMode)                                                        \
    __VEC_SCOPE__                                                                                 \
    {                                                                                             \
        RegTensor<T> vSrc0;                                                                       \
        uint32_t sreg = (uint32_t)count;                                                       \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                            \
        uint16_t repeatTime = CeilDivision(count, sregLower);                                 \
        MaskReg preg;                                                                             \
        MaskReg dstReg;                                                                           \
        AddrReg dstOffset;                                                                        \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                              \
            preg = CreatePredicate<T>(sreg);                                                      \
            dstReg = CreatePredicate<T>();                                                        \
            dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);              \
            DataCopy(vSrc0, src0, i * sregLower);                                                 \
            CompareScalar<T, cmpMode>(dstReg, vSrc0, src1Scalar, preg);                           \
            DataCopy<uint32_t, Dist::DIST_NORM>((__ubuf__ uint32_t *)dst, dstReg, dstOffset);     \
        }                                                                                         \
    }

#define COUNTER_MODE_B16_VCMPVS_VF(cmpMode)                                                     \
    __VEC_SCOPE__                                                                               \
    {                                                                                           \
        RegTensor<T> vSrc0;                                                                     \
        uint32_t sreg = (uint32_t)count;                                                     \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                          \
        uint16_t repeatTime = CeilDivision(count, sregLower);                               \
        MaskReg preg;                                                                           \
        MaskReg dstReg;                                                                         \
        AddrReg dstOffset;                                                                      \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                            \
            preg = CreatePredicate<T>(sreg);                                                    \
            dstReg = CreatePredicate<T>();                                                      \
            dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);            \
            DataCopy(vSrc0, src0, i * sregLower);                                               \
            CompareScalar<T, cmpMode>(dstReg, vSrc0, src1Scalar, preg);                         \
            DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg, dstOffset);     \
        }                                                                                       \
    }

#define COUNTER_MODE_B32_VCMPVS_VF(cmpMode)                                                                        \
    uint32_t sreg = (uint32_t)count;                                                                            \
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);                                                             \
    uint16_t repeatTime = CeilDivision(count, sregLower);                                                      \
    uint16_t halfRepeatTimes = repeatTime / 2;                                                                    \
    if (halfRepeatTimes > 0) {                                                                                     \
        __VEC_SCOPE__                                                                                              \
        {                                                                                                          \
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {                                                       \
                RegTensor<T> vSrc00, vSrc01;                                                                       \
                MaskReg dstReg0 = CreatePredicate<T>();                                                            \
                MaskReg dstReg1 = CreatePredicate<T>();                                                            \
                MaskReg dstReg2 = CreatePredicate<T>();                                                            \
                MaskReg dstReg3 = CreatePredicate<T>();                                                            \
                MaskReg preg = CreatePredicate<T>(sreg);                                                           \
                AddrReg dstOffset = CreateAddrReg<U>(2 * sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);               \
                DataCopy(vSrc00, src0, 2 * i * sregLower);                                                         \
                DataCopy(vSrc01, src0 + sregLower, 2 * i * sregLower);                                             \
                CompareScalar<T, cmpMode>(dstReg0, vSrc00, src1Scalar, preg);                                      \
                CompareScalar<T, cmpMode>(dstReg1, vSrc01, src1Scalar, preg);                                      \
                PredicateDeInterleave<uint8_t>(dstReg2, dstReg3, dstReg0, dstReg1);                                \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg2, dstOffset);                   \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;                                                        \
    if (tailTimes > 0) {                                                                                           \
        __ubuf__ T *src0Tail = src0 + sregLower * halfRepeatTimes * 2;                                             \
        __ubuf__ U *dstTail = (__ubuf__ U *)dst + halfRepeatTimes * sregLower * 2 / sizeof(U) / ONE_BYTE_BIT_SIZE; \
        __VEC_SCOPE__                                                                                              \
        {                                                                                                          \
            for (uint16_t i = 0; i < tailTimes; ++i) {                                                             \
                RegTensor<T> vSrc0, vSrc1;                                                                         \
                MaskReg dstReg0 = CreatePredicate<T>();                                                            \
                MaskReg dstReg1 = CreatePredicate<T>();                                                            \
                AddrReg dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);                   \
                MaskReg preg = CreatePredicate<T>(sreg);                                                           \
                DataCopy(vSrc0, src0Tail, i * sregLower);                                                          \
                CompareScalar<T, cmpMode>(dstReg0, vSrc0, src1Scalar, preg);                                       \
                PredicatePack(dstReg1, dstReg0);                                                                   \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dstTail, dstReg1, dstOffset);               \
            }                                                                                                      \
        }                                                                                                          \
    }

// CompareScalar::level 0, mask count mode
#define CONTINUOUS_MODE_B8_VCMPVS_VF(cmpMode)                                                           \
    __VEC_SCOPE__                                                                                       \
    {                                                                                                   \
        RegTensor<T> vSrc0;                                                                             \
        uint32_t sreg = (uint32_t)mask;                                                                 \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                                  \
        uint32_t dstCalcElm = sregLower / ONE_BYTE_BIT_SIZE;                                \
        MaskReg preg = CreatePredicate<T>(sreg);                                                        \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                                    \
            MaskReg dstReg = CreatePredicate<T>();                                                      \
            uint32_t dstOffsetUint32 = i * dstCalcElm;                                                  \
            DataCopy(vSrc0, src0, repeatParams.srcBlkStride, i * repeatParams.srcRepStride, preg);      \
            CompareScalar<T, cmpMode>(dstReg, vSrc0, src1Scalar, preg);                                 \
            DataCopy<uint32_t, Dist::DIST_NORM>((__ubuf__ uint32_t *)dst, dstReg, dstOffsetUint32);     \
        }                                                                                               \
    }

#define CONTINUOUS_MODE_B16_VCMPVS_VF(cmpMode)                                                        \
    __VEC_SCOPE__                                                                                     \
    {                                                                                                 \
        RegTensor<T> vSrc0;                                                                           \
        uint32_t sreg = (uint32_t)mask;                                                               \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                                \
        uint32_t dstCalcElm = sregLower / ONE_BYTE_BIT_SIZE;                              \
        MaskReg preg = CreatePredicate<T>(sreg);                                                      \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                                  \
            MaskReg dstReg = CreatePredicate<T>();                                                    \
            uint32_t dstOffsetUint32 = i * dstCalcElm;                                                \
            DataCopy(vSrc0, src0, repeatParams.srcBlkStride, i * repeatParams.srcRepStride, preg);    \
            CompareScalar<T, cmpMode>(dstReg, vSrc0, src1Scalar, preg);                               \
            DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg, dstOffsetUint32);     \
        }                                                                                             \
    }

#define CONTINUOUS_MODE_B32_VCMPVS_VF(cmpMode)                                                                    \
    uint32_t sreg = (uint32_t)mask;                                                                               \
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);                                                            \
    uint16_t halfRepeatTimes = repeatTime / 2;                                                                   \
    uint32_t dstCalcElm = 2 * sregLower / ONE_BYTE_BIT_SIZE;                                          \
    if (halfRepeatTimes > 0) {                                                                                    \
        __VEC_SCOPE__                                                                                             \
        {                                                                                                         \
            MaskReg preg = CreatePredicate<T>(sreg);                                                              \
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {                                                      \
                RegTensor<T> vSrc00, vSrc01;                                                                      \
                RegTensor<T> vSrc10, vSrc11;                                                                      \
                MaskReg dstReg0 = CreatePredicate<T>();                                                           \
                MaskReg dstReg1 = CreatePredicate<T>();                                                           \
                MaskReg dstReg2 = CreatePredicate<T>();                                                           \
                MaskReg dstReg3 = CreatePredicate<T>();                                                           \
                uint32_t dstOffsetUint32 = i * dstCalcElm;                                                        \
                DataCopy(vSrc00, src0, repeatParams.srcBlkStride, 2 * i * repeatParams.srcRepStride, preg);       \
                DataCopy(vSrc01, src0, repeatParams.srcBlkStride, (2 * i + 1) * repeatParams.srcRepStride, preg); \
                CompareScalar<T, cmpMode>(dstReg0, vSrc00, src1Scalar, preg);                                     \
                CompareScalar<T, cmpMode>(dstReg1, vSrc01, src1Scalar, preg);                                     \
                PredicateDeInterleave<uint8_t>(dstReg2, dstReg3, dstReg0, dstReg1);                               \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg2, dstOffsetUint32);            \
            }                                                                                                     \
        }                                                                                                         \
    }                                                                                                             \
    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;                                                       \
    if (tailTimes > 0) {                                                                                          \
        __ubuf__ T *src0Tail = src0 + sregLower * halfRepeatTimes * 2 * repeatParams.srcBlkStride;                \
        __ubuf__ U *dstTail = (__ubuf__ U *)dst + halfRepeatTimes * dstCalcElm;                                   \
        __VEC_SCOPE__                                                                                             \
        {                                                                                                         \
            MaskReg preg = CreatePredicate<T>(sreg);                                                              \
            for (uint16_t i = 0; i < tailTimes; ++i) {                                                            \
                RegTensor<T> vSrc0;                                                                               \
                MaskReg dstReg0 = CreatePredicate<T>();                                                           \
                MaskReg dstReg1 = CreatePredicate<T>();                                                           \
                AddrReg dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);                  \
                DataCopy(vSrc0, src0Tail, repeatParams.srcBlkStride, 0, preg);                                    \
                CompareScalar<T, cmpMode>(dstReg0, vSrc0, src1Scalar, preg);                                      \
                PredicatePack(dstReg1, dstReg0);                                                                  \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dstTail, dstReg1, dstOffset);              \
            }                                                                                                     \
        }                                                                                                         \
    }

// CompareScalar::level 0, mask bit mode
#define BITS_MODE_B16_VCMPVS_VF(cmpMode)                                                              \
    __VEC_SCOPE__                                                                                     \
    {                                                                                                 \
        RegTensor<T> vSrc0;                                                                           \
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));                                \
        uint32_t dstCalcElm = sregLower / ONE_BYTE_BIT_SIZE;                              \
        MaskReg preg;                                                                                 \
        DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)tempBuf), 0);                   \
        for (uint16_t i = 0; i < repeatTime; ++i) {                                                  \
            MaskReg dstReg = CreatePredicate<T>();                                                    \
            uint32_t dstOffsetUint32 = i * dstCalcElm;                                                \
            DataCopy(vSrc0, src0, repeatParams.srcBlkStride, i * repeatParams.srcRepStride, preg);    \
            CompareScalar<T, cmpMode>(dstReg, vSrc0, src1Scalar, preg);                               \
            DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg, dstOffsetUint32);     \
        }                                                                                             \
    }

#define BITS_MODE_B32_VCMPVS_VF(cmpMode)                                                                           \
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);                                                             \
    uint16_t halfRepeatTimes = repeatTime / 2;                                                                    \
    uint32_t dstCalcElm = 2 * sregLower / ONE_BYTE_BIT_SIZE;                                           \
    if (halfRepeatTimes > 0) {                                                                                     \
        __VEC_SCOPE__                                                                                              \
        {                                                                                                          \
            MaskReg preg;                                                                                          \
            MaskReg preg1;                                                                                         \
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)tempBuf), 0);                            \
            PredicateUnPack(preg1, preg);                                                                          \
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {                                                       \
                RegTensor<T> vSrc00, vSrc01;                                                                       \
                MaskReg dstReg0 = CreatePredicate<T>();                                                            \
                MaskReg dstReg1 = CreatePredicate<T>();                                                            \
                MaskReg dstReg2 = CreatePredicate<T>();                                                            \
                MaskReg dstReg3 = CreatePredicate<T>();                                                            \
                uint32_t dstOffsetUint32 = i * dstCalcElm;                                                         \
                DataCopy(vSrc00, src0, repeatParams.srcBlkStride, 2 * i * repeatParams.srcRepStride, preg1);       \
                DataCopy(vSrc01, src0, repeatParams.srcBlkStride, (2 * i + 1) * repeatParams.srcRepStride, preg1); \
                CompareScalar<T, cmpMode>(dstReg0, vSrc00, src1Scalar, preg1);                                     \
                CompareScalar<T, cmpMode>(dstReg1, vSrc01, src1Scalar, preg1);                                     \
                PredicateDeInterleave<uint8_t>(dstReg2, dstReg3, dstReg0, dstReg1);                                \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dst, dstReg2, dstOffsetUint32);             \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;                                                        \
    if (tailTimes > 0) {                                                                                           \
        __ubuf__ T *src0Tail = src0 + sregLower * halfRepeatTimes * 2 * repeatParams.srcBlkStride;                 \
        __ubuf__ U *dstTail = (__ubuf__ U *)dst + halfRepeatTimes * dstCalcElm;                                    \
        __VEC_SCOPE__                                                                                              \
        {                                                                                                          \
            MaskReg preg;                                                                                          \
            MaskReg preg1;                                                                                         \
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)tempBuf), 0);                            \
            PredicateUnPack(preg1, preg);                                                                          \
            for (uint16_t i = 0; i < tailTimes; ++i) {                                                             \
                RegTensor<T> vSrc0;                                                                                \
                MaskReg dstReg0 = CreatePredicate<T>();                                                            \
                MaskReg dstReg1 = CreatePredicate<T>();                                                            \
                AddrReg dstOffset = CreateAddrReg<U>(sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE);                   \
                DataCopy(vSrc0, src0Tail, repeatParams.srcBlkStride, 0, preg1);                                    \
                CompareScalar<T, cmpMode>(dstReg0, vSrc0, src1Scalar, preg1);                                      \
                PredicatePack(dstReg1, dstReg0);                                                                   \
                DataCopy<uint32_t, Dist::DIST_PK>((__ubuf__ uint32_t *)dstTail, dstReg1, dstOffset);               \
            }                                                                                                      \
        }                                                                                                          \
    }

// CompareScalar::Level 2
template <typename T, typename U>
typename std::enable_if_t<
!std::is_same<U, uint8_t>::value &&
!std::is_same<U, int8_t>::value &&
!std::is_same<U, uint16_t>::value &&
!std::is_same<U, int16_t>::value &&
!std::is_same<U, half>::value &&
!std::is_same<U, uint32_t>::value &&
!std::is_same<U, int32_t>::value &&
!std::is_same<U, float>::value
>
__aicore__ inline VcmpvsImpl(__ubuf__ T* dst, __ubuf__ U* src0, const U src1Scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
typename std::enable_if_t<std::is_same<U, uint8_t>::value || std::is_same<U, int8_t>::value>
__aicore__ inline VcmpvsImpl(__ubuf__ T* dst, __ubuf__ U* src0, const U src1Scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B8_VCMPVS_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B8_VCMPVS_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B8_VCMPVS_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B8_VCMPVS_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B8_VCMPVS_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B8_VCMPVS_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<U, uint16_t>::value ||
std::is_same<U, int16_t>::value ||
std::is_same<U, half>::value
>
__aicore__ inline VcmpvsImpl(__ubuf__ T* dst, __ubuf__ U* src0, const U src1Scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B16_VCMPVS_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B16_VCMPVS_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B16_VCMPVS_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B16_VCMPVS_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B16_VCMPVS_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B16_VCMPVS_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<U, uint32_t>::value ||
std::is_same<U, int32_t>::value ||
std::is_same<U, float>::value
>
__aicore__ inline VcmpvsImpl(__ubuf__ T* dst, __ubuf__ U* src0, const U src1Scalar,
    CMPMODE cmpMode, const uint32_t count)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B32_VCMPVS_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B32_VCMPVS_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B32_VCMPVS_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B32_VCMPVS_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B32_VCMPVS_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B32_VCMPVS_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

// CompareScalar::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
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
__aicore__ inline VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, const T src1Scalar,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value>
__aicore__ inline VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, const T src1Scalar,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B8_VCMPVS_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B8_VCMPVS_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B8_VCMPVS_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B8_VCMPVS_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B8_VCMPVS_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B8_VCMPVS_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, const T src1Scalar,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B16_VCMPVS_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B16_VCMPVS_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B16_VCMPVS_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B16_VCMPVS_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B16_VCMPVS_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B16_VCMPVS_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, const T src1Scalar,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B32_VCMPVS_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B32_VCMPVS_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B32_VCMPVS_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B32_VCMPVS_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B32_VCMPVS_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B32_VCMPVS_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
}

// CompareScalar::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
!std::is_same<T, uint16_t>::value &&
!std::is_same<T, int16_t>::value &&
!std::is_same<T, half>::value &&
!std::is_same<T, uint32_t>::value &&
!std::is_same<T, int32_t>::value &&
!std::is_same<T, float>::value
>
__aicore__ inline VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, const T src1Scalar,
    CMPMODE cmpMode, const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, const T src1Scalar,
    CMPMODE cmpMode, const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    switch (cmpMode) {
        case CMPMODE::LT: {
            BITS_MODE_B16_VCMPVS_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            BITS_MODE_B16_VCMPVS_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            BITS_MODE_B16_VCMPVS_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            BITS_MODE_B16_VCMPVS_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            BITS_MODE_B16_VCMPVS_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            BITS_MODE_B16_VCMPVS_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T, typename U, bool isSetMask = true>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, const T src1Scalar,
    CMPMODE cmpMode, const uint64_t mask[2], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    switch (cmpMode) {
        case CMPMODE::LT: {
            BITS_MODE_B32_VCMPVS_VF(CMPMODE::LT);
            break;
        }
        case CMPMODE::GT: {
            BITS_MODE_B32_VCMPVS_VF(CMPMODE::GT);
            break;
        }
        case CMPMODE::EQ: {
            BITS_MODE_B32_VCMPVS_VF(CMPMODE::EQ);
            break;
        }
        case CMPMODE::LE: {
            BITS_MODE_B32_VCMPVS_VF(CMPMODE::LE);
            break;
        }
        case CMPMODE::GE: {
            BITS_MODE_B32_VCMPVS_VF(CMPMODE::GE);
            break;
        }
        case CMPMODE::NE: {
            BITS_MODE_B32_VCMPVS_VF(CMPMODE::NE);
            break;
        }
        default:
            break;
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

// /* ***************************************************************************************
//  * *************************************** Select ****************************************
//  * ************************************************************************************** */
// Level 2, select mode: 1
template <typename T, typename U>
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
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    T src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint8_t>::value ||
std::is_same<T, int8_t>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    T src1, SELMODE selMode, uint32_t count)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        Duplicate(vSrc1, src1);
        uint32_t sreg = (uint32_t)count;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTime = CeilDivision(count, sregLower);
        uint32_t selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_NORM>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            MaskReg dstReg = CreatePredicate<T>(sreg);
            DataCopy(vSrc0, src0, i * sregLower);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, i * sregLower, dstReg);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    T src1, SELMODE selMode, uint32_t count)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        Duplicate(vSrc1, src1);
        uint32_t sreg = (uint32_t)count;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTime = CeilDivision(count, sregLower);
        uint32_t selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            MaskReg dstReg = CreatePredicate<T>(sreg);
            DataCopy(vSrc0, src0, i * sregLower);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, i * sregLower, dstReg);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    T src1, SELMODE selMode, uint32_t count)
{
    uint32_t sreg = (uint32_t)count;
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);
    uint16_t repeatTime = CeilDivision(count, sregLower);
    uint16_t halfRepeatTimes = repeatTime / 2;

    if (halfRepeatTimes > 0) {
        __VEC_SCOPE__
        {
            RegTensor<T> vSrc1;
            Duplicate(vSrc1, src1);
            uint32_t selMaskOffset = 2 * sregLower / ONE_BYTE_BIT_SIZE;
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {
                RegTensor<T> vSrc00, vSrc01;
                RegTensor<T> vDst0, vDst1;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T>();
                MaskReg preg2 = CreatePredicate<T>();
                MaskReg preg3 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
                MaskReg dstReg = CreatePredicate<T>(sreg);
                DataCopy(vSrc00, src0, 2 * i * sregLower);
                DataCopy(vSrc01, src0 + sregLower, 2 * i * sregLower);
                PredicateInterleave<uint16_t>(preg2, preg3, preg0, preg1);
                Select<T>(vDst0, vSrc00, vSrc1, preg2);
                Select<T>(vDst1, vSrc01, vSrc1, preg3);
                DataCopy(dst, vDst0, 2 * i * sregLower, dstReg);
                DataCopy(dst + sregLower, vDst1, 2 * i * sregLower, dstReg);
            }
        }
    }

    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;
    if (tailTimes > 0) {
        __ubuf__ T* src0Tail = src0 + halfRepeatTimes * sregLower * 2;
        __ubuf__ U* selTail = (__ubuf__ U *)sel + halfRepeatTimes * 2 * sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE;
        __ubuf__ T* dstTail = dst + halfRepeatTimes * sregLower * 2;
        __VEC_SCOPE__
        {
            RegTensor<T> vSrc1;
            Duplicate(vSrc1, src1);
            for (uint16_t i = 0; i < tailTimes; ++i) {
                RegTensor<T> vSrc0;
                RegTensor<T> vDst;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)selTail), 0);
                MaskReg dstReg = CreatePredicate<T>(sreg);
                DataCopy(vSrc0, src0Tail, 0);
                PredicateUnPack(preg1, preg0);
                Select<T>(vDst, vSrc0, vSrc1, preg1);
                DataCopy(dstTail, vDst, 0, dstReg);
            }
        }
    }
}


// Level 2, select mode: 0/2
template <typename T, typename U>
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
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    __ubuf__ T* src1, SELMODE selMode, uint32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint8_t>::value ||
std::is_same<T, int8_t>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    __ubuf__ T* src1, SELMODE selMode, uint32_t count)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        uint32_t sreg = (uint32_t)count;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTime = CeilDivision(count, sregLower);
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        }
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_NORM>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            MaskReg dstReg = CreatePredicate<T>(sreg);
            DataCopy(vSrc0, src0, i * sregLower);
            DataCopy(vSrc1, src1, i * sregLower);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, i * sregLower, dstReg);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    __ubuf__ T* src1, SELMODE selMode, uint32_t count)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        uint32_t sreg = (uint32_t)count;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTime = CeilDivision(count, sregLower);
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        }
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            MaskReg dstReg = CreatePredicate<T>(sreg);
            DataCopy(vSrc0, src0, i * sregLower);
            DataCopy(vSrc1, src1, i * sregLower);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, i * sregLower, dstReg);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    __ubuf__ T* src1, SELMODE selMode, uint32_t count)
{
    uint32_t sreg = (uint32_t)count;
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);
    uint16_t repeatTime = CeilDivision(count, sregLower);
    uint16_t halfRepeatTimes = repeatTime / 2;
    if (halfRepeatTimes > 0) {
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = 2 * sregLower / ONE_BYTE_BIT_SIZE;
        }
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {
                RegTensor<T> vSrc00, vSrc01;
                RegTensor<T> vSrc10, vSrc11;
                RegTensor<T> vDst0, vDst1;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T, Pat::ALLF>();
                MaskReg preg2 = CreatePredicate<T>();
                MaskReg preg3 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
                MaskReg dstReg = CreatePredicate<T>(sreg);
                DataCopy(vSrc00, src0, 2 * i * sregLower);
                DataCopy(vSrc10, src1, 2 * i * sregLower);
                DataCopy(vSrc01, src0 + sregLower, 2 * i * sregLower);
                DataCopy(vSrc11, src1 + sregLower, 2 * i * sregLower);
                PredicateInterleave<uint16_t>(preg2, preg3, preg0, preg1); // u8,u16均可以
                Select<T>(vDst0, vSrc00, vSrc10, preg2);
                if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
                    Select<T>(vDst1, vSrc01, vSrc11, preg2); // SELMODE::VSEL_CMPMASK_SPR使用preg2,固定使用前64bit
                }
                else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
                    Select<T>(vDst1, vSrc01, vSrc11, preg3); // SELMODE::VSEL_TENSOR_TENSOR_MODE使用preg3,连续消耗
                }
                DataCopy(dst, vDst0, 2 * i * sregLower, dstReg);
                DataCopy(dst + sregLower, vDst1, 2 * i * sregLower, dstReg);
            }
        }
    }

    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;
    if (tailTimes > 0) {
        __ubuf__ T* src0Tail = src0 + halfRepeatTimes * sregLower * 2;
        __ubuf__ T* src1Tail = src1 + halfRepeatTimes * sregLower * 2;
        uint16_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = halfRepeatTimes * 2 * sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE; // 单位为元素个数
        }
        __ubuf__ U* selTail = (__ubuf__ U *)sel + selMaskOffset;
        __ubuf__ T* dstTail = dst + halfRepeatTimes * sregLower * 2;
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < tailTimes; ++i) {
                RegTensor<T> vSrc0;
                RegTensor<T> vSrc1;
                RegTensor<T> vDst;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)selTail), 0);
                MaskReg dstReg = CreatePredicate<T>(sreg);
                DataCopy(vSrc0, src0Tail, 0);
                DataCopy(vSrc1, src1Tail, 0);
                PredicateUnPack(preg1, preg0);
                Select<T>(vDst, vSrc0, vSrc1, preg1);
                DataCopy(dstTail, vDst, 0, dstReg);
            }
        }
    }
}


// Level 0, continuous mode, select mode: 1
template <typename T, typename U>
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
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* selMask, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint8_t>::value ||
std::is_same<T, int8_t>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        Duplicate(vSrc1, src1);
        uint32_t sreg = (uint32_t)mask;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint32_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
        MaskReg preg1 = CreatePredicate<T>(sreg);
        uint32_t selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_NORM>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg1);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, repeatParams.dstBlkStride, i * repeatParams.dstRepStride, preg1);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        Duplicate(vSrc1, src1);
        uint32_t sreg = (uint32_t)mask;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint32_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
        MaskReg preg1 = CreatePredicate<T>(sreg);
        uint32_t selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg1);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, repeatParams.dstBlkStride, i * repeatParams.dstRepStride, preg1);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    uint32_t sreg = (uint32_t)mask;
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);
    uint16_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
    uint16_t halfRepeatTimes = repeatTime / 2;
    if (halfRepeatTimes > 0) {
        __VEC_SCOPE__
        {
            RegTensor<T> vSrc1;
            Duplicate(vSrc1, src1);
            MaskReg dstReg = CreatePredicate<T>(sreg);
            uint32_t selMaskOffset = 2 * sregLower / ONE_BYTE_BIT_SIZE;
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {
                RegTensor<T> vSrc00, vSrc01;
                RegTensor<T> vDst0, vDst1;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T>();
                MaskReg preg2 = CreatePredicate<T>();
                MaskReg preg3 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
                DataCopy(vSrc00, src0, repeatParams.src0BlkStride, 2 * i * repeatParams.src0RepStride, dstReg);
                DataCopy(vSrc01, src0, repeatParams.src0BlkStride, (2 * i + 1) * repeatParams.src0RepStride, dstReg);
                PredicateInterleave<uint16_t>(preg2, preg3, preg0, preg1);
                Select<T>(vDst0, vSrc00, vSrc1, preg2);
                Select<T>(vDst1, vSrc01, vSrc1, preg3);
                DataCopy(dst, vDst0, repeatParams.dstBlkStride,  2 * i * repeatParams.dstRepStride, dstReg);
                DataCopy(dst, vDst1, repeatParams.dstBlkStride, (2 * i + 1) * repeatParams.dstRepStride, dstReg);
            }
        }
    }

    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;
    if (tailTimes > 0) {
        __ubuf__ T* src0Tail = src0 + 2 * halfRepeatTimes * repeatParams.src0RepStride * blockElm;
        __ubuf__ U* selTail = (__ubuf__ U *)sel + halfRepeatTimes * 2 * sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE;
        __ubuf__ T* dstTail = dst + 2 * halfRepeatTimes * repeatParams.dstRepStride * blockElm;
        __VEC_SCOPE__
        {
            RegTensor<T> vSrc1;
            Duplicate(vSrc1, src1);
            MaskReg dstReg = CreatePredicate<T>(sreg);
            for (uint16_t i = 0; i < tailTimes; ++i) {
                RegTensor<T> vSrc0;
                RegTensor<T> vDst;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)selTail), 0);
                DataCopy(vSrc0, src0Tail, repeatParams.src0BlkStride, 0, dstReg);
                PredicateUnPack(preg1, preg0);
                Select<T>(vDst, vSrc0, vSrc1, preg1);
                DataCopy(dstTail, vDst, repeatParams.dstBlkStride, 0, dstReg);
            }
        }
    }
}

// Level 0, continuous mode, select mode: 0/2
template <typename T, typename U>
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
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* selMask, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint8_t>::value ||
std::is_same<T, int8_t>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        uint32_t sreg = (uint32_t)mask;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint32_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
        MaskReg preg1 = CreatePredicate<T>(sreg);
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        }
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_NORM>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg1);
            DataCopy(vSrc1, src1, repeatParams.src1BlkStride, i * repeatParams.src1RepStride, preg1);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, repeatParams.dstBlkStride, i * repeatParams.dstRepStride, preg1);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        uint32_t sreg = (uint32_t)mask;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint32_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
        MaskReg preg1 = CreatePredicate<T>(sreg);
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        }
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg1);
            DataCopy(vSrc1, src1, repeatParams.src1BlkStride, i * repeatParams.src1RepStride, preg1);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, repeatParams.dstBlkStride, i * repeatParams.dstRepStride, preg1);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    uint32_t sreg = (uint32_t)mask;
    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);
    uint16_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
    uint16_t halfRepeatTimes = repeatTime / 2;
    if (halfRepeatTimes > 0) {
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = 2 * sregLower / ONE_BYTE_BIT_SIZE;
        }
        __VEC_SCOPE__
        {
            MaskReg dstReg = CreatePredicate<T>(sreg);
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {
                RegTensor<T> vSrc00, vSrc01;
                RegTensor<T> vSrc10, vSrc11;
                RegTensor<T> vDst0, vDst1;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T>();
                MaskReg preg2 = CreatePredicate<T>();
                MaskReg preg3 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
                DataCopy(vSrc00, src0, repeatParams.src0BlkStride, 2 * i * repeatParams.src0RepStride, dstReg);
                DataCopy(vSrc10, src1, repeatParams.src1BlkStride, 2 * i * repeatParams.src1RepStride, dstReg);
                DataCopy(vSrc01, src0, repeatParams.src0BlkStride, (2 * i + 1) * repeatParams.src0RepStride, dstReg);
                DataCopy(vSrc11, src1, repeatParams.src1BlkStride, (2 * i + 1) * repeatParams.src1RepStride, dstReg);
                PredicateInterleave<uint16_t>(preg2, preg3, preg0, preg1);
                Select<T>(vDst0, vSrc00, vSrc10, preg2);
                if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
                    Select<T>(vDst1, vSrc01, vSrc11, preg2); // SELMODE::VSEL_CMPMASK_SPR使用preg2,固定使用前64bit
                }
                else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
                    Select<T>(vDst1, vSrc01, vSrc11, preg3); // SELMODE::VSEL_TENSOR_TENSOR_MODE使用preg3,连续消耗
                }
                DataCopy(dst, vDst0, repeatParams.dstBlkStride,  2 * i * repeatParams.dstRepStride, dstReg);
                DataCopy(dst, vDst1, repeatParams.dstBlkStride, (2 * i + 1) * repeatParams.dstRepStride, dstReg);
            }
        }
    }

    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;
    if (tailTimes > 0) {
        __ubuf__ T* src0Tail = src0 + 2 * halfRepeatTimes * repeatParams.src0RepStride * blockElm;
        __ubuf__ T* src1Tail = src1 + 2 * halfRepeatTimes * repeatParams.src1RepStride * blockElm;
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = halfRepeatTimes * 2 * sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE;
        }
        __ubuf__ U* selTail = (__ubuf__ U *)sel + selMaskOffset;
        __ubuf__ T* dstTail = dst + 2 * halfRepeatTimes * repeatParams.dstRepStride * blockElm;
        __VEC_SCOPE__
        {
            MaskReg dstReg = CreatePredicate<T>(sreg);
            for (uint16_t i = 0; i < tailTimes; ++i) {
                RegTensor<T> vSrc0;
                RegTensor<T> vSrc1;
                RegTensor<T> vDst;
                MaskReg preg0 = CreatePredicate<T>();
                MaskReg preg1 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)selTail), 0);
                DataCopy(vSrc0, src0Tail, repeatParams.src0BlkStride, 0, dstReg);
                DataCopy(vSrc1, src1Tail, repeatParams.src1BlkStride, 0, dstReg);
                PredicateUnPack(preg1, preg0);
                Select<T>(vDst, vSrc0, vSrc1, preg1);
                DataCopy(dstTail, vDst, repeatParams.dstBlkStride, 0, dstReg);
            }
        }
    }
}


// Level 0, bit mode, select mode: 1
template <typename T, typename U>
typename std::enable_if_t<
!std::is_same<T, uint16_t>::value &&
!std::is_same<T, int16_t>::value &&
!std::is_same<T, half>::value &&
!std::is_same<T, uint32_t>::value &&
!std::is_same<T, int32_t>::value &&
!std::is_same<T, float>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* selMask, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        Duplicate(vSrc1, src1);
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint32_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
        uint32_t selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        MaskReg preg1;
        DataCopy<uint32_t, Dist::DIST_US>(preg1, ((__ubuf__ uint32_t *)tempBuf), 0);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg1);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, repeatParams.dstBlkStride, i * repeatParams.dstRepStride, preg1);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);
    uint16_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
    uint16_t halfRepeatTimes = repeatTime / 2;
    if (halfRepeatTimes > 0) {
        uint32_t selMaskOffset = 2 * sregLower / ONE_BYTE_BIT_SIZE;
        __VEC_SCOPE__
        {
            RegTensor<T> vSrc1;
            Duplicate(vSrc1, src1);
            MaskReg dstReg;
            MaskReg dstReg1;
            DataCopy<uint32_t, Dist::DIST_US>(dstReg, ((__ubuf__ uint32_t *)tempBuf), 0);
            PredicateUnPack(dstReg1, dstReg);
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {
                RegTensor<T> vSrc00, vSrc01;
                RegTensor<T> vDst0, vDst1;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T>();
                MaskReg preg2 = CreatePredicate<T>();
                MaskReg preg3 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
                DataCopy(vSrc00, src0, repeatParams.src0BlkStride, 2 * i * repeatParams.src0RepStride, dstReg1);
                DataCopy(vSrc01, src0, repeatParams.src0BlkStride, (2 * i + 1) * repeatParams.src0RepStride, dstReg1);
                PredicateInterleave<uint16_t>(preg2, preg3, preg0, preg1);
                Select<T>(vDst0, vSrc00, vSrc1, preg2);
                Select<T>(vDst1, vSrc01, vSrc1, preg3);
                DataCopy(dst, vDst0, repeatParams.dstBlkStride,  2 * i * repeatParams.dstRepStride, dstReg1);
                DataCopy(dst, vDst1, repeatParams.dstBlkStride, (2 * i + 1) * repeatParams.dstRepStride, dstReg1);
            }
        }
    }

    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;
    if (tailTimes > 0) {
        __ubuf__ T* src0Tail = src0 + 2 * halfRepeatTimes * repeatParams.src0RepStride * blockElm;
        __ubuf__ U* selTail = (__ubuf__ U *)sel + halfRepeatTimes * 2 * sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE;
        __ubuf__ T* dstTail = dst + 2 * halfRepeatTimes * repeatParams.dstRepStride * blockElm;
        __VEC_SCOPE__
        {
            RegTensor<T> vSrc1;
            Duplicate(vSrc1, src1);
            MaskReg dstReg;
            MaskReg dstReg1;
            DataCopy<uint32_t, Dist::DIST_US>(dstReg, ((__ubuf__ uint32_t *)tempBuf), 0);
            PredicateUnPack(dstReg1, dstReg);
            for (uint16_t i = 0; i < tailTimes; ++i) {
                RegTensor<T> vSrc0;
                RegTensor<T> vDst;
                MaskReg preg0;
                MaskReg preg1;
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)selTail), 0);
                PredicateUnPack(preg1, preg0);
                DataCopy(vSrc0, src0Tail, repeatParams.src0BlkStride, 0, dstReg1);
                Select<T>(vDst, vSrc0, vSrc1, preg1);
                DataCopy(dstTail, vDst, repeatParams.dstBlkStride, 0, dstReg1);
            }
        }
    }
}


// Level 0, bit mode, select mode: 0/2
template <typename T, typename U>
typename std::enable_if_t<
!std::is_same<T, uint16_t>::value &&
!std::is_same<T, int16_t>::value &&
!std::is_same<T, half>::value &&
!std::is_same<T, uint32_t>::value &&
!std::is_same<T, int32_t>::value &&
!std::is_same<T, float>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* selMask, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint16_t>::value ||
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        RegTensor<T> vSrc0, vSrc1;
        RegTensor<T> vDst;
        uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
        uint32_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = sregLower / ONE_BYTE_BIT_SIZE;
        }
        MaskReg preg1;
        DataCopy<uint32_t, Dist::DIST_US>(preg1, ((__ubuf__ uint32_t *)tempBuf), 0);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            MaskReg preg;
            DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
            DataCopy(vSrc0, src0, repeatParams.src0BlkStride, i * repeatParams.src0RepStride, preg1);
            DataCopy(vSrc1, src1, repeatParams.src1BlkStride, i * repeatParams.src1RepStride, preg1);
            Select<T>(vDst, vSrc0, vSrc1, preg);
            DataCopy(dst, vDst, repeatParams.dstBlkStride, i * repeatParams.dstRepStride, preg1);
        }
    }
}

template <typename T, typename U>
typename std::enable_if_t<
std::is_same<T, uint32_t>::value ||
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask[2], const uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    uint32_t sregLower = VECTOR_REG_WIDTH / sizeof(T);
    uint16_t blockElm = ONE_BLOCK_SIZE / sizeof(T);
    uint16_t halfRepeatTimes = repeatTime / 2;
    if (halfRepeatTimes > 0) {
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = 2 * sregLower / ONE_BYTE_BIT_SIZE;
        }
        __VEC_SCOPE__
        {
            MaskReg dstReg;
            MaskReg dstReg1;
            DataCopy<uint32_t, Dist::DIST_US>(dstReg, ((__ubuf__ uint32_t *)tempBuf), 0);
            PredicateUnPack(dstReg1, dstReg);
            for (uint16_t i = 0; i < halfRepeatTimes; ++i) {
                RegTensor<T> vSrc00, vSrc01;
                RegTensor<T> vSrc10, vSrc11;
                RegTensor<T> vDst0, vDst1;
                MaskReg preg0;
                MaskReg preg1 = CreatePredicate<T>();
                MaskReg preg2 = CreatePredicate<T>();
                MaskReg preg3 = CreatePredicate<T>();
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)sel), i * selMaskOffset);
                DataCopy(vSrc00, src0, repeatParams.src0BlkStride, 2 * i * repeatParams.src0RepStride, dstReg1);
                DataCopy(vSrc10, src1, repeatParams.src1BlkStride, 2 * i * repeatParams.src1RepStride, dstReg1);
                DataCopy(vSrc01, src0, repeatParams.src0BlkStride, (2 * i + 1) * repeatParams.src0RepStride, dstReg1);
                DataCopy(vSrc11, src1, repeatParams.src1BlkStride, (2 * i + 1) * repeatParams.src1RepStride, dstReg1);
                PredicateInterleave<uint16_t>(preg2, preg3, preg0, preg1);
                Select<T>(vDst0, vSrc00, vSrc10, preg2);
                if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
                    Select<T>(vDst1, vSrc01, vSrc11, preg2); // SELMODE::VSEL_CMPMASK_SPR使用preg2,固定使用前64bit
                }
                else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
                    Select<T>(vDst1, vSrc01, vSrc11, preg3); // SELMODE::VSEL_TENSOR_TENSOR_MODE使用preg3,连续消耗
                }
                DataCopy(dst, vDst0, repeatParams.dstBlkStride,  2 * i * repeatParams.dstRepStride, dstReg1);
                DataCopy(dst, vDst1, repeatParams.dstBlkStride, (2 * i + 1) * repeatParams.dstRepStride, dstReg1);
            }
        }
    }

    uint16_t tailTimes = repeatTime - halfRepeatTimes * 2;
    if (tailTimes > 0) {
        __ubuf__ T* src0Tail = src0 + 2 * halfRepeatTimes * repeatParams.src0RepStride * blockElm;
        __ubuf__ T* src1Tail = src1 + 2 * halfRepeatTimes * repeatParams.src1RepStride * blockElm;
        uint32_t selMaskOffset;
        if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
            selMaskOffset = 0;
        }
        else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            selMaskOffset = halfRepeatTimes * 2 * sregLower / sizeof(U) / ONE_BYTE_BIT_SIZE;
        }
        __ubuf__ U* selTail = (__ubuf__ U *)sel + selMaskOffset;
        __ubuf__ T* dstTail = dst + 2 * halfRepeatTimes * repeatParams.dstRepStride * blockElm;
        __VEC_SCOPE__
        {
            MaskReg dstReg;
            MaskReg dstReg1;
            DataCopy<uint32_t, Dist::DIST_US>(dstReg, ((__ubuf__ uint32_t *)tempBuf), 0);
            PredicateUnPack(dstReg1, dstReg);
            for (uint16_t i = 0; i < tailTimes; ++i) {
                RegTensor<T> vSrc0;
                RegTensor<T> vSrc1;
                RegTensor<T> vDst;
                MaskReg preg0;
                MaskReg preg1;
                DataCopy<uint32_t, Dist::DIST_US>(preg0, ((__ubuf__ uint32_t *)selTail), 0);
                PredicateUnPack(preg1, preg0);
                DataCopy(vSrc0, src0Tail, repeatParams.src0BlkStride, 0, dstReg1);
                DataCopy(vSrc1, src1Tail, repeatParams.src1BlkStride, 0, dstReg1);
                Select<T>(vDst, vSrc0, vSrc1, preg1);
                DataCopy(dstTail, vDst, repeatParams.dstBlkStride, 0, dstReg1);
            }
        }
    }
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

template <typename T, bool isSetMask>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask[2], const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Vcmp is not supported!"); });
}

template <typename T, bool isSetMask>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1,
    CMPMODE cmpMode, const uint64_t mask, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Vcmp is not supported!"); });
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H