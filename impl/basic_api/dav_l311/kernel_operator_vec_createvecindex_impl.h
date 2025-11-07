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
 * \file kernel_operator_vec_createvecindex_impl.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#include "kernel_tensor.h"
#include "kernel_operator_common_impl.h"
#if __CCE_KT_TEST__
#include "kernel_check.h"
#endif

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckCreateVecIndexApi0SupportedType()
{
    static_assert(std::is_same<T, int8_t>::value ||
        std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value ||
        std::is_same<T, half>::value || std::is_same<T, float>::value,
        "CreateVecIndex level-0 api only support int8_t/int16_t/int32_t/half/float");
}

template <typename T> constexpr __aicore__ inline void CheckCreateVecIndexApi2SupportedType()
{
    static_assert(std::is_same<T, int8_t>::value ||
        std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value ||
        std::is_same<T, half>::value || std::is_same<T, float>::value,
        "CreateVecIndex level-2 api only support int8_t/int16_t/int32_t/half/float");
}

// VCI level-0 normal
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dst, const T firstValue,
    uint64_t mask, uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    CheckCreateVecIndexApi0SupportedType<T>();

    __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
    uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint32_t sreg = static_cast<uint32_t>(mask);
    int64_t addLength = static_cast<int64_t>(sregLower);

    __VEC_SCOPE__
    {
        RegTensor<T> vDst;
        MaskReg preg = CreatePredicate<T>(sreg);
        CreateVecIndex(vDst, firstValue);
        DataCopy(dstAddr, vDst, dstBlkStride, 0, preg);
        for (uint16_t i = 1; i < (uint16_t)repeatTime; ++i) {
            Adds(vDst, vDst, addLength, preg);
            DataCopy(dstAddr, vDst, dstBlkStride, i * dstRepStride, preg);
        }
    }
}

// VCI level-0 bitwise
template <typename T>
typename std::enable_if_t<
!std::is_same<T, int16_t>::value &&
!std::is_same<T, half>::value &&
!std::is_same<T, int32_t>::value &&
!std::is_same<T, float>::value
>
__aicore__ inline CreateVecIndexCalc(LocalTensor<T> &dst, const T firstValue,
    uint64_t mask[2], uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    static_assert(std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value ||
        std::is_same<T, half>::value || std::is_same<T, float>::value,
        "CreateVecIndex level-0 bit mode api only support int16_t/int32_t/half/float");
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T>
typename std::enable_if_t<
std::is_same<T, int16_t>::value ||
std::is_same<T, half>::value
>
__aicore__ inline CreateVecIndexCalc(LocalTensor<T> &dst, const T firstValue,
    uint64_t mask[2], uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    __ubuf__ uint8_t* tmpBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tmpBuf) = mask[0];
    *((__ubuf__ uint64_t*)tmpBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
    uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    int64_t addLength = (int64_t)sregLower;

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        MaskReg preg;
        DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)tmpBuf), 0);
        CreateVecIndex(vreg0, firstValue);
        DataCopy(dstAddr, vreg0, dstBlkStride, 0, preg);
        for (uint16_t i = 1; i < (uint16_t)repeatTime; ++i) {
            Adds(vreg0, vreg0, addLength, preg);
            DataCopy(dstAddr, vreg0, dstBlkStride, i * dstRepStride, preg);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tmpBuf);
}

template <typename T>
typename std::enable_if_t<
std::is_same<T, int32_t>::value ||
std::is_same<T, float>::value
>
__aicore__ inline CreateVecIndexCalc(LocalTensor<T> &dst, const T firstValue,
    uint64_t mask[2], uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    __ubuf__ uint8_t* tmpBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tmpBuf) = mask[0];
    *((__ubuf__ uint64_t*)tmpBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
    uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    int64_t addLength = static_cast<int64_t>(sregLower);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        MaskReg preg;
        MaskReg preg1;
        DataCopy<uint32_t, Dist::DIST_US>(preg, ((__ubuf__ uint32_t *)tmpBuf), 0);
        PredicateUnPack(preg1, preg);
        CreateVecIndex(vreg0, firstValue);
        DataCopy(dstAddr, vreg0, dstBlkStride, 0, preg1);
        for (uint16_t i = 1; i < (uint16_t)repeatTime; ++i) {
            Adds(vreg0, vreg0, addLength, preg1);
            DataCopy(dstAddr, vreg0, dstBlkStride, i * dstRepStride, preg1);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tmpBuf);
}

// VCI level-2
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> dst, const T firstValue, uint32_t count)
{
    CheckCreateVecIndexApi2SupportedType<T>();

    __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
    uint32_t sreg = (uint32_t)count;
    uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTime = CeilDivision(count, sregLower);
    int64_t addLength = static_cast<int64_t>(sregLower);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        MaskReg preg = CreatePredicate<T>(sreg);
        CreateVecIndex(vreg0, firstValue);
        DataCopy(dstAddr, vreg0, 0, preg);
        for (uint16_t i = 1; i < (uint16_t)repeatTime; ++i) {
            preg = CreatePredicate<T>(sreg);
            Adds(vreg0, vreg0, addLength, preg);
            DataCopy(dstAddr, vreg0, i * sregLower, preg);
        }
    }
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H