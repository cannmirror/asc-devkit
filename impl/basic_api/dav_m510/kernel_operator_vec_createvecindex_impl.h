/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file kernel_operator_vec_createvecindex_impl.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#include "kernel_tensor.h"
#include "kernel_operator_vec_template_impl.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckCreateVecIndexApi0SupportedType()
{
    static_assert(SupportType<T, int16_t, int32_t, half, float>(),
        "CreateVecIndex level-0 api only support int16_t/int32_t/half/float on current device");
}

template <typename T> constexpr __aicore__ inline void CheckCreateVecIndexApi2SupportedType()
{
    static_assert(SupportType<T, int8_t, int16_t, int32_t, half, float, int64_t>(),
        "CreateVecIndex level-2 api only support int8_t/int16_t/int32_t/half/float/int64_t on current device");
}

namespace Internal {
template <bool isMaskBitMode, bool isNormalMode, typename T>
__aicore__ inline void VecCreateVecIndexLevel0VFImpl(__ubuf__ T *dst, const T firstValue, const uint64_t maskArray[],
    const uint64_t maskCount, const uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride,
    __ubuf__ uint64_t *maskBuf)
{
    constexpr uint32_t sreg = GetVecLen() / sizeof(T);
    uint32_t count = VecMicroGetCount<true, isNormalMode, isMaskBitMode>(maskArray, maskCount, maskBuf);
    uint16_t newRepeatTimes = VecMicroGetRepeatTimes<T, isNormalMode>(count, repeatTime);
    MicroAPI::MaskReg maskReg;
    if constexpr (isNormalMode) {
        maskReg = VecMicroGetMaskReg<T, true, isNormalMode, isMaskBitMode>(maskBuf, count);
    }
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::Arange(dstVreg, firstValue);
    for (uint16_t index = 0; index < newRepeatTimes; ++index) {
        if constexpr (!isNormalMode) {
            maskReg = VecMicroGetMaskReg<T, true, isNormalMode, isMaskBitMode>(maskBuf, count);
        }
        MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + index * dstRepStride * ElePerBlkT, dstVreg, dstBlkStride, maskReg);
        MicroAPI::Adds(dstVreg, dstVreg, sreg, maskReg);
    }
}
 
template <bool isMaskBitMode, typename T>
__aicore__ inline void VecCreateVecIndexLevel0Template(__ubuf__ T *dst, const T firstValue, const uint64_t maskArray[],
    const uint64_t maskCount, const uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    if constexpr (isMaskBitMode) {
        ASCENDC_ASSERT(maskCount == 0, "maskCount must be 0 when isMaskBitMode is true.");
    } else {
        ASCENDC_ASSERT(maskArray == nullptr, "maskArray must be nullptr when isMaskBitMode is false.");
    }
 
    if (Internal::IsCounterMode()) {
        VF_CALL<VecCreateVecIndexLevel0VFImpl<isMaskBitMode, false, T>>(dst, firstValue, maskArray, maskCount,
            repeatTime, dstBlkStride, dstRepStride, nullptr);
    } else {
        if constexpr (isMaskBitMode) {
            SetVectorMask<T>(maskArray[1], maskArray[0]); // set mask to SPR.MASK, movp in VF
        }
        VF_CALL<VecCreateVecIndexLevel0VFImpl<isMaskBitMode, true, T>>(dst, firstValue, maskArray, maskCount,
            repeatTime, dstBlkStride, dstRepStride, nullptr);
    }
}
} // namespace Internal

// VCI level-0 normal
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dstLocal, const T firstValue, uint64_t mask,
    uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    CheckCreateVecIndexApi0SupportedType<T>();

    __ubuf__ T* dst = (__ubuf__ T*)dstLocal.GetPhyAddr();
    Internal::VecCreateVecIndexLevel0Template<false>(dst, firstValue, nullptr, mask, repeatTime, dstBlkStride, dstRepStride);
}

// VCI level-0 bitwise
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dstLocal, const T firstValue,
    uint64_t mask[], uint8_t repeatTime, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    CheckCreateVecIndexApi0SupportedType<T>();

    __ubuf__ T* dst = (__ubuf__ T*)dstLocal.GetPhyAddr();
    Internal::VecCreateVecIndexLevel0Template<true>(dst, firstValue, mask, 0, repeatTime, dstBlkStride, dstRepStride);
}

// VCI level-2
template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> dstLocal, const T firstValue, uint32_t calCount)
{
    CheckCreateVecIndexApi2SupportedType<T>();

    __ubuf__ T* dstLocalAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    uint32_t sreg = (uint32_t)calCount;
    uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTime = CeilDivision(calCount, sregLower);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg0;
        MicroAPI::MaskReg preg;
        MicroAPI::Arange(vreg0, firstValue);
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            preg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::StoreAlign(dstLocalAddr + i * sregLower, vreg0, preg);
            MicroAPI::Adds(vreg0, vreg0, sregLower, preg);
        }
    }
}

template <typename T = int64_t>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<int64_t> dstLocal, const int64_t firstValue, uint32_t calCount) {
    __ubuf__ int64_t* dstLocalAddr = (__ubuf__ int64_t*)dstLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> vreg0;
        uint32_t sreg = (uint32_t)calCount;
        MicroAPI::MaskReg preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTime = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            preg = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sreg);
            int64_t offset = static_cast<int64_t>(firstValue + i * 64);
            MicroAPI::Arange(vreg0, offset);
            MicroAPI::StoreAlign(dstLocalAddr + i * sregLower, vreg0, preg);
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H