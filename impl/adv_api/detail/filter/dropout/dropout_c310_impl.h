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
 * \file dropout_c310_impl.h
 * \brief
 */
#ifndef LIB_DROPOUT_DROPOUT_C310_IMPL_H
#define LIB_DROPOUT_DROPOUT_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"

namespace AscendC {
namespace Internal {
__simd_callee__ inline void DropOutBitModeFP32Main(__ubuf__ float* dstUb, __ubuf__ float* srcUb,
    __ubuf__ uint8_t* maskUb, MicroAPI::RegTensor<float>& vDivValueReg, uint32_t sreg, uint32_t newRepeatTimes,
    uint16_t loopH, uint32_t srcLastAxis, uint32_t maskLastAxis)
{
    constexpr uint32_t unRollConstant = 2;
    constexpr uint32_t maskBitToByte = 8;
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(float);
    constexpr uint32_t selOffset = repeatElm / maskBitToByte * unRollConstant / (sizeof(float) / sizeof(uint8_t));
    MicroAPI::RegTensor<float> src0Reg;
    MicroAPI::RegTensor<float> src1Reg;
    MicroAPI::RegTensor<float> scalarReg;
    MicroAPI::RegTensor<float> dst0Reg;
    MicroAPI::RegTensor<float> dst1Reg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg selMask0;
    MicroAPI::MaskReg selMask1;
    MicroAPI::MaskReg tmpMask0;
    MicroAPI::MaskReg tmpMask1 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(scalarReg, (const float&)0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(newRepeatTimes); ++i) {
        MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_US>(
            tmpMask0, (__ubuf__ uint32_t*)maskUb + loopH * (maskLastAxis >> 2) + i * selOffset);
        MicroAPI::MaskInterleave<uint16_t>(selMask0, selMask1, tmpMask0, tmpMask1);
        maskReg = MicroAPI::UpdateMask<float>(sreg);
        MicroAPI::DataCopy<float>(src0Reg, srcUb + loopH * srcLastAxis + i * unRollConstant * repeatElm);
        MicroAPI::Select(dst0Reg, src0Reg, scalarReg, selMask0);
        MicroAPI::Mul(dst0Reg, dst0Reg, vDivValueReg, selMask0);
        MicroAPI::DataCopy<float>(dstUb + loopH * srcLastAxis + i * unRollConstant * repeatElm, dst0Reg, maskReg);
        maskReg = MicroAPI::UpdateMask<float>(sreg);
        MicroAPI::DataCopy<float>(src1Reg, srcUb + loopH * srcLastAxis + (i * unRollConstant + 1) * repeatElm);
        MicroAPI::Select(dst1Reg, src1Reg, scalarReg, selMask1);
        MicroAPI::Mul(dst1Reg, dst1Reg, vDivValueReg, selMask1);
        MicroAPI::DataCopy<float>(dstUb + loopH * srcLastAxis + (i * unRollConstant + 1) * repeatElm, dst1Reg, maskReg);
    }
}

template <typename T>
__simd_vf__ inline void VFDropOutBitModeCalc(__ubuf__ T* dstUb, __ubuf__ T* srcUb,
    __ubuf__ uint8_t* maskUb, const T divValue, const uint32_t dataSize)
{
    MicroAPI::RegTensor<T> vDivValueReg;
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint32_t repeatTimes = CeilDivision(dataSize, repeatElm);
    uint32_t sreg = dataSize;
    MicroAPI::Duplicate(vDivValueReg, divValue);
    uint32_t tail = repeatTimes & 1;
    uint32_t newRepeatTimes = repeatTimes >> 1;
    if constexpr (sizeof(T) == 4) {
        DropOutBitModeFP32Main(dstUb, srcUb, maskUb, vDivValueReg, sreg, newRepeatTimes, 0, 0, 0);
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg selMask2;
        MicroAPI::RegTensor<float> src2Reg;
        MicroAPI::RegTensor<float> dst2Reg;
        MicroAPI::RegTensor<float> scalarReg;
        MicroAPI::Duplicate(scalarReg, (const T&)0);
        uint32_t offset = newRepeatTimes * 2 * repeatElm;
        uint32_t selOffset = newRepeatTimes * 4;
        for (uint16_t i = 0; i < static_cast<uint16_t>(tail); ++i) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_US>(selMask2, (__ubuf__ uint32_t*)maskUb + selOffset);
            MicroAPI::MaskUnPack(selMask2, selMask2);
            maskReg = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::DataCopy<float>(src2Reg, srcUb + offset);
            MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
            MicroAPI::Mul(dst2Reg, dst2Reg, vDivValueReg, selMask2);
            MicroAPI::DataCopy<float>(dstUb + offset, dst2Reg, maskReg);
        }
    } else {
        MicroAPI::RegTensor<T> src0Reg;
        MicroAPI::RegTensor<T> src1Reg;
        MicroAPI::RegTensor<T> dstReg;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg selMask;
        MicroAPI::Duplicate(src1Reg, (const T&)0);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_US>(selMask, (__ubuf__ uint32_t*)maskUb + i * 4);
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy<T>(src0Reg, srcUb + i * repeatElm);
            MicroAPI::Select(dstReg, src0Reg, src1Reg, selMask);
            MicroAPI::Mul(dstReg, dstReg, vDivValueReg, selMask);
            MicroAPI::DataCopy<T>(dstUb + i * repeatElm, dstReg, maskReg);
        }
    }
}

template <typename T>
__simd_vf__ inline void VFDropOutBitModeCalcInfo(__ubuf__ T* dstUb, __ubuf__ T* srcUb,
    __ubuf__ uint8_t* maskUb, const T divValue, const DropOutShapeInfo info)
{
    MicroAPI::RegTensor<T> vDivValueReg;
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint32_t repeatTimes = CeilDivision(info.srcLastAxis, repeatElm);
    MicroAPI::Duplicate(vDivValueReg, divValue);
    uint32_t tail = repeatTimes & 1;
    uint32_t newRepeatTimes = repeatTimes >> 1;
    for (uint16_t loopH = 0; loopH < static_cast<uint16_t>(info.firstAxis); ++loopH) {
        uint32_t width = info.srcLastAxis;
        if constexpr (sizeof(T) == 4) {
            DropOutBitModeFP32Main(dstUb, srcUb, maskUb, vDivValueReg, width, newRepeatTimes, loopH,
                info.srcLastAxis, info.maskLastAxis);
        } else {
            MicroAPI::RegTensor<T> src0Reg;
            MicroAPI::RegTensor<T> scalarReg;
            MicroAPI::RegTensor<T> dstReg;
            MicroAPI::MaskReg maskReg;
            MicroAPI::MaskReg selMask;
            MicroAPI::Duplicate(scalarReg, (const T&)0);
            for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_US>(
                    selMask, (__ubuf__ uint32_t*)maskUb + loopH * (info.maskLastAxis >> 2) + i * 4);
                maskReg = MicroAPI::UpdateMask<T>(width);
                MicroAPI::DataCopy<T>(src0Reg, srcUb + loopH * info.srcLastAxis + i * repeatElm);
                MicroAPI::Select(dstReg, src0Reg, scalarReg, selMask);
                MicroAPI::Mul(dstReg, dstReg, vDivValueReg, selMask);
                MicroAPI::DataCopy<T>(dstUb + loopH * info.srcLastAxis + i * repeatElm, dstReg, maskReg);
            }
        }
    }
    if constexpr (sizeof(T) == 4) {
        if (tail != 0) {
            for (uint16_t loopH = 0; loopH < static_cast<uint16_t>(info.firstAxis); ++loopH) {
                uint32_t selOffset = newRepeatTimes * 4;
                uint32_t offset = newRepeatTimes * 2 * repeatElm;
                uint32_t sreg = info.srcLastAxis - offset;
                MicroAPI::MaskReg maskReg;
                MicroAPI::MaskReg selMask2;
                MicroAPI::RegTensor<float> src2Reg;
                MicroAPI::RegTensor<float> dst2Reg;
                MicroAPI::RegTensor<float> scalarReg;
                MicroAPI::Duplicate(scalarReg, (const T&)0);
                MicroAPI::DataCopy<uint32_t, MicroAPI::MaskDist::DIST_US>(
                    selMask2, (__ubuf__ uint32_t*)maskUb + selOffset + loopH * (info.maskLastAxis >> 2));
                MicroAPI::MaskUnPack(selMask2, selMask2);
                maskReg = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::DataCopy<float>(src2Reg, srcUb + offset + loopH * info.srcLastAxis);
                MicroAPI::Select(dst2Reg, src2Reg, scalarReg, selMask2);
                MicroAPI::Mul(dst2Reg, dst2Reg, vDivValueReg, selMask2);
                MicroAPI::DataCopy<float>(dstUb + offset + loopH * info.srcLastAxis, dst2Reg, maskReg);
            }
        }
    }
}

template <typename T>
__simd_vf__ inline void VFDropOutByteModeCalc(__ubuf__ T* dstUb, __ubuf__ T* srcUb,
    __ubuf__ uint8_t* maskUb, const T divValue, const uint32_t dataSize)
{
    MicroAPI::RegTensor<T> vSrcReg;
    MicroAPI::RegTensor<T> vDstReg;
    MicroAPI::RegTensor<T> vDivValueReg;
    MicroAPI::RegTensor<uint8_t> vMaskReg;
    MicroAPI::RegTensor<half> vFP16Reg;
    MicroAPI::RegTensor<float> vFP32Reg;
    MicroAPI::RegTensor<bfloat16_t> vBF16Reg;
    MicroAPI::MaskReg maskReg;
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint32_t sreg = dataSize;
    uint32_t repeatTimes = CeilDivision(dataSize, repeatElm);
    MicroAPI::Duplicate(vDivValueReg, divValue);
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(vSrcReg, srcUb + i * repeatElm);
        if constexpr (sizeof(T) == 2) {
            MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(vMaskReg, maskUb + i * repeatElm);
            MicroAPI::Cast<half, uint8_t, layoutZMrgZ>(vFP16Reg, vMaskReg, maskReg);
            if constexpr (SupportType<T, half>()) {
                MicroAPI::Mul(vDstReg, vFP16Reg, vSrcReg, maskReg);
            } else {
                MicroAPI::Cast<bfloat16_t, half, MrgZRndR>(vBF16Reg, vFP16Reg, maskReg);
                MicroAPI::Mul(vDstReg, vBF16Reg, vSrcReg, maskReg);
            }
        } else {
            MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(vMaskReg, maskUb + i * repeatElm);
            MicroAPI::Cast<half, uint8_t, layoutZMrgZ>(vFP16Reg, vMaskReg, maskReg);
            MicroAPI::Cast<float, half, layoutZMrgZ>(vFP32Reg, vFP16Reg, maskReg);
            MicroAPI::Mul(vDstReg, vFP32Reg, vSrcReg, maskReg);
        }
        MicroAPI::Mul(vDstReg, vDivValueReg, vDstReg, maskReg);
        MicroAPI::DataCopy(dstUb + i * repeatElm, vDstReg, maskReg);
    }
}

template <typename T>
__simd_vf__ inline void VFDropOutByteModeCalcInfo(__ubuf__ T* dstUb, __ubuf__ T* srcUb,
    __ubuf__ uint8_t* maskUb, const T divValue, const DropOutShapeInfo info)
{
    MicroAPI::RegTensor<T> vSrcReg;
    MicroAPI::RegTensor<T> vDstReg;
    MicroAPI::RegTensor<T> vDivValueReg;
    MicroAPI::RegTensor<uint8_t> vMaskReg;
    MicroAPI::RegTensor<half> vFP16Reg;
    MicroAPI::RegTensor<float> vFP32Reg;
    MicroAPI::RegTensor<bfloat16_t> vBF16Reg;
    MicroAPI::MaskReg maskReg;
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint32_t loopWNum = CeilDivision(info.srcLastAxis, repeatElm);
    MicroAPI::Duplicate(vDivValueReg, divValue);
    for (uint16_t loopH = 0; loopH < static_cast<uint16_t>(info.firstAxis); ++loopH) {
        uint32_t width = info.srcLastAxis;
        for (uint16_t loopW = 0; loopW < static_cast<uint16_t>(loopWNum); ++loopW) {
            maskReg = MicroAPI::UpdateMask<T>(width);
            MicroAPI::DataCopy<T>(vSrcReg, srcUb + loopH * info.srcLastAxis + loopW * repeatElm);
            if constexpr (sizeof(T) == 2) {
                MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                    vMaskReg, maskUb + loopH * info.maskLastAxis + loopW * repeatElm);
                MicroAPI::Cast<half, uint8_t, layoutZMrgZ>(vFP16Reg, vMaskReg, maskReg);
                if constexpr (SupportType<T, half>()) {
                    MicroAPI::Mul(vDstReg, vFP16Reg, vSrcReg, maskReg);
                } else {
                    MicroAPI::Cast<bfloat16_t, half, MrgZRndR>(vBF16Reg, vFP16Reg, maskReg);
                    MicroAPI::Mul(vDstReg, vBF16Reg, vSrcReg, maskReg);
                }
            } else {
                MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    vMaskReg, maskUb + loopH * info.maskLastAxis + loopW * repeatElm);
                MicroAPI::Cast<half, uint8_t, layoutZMrgZ>(vFP16Reg, vMaskReg, maskReg);
                MicroAPI::Cast<float, half, layoutZMrgZ>(vFP32Reg, vFP16Reg, maskReg);
                MicroAPI::Mul(vDstReg, vFP32Reg, vSrcReg, maskReg);
            }
            MicroAPI::Mul(vDstReg, vDivValueReg, vDstReg, maskReg);
            MicroAPI::DataCopy(dstUb + loopH * info.srcLastAxis + loopW * repeatElm, vDstReg, maskReg);
        }
    }
}
} // namespace Internal

template <typename T, bool isInitBitMode = false>
__aicore__ inline void DropOutBitMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const uint32_t dataSize)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
    (void)sharedTmpBuffer;
    __ubuf__ T *srcUb = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T *dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ uint8_t *maskUb = (__ubuf__ uint8_t *)maskLocal.GetPhyAddr();

    Internal::VFDropOutBitModeCalc<T>(dstUb, srcUb, maskUb, divValue, dataSize);
}

template <typename T, bool isInitBitMode = false>
__aicore__ inline void DropOutBitMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const DropOutShapeInfo& info)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
    __ubuf__ T *srcUb = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T *dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ uint8_t *maskUb = (__ubuf__ uint8_t *)maskLocal.GetPhyAddr();

    Internal::VFDropOutBitModeCalcInfo<T>(dstUb, srcUb, maskUb, divValue, info);
}

template <typename T>
__aicore__ inline void DropOutByteMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const uint32_t dataSize)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
    (void)sharedTmpBuffer;
    __ubuf__ T *srcUb = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T *dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ uint8_t *maskUb = (__ubuf__ uint8_t *)maskLocal.GetPhyAddr();

    Internal::VFDropOutByteModeCalc<T>(dstUb, srcUb, maskUb, divValue, dataSize);
}

template <typename T>
__aicore__ inline void DropOutByteMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const DropOutShapeInfo& info)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
    (void)sharedTmpBuffer;
    __ubuf__ T *srcUb = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T *dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ uint8_t *maskUb = (__ubuf__ uint8_t *)maskLocal.GetPhyAddr();

    Internal::VFDropOutByteModeCalcInfo<T>(dstUb, srcUb, maskUb, divValue, info);
}
} // namespace AscendC
#endif // LIB_DROPOUT_DROPOUT_C310_IMPL_H
