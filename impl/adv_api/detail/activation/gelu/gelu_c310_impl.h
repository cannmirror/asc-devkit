/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file gelu_c310_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_GELU_GELU_IMPL_C310_H
#define IMPL_ACTIVATION_GELU_GELU_IMPL_C310_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../../common/check.h"
#include "../../common/common.h"

namespace AscendC {
namespace Internal {

template <typename T, bool highPrecision>
__aicore__ inline void GeluImplVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t count, const uint16_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    constexpr float coefficientsA = 0.044715;
    constexpr float coefficientsB = 1.5957691216057308;
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<T> tmpReg0;
    MicroAPI::RegTensor<T> tmpReg1;
    MicroAPI::RegTensor<T> tmpReg2;
    MicroAPI::RegTensor<T> tmpReg3;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(count);
        if constexpr (highPrecision) {
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                (MicroAPI::RegTensor<half>&)srcVreg, (__ubuf__ half*)src + i * oneRepElm);
            MicroAPI::Cast<float, half, castTraitB16ToB32>(srcVreg, (MicroAPI::RegTensor<half>&)srcVreg, mask);
        } else {
            MicroAPI::DataCopy(srcVreg, src + i * oneRepElm);
        }
        // y = (input_x + 0.044715 * input_x ^ 3) * 1.5957691
        MicroAPI::Mul(tmpReg0, srcVreg, srcVreg, mask);
        MicroAPI::Mul(tmpReg0, tmpReg0, srcVreg, mask);
        MicroAPI::Muls(tmpReg0, tmpReg0, coefficientsA, mask);
        MicroAPI::Add(tmpReg0, tmpReg0, srcVreg, mask);
        MicroAPI::Muls(tmpReg0, tmpReg0, coefficientsB, mask);
        // exp(min(y, 0))
        MicroAPI::Mins(tmpReg1, tmpReg0, 0.0f, mask);
        MicroAPI::Exp(tmpReg1, tmpReg1, mask);
        // x / (exp^(-abs(y)) + 1)
        MicroAPI::Abs(tmpReg2, tmpReg0, mask);
        MicroAPI::Muls(tmpReg2, tmpReg2, -1.0f, mask);
        MicroAPI::Exp(tmpReg3, tmpReg2, mask);
        MicroAPI::Adds(tmpReg3, tmpReg3, 1.0f, mask);
        MicroAPI::Div(tmpReg3, srcVreg, tmpReg3, mask);
        // x / (exp^(-abs(y)) + 1) * exp(min(y, 0))
        MicroAPI::Mul(dstVreg, tmpReg1, tmpReg3, mask);
        if constexpr (highPrecision) {
            MicroAPI::Cast<half, float, castTraitB32ToB16>((MicroAPI::RegTensor<half>&)dstVreg, dstVreg, mask);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(
                (__ubuf__ half*)dst + i * oneRepElm, (MicroAPI::RegTensor<half>&)dstVreg, mask);
        } else {
            MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
        }
    }
}

template <typename T>
__simd_callee__ inline void FastGeluCoreAlg(MicroAPI::RegTensor<T>& dstVreg, 
    MicroAPI::RegTensor<T>& srcVreg, MicroAPI::MaskReg& mask, MicroAPI::RegTensor<T>& stackVreg)
{
    constexpr float coefficients = -1.702f;
    constexpr float oneFloatScalar = 1.0f;
    MicroAPI::Muls(stackVreg, srcVreg, coefficients, mask);
    MicroAPI::Exp(stackVreg, stackVreg, mask);
    MicroAPI::Adds(stackVreg, stackVreg, oneFloatScalar, mask);
    MicroAPI::Div(dstVreg, srcVreg, stackVreg, mask);
}

template <typename T = half>
__simd_vf__ inline void FastGeluHighPrecisionAlgVF(__ubuf__ T* dst, __ubuf__ T* src,
    const uint32_t dataSize)
{
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<float> srcVregFloat;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<float> dstVregFloat;

    constexpr uint32_t stackSize = VECTOR_REG_WIDTH / sizeof(float);
    uint32_t sreg = dataSize;

    MicroAPI::RegTensor<float> stackVregFloat;
    
    MicroAPI::MaskReg mask;

    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataSize, stackSize));
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(sreg);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, src + i * stackSize);
        MicroAPI::Cast<float, half, castTraitB16ToB32>(srcVregFloat, srcVreg, mask);

        FastGeluCoreAlg<float>(dstVregFloat, srcVregFloat, mask, stackVregFloat);

        MicroAPI::Cast<half, float, castTraitB32ToB16>(dstVreg, dstVregFloat, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * stackSize, dstVreg, mask);
    }
}

template <typename T = half>
__aicore__ inline void FastGeluHighPrecisionAlg(const LocalTensor<half>& dstLocal, const LocalTensor<half>& srcLocal,
    const uint32_t dataSize)
{
    __ubuf__ T* src = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T* dst = (__ubuf__ T *)dstLocal.GetPhyAddr();

    FastGeluHighPrecisionAlgVF<T>(dst, src, dataSize);
}

template <typename T>
__simd_vf__ inline void FastGeluAlgVF(__ubuf__ T* dst, __ubuf__ T* src,
    const uint32_t dataSize)
{
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    constexpr uint32_t stackSize = VECTOR_REG_WIDTH / sizeof(T);
    uint32_t sreg = dataSize;
    MicroAPI::RegTensor<T> stackVreg;
    MicroAPI::MaskReg mask;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataSize, stackSize));
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy<T>(srcVreg, src + i * stackSize);
        FastGeluCoreAlg<T>(dstVreg, srcVreg, mask, stackVreg);
        MicroAPI::DataCopy<T>(dst + i * stackSize, dstVreg, mask);
    }
}

template <typename T>
__aicore__ inline void FastGeluAlg(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t dataSize)
{
    __ubuf__ T* src = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T* dst = (__ubuf__ T *)dstLocal.GetPhyAddr();

    FastGeluAlgVF<T>(dst, src, dataSize);
}

template <typename T>
__simd_callee__ inline void FastGeluV2CoreAlg(MicroAPI::RegTensor<T>& dstVreg, 
    MicroAPI::RegTensor<T>& srcVreg, MicroAPI::MaskReg& mask, MicroAPI::RegTensor<T>& stackVregA,
    MicroAPI::RegTensor<T>& stackVregB, MicroAPI::RegTensor<T>& stackVregC)
{
    constexpr float coefficients = 0.000000000001;
    constexpr float coefficientsHalf = 0.5;
    constexpr float coefficientsA = -0.1444;
    constexpr float coefficientsB = -1.769;
    constexpr float coefficientsBInv = 1.769;
    constexpr float coefficientsC = 0.7071;
    constexpr float coefficientsD = 0.5;
    MicroAPI::Muls(stackVregA, srcVreg, coefficientsC, mask);
    MicroAPI::Abs(stackVregA, stackVregA, mask);
    MicroAPI::Mins(stackVregA, stackVregA, coefficientsBInv, mask);
    MicroAPI::Adds(stackVregA, stackVregA, coefficientsB, mask);
    MicroAPI::Mul(stackVregA, stackVregA, stackVregA, mask);
    MicroAPI::Muls(stackVregA, stackVregA, coefficientsA, mask);
    MicroAPI::Adds(stackVregA, stackVregA, coefficientsD, mask);

    MicroAPI::Adds(stackVregB, srcVreg, coefficients, mask);
    MicroAPI::Abs(stackVregC, stackVregB, mask);
    MicroAPI::Div(stackVregB, stackVregB, stackVregC, mask);

    MicroAPI::Mul(stackVregA, stackVregA, stackVregB, mask);
    MicroAPI::Adds(stackVregA, stackVregA, coefficientsHalf, mask);

    MicroAPI::Mul(dstVreg, srcVreg, stackVregA, mask);
}

template <typename T = half>
__simd_vf__ inline void FastGeluV2HighPrecisionAlgVF(__ubuf__ T* dst, __ubuf__ T* src,
    const uint32_t dataSize)
{
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<float> srcVregFloat;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<float> dstVregFloat;

    constexpr uint32_t stackSize = VECTOR_REG_WIDTH / sizeof(float);
    uint32_t sreg = dataSize;

    MicroAPI::RegTensor<float> stackVregFloat;
    
    MicroAPI::MaskReg mask;

    MicroAPI::RegTensor<float> stackVregA;
    MicroAPI::RegTensor<float> stackVregB;
    MicroAPI::RegTensor<float> stackVregC;

    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataSize, stackSize));
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(sreg);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, src + i * stackSize);
        MicroAPI::Cast<float, half, castTraitB16ToB32>(srcVregFloat, srcVreg, mask);

        FastGeluV2CoreAlg<float>(dstVregFloat, srcVregFloat, mask, stackVregA, stackVregB, stackVregC);

        MicroAPI::Cast<half, float, castTraitB32ToB16>(dstVreg, dstVregFloat, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * stackSize, dstVreg, mask);
    }
}

template <typename T = half>
__aicore__ inline void FastGeluV2HighPrecisionAlg(const LocalTensor<half>& dstLocal, const LocalTensor<half>& srcLocal,
    const uint32_t dataSize)
{
    __ubuf__ T* src = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T* dst = (__ubuf__ T *)dstLocal.GetPhyAddr();

    FastGeluV2HighPrecisionAlgVF<T>(dst, src, dataSize);
}

template <typename T>
__simd_vf__ inline void FastGeluV2AlgVF(__ubuf__ T* dst, __ubuf__ T* src,
    const uint32_t dataSize)
{
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    constexpr uint32_t stackSize = VECTOR_REG_WIDTH / sizeof(T);
    uint32_t sreg = dataSize;

    MicroAPI::RegTensor<T> stackVregA;
    MicroAPI::RegTensor<T> stackVregB;
    MicroAPI::RegTensor<T> stackVregC;
    MicroAPI::MaskReg mask;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataSize, stackSize));
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy<T>(srcVreg, src + i * stackSize);
        FastGeluV2CoreAlg<T>(dstVreg, srcVreg, mask, stackVregA, stackVregB, stackVregC);
        MicroAPI::DataCopy<T>(dst + i * stackSize, dstVreg, mask);
    }
}

template <typename T>
__aicore__ inline void FastGeluV2Alg(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t dataSize)
{
    __ubuf__ T* src = (__ubuf__ T *)srcLocal.GetPhyAddr();
    __ubuf__ T* dst = (__ubuf__ T *)dstLocal.GetPhyAddr();

    FastGeluV2AlgVF<T>(dst, src, dataSize);
}
} // namespace Internal

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void GeluImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    (void)highPerformance;
    static_assert(SupportType<T, half, float>(), "Gelu only support half/float data type on current device!");
    CheckTensorPosition(dstLocal, "dstLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcLocal, "srcLocal", "VECIN, VECOUT, VECCALC");
    CheckCalCount(count, "calCount", dstLocal, "dstLocal", "Gelu");
    CheckCalCount(count, "calCount", srcLocal, "srcLocal", "Gelu");
    if constexpr (highPrecision && sizeof(T) == sizeof(half)) {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(float));
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        VF_CALL<Internal::GeluImplVF<float, true>>(
            (__ubuf__ float*)dstLocal.GetPhyAddr(), (__ubuf__ float*)srcLocal.GetPhyAddr(), count, repeatTimes);
    } else {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        VF_CALL<Internal::GeluImplVF<T, false>>(
            (__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), count, repeatTimes);
    }
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void GeluImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    GeluImpl<T, highPrecision, highPerformance>(dstLocal, srcLocal, count);
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
{
    (void)sharedTmpBuffer;
    (void)highPerformance;
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
#if ASCENDC_CPU_DEBUG
    bool ret = (dataSize <= srcLocal.GetSize()) && (dataSize <= dstLocal.GetSize()) && (dataSize > 0);
    ASCENDC_ASSERT(
        ret, { KERNEL_LOG(KERNEL_ERROR, "DataSize must bigger than 0 and smaller than or equal to src&dst tensor."); });
#endif

    if constexpr (highPrecision && (IsSameType<T, half>::value)) {
        Internal::FastGeluHighPrecisionAlg(dstLocal, srcLocal, dataSize);
    } else {
        Internal::FastGeluAlg(dstLocal, srcLocal, dataSize);
    }
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t dataSize)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    FasterGeluImpl<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluV2Impl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
{
    (void)sharedTmpBuffer;
    (void)highPerformance;
    static_assert((SupportType<T, half, float>()), "current data type is not supported on current device!");
#if ASCENDC_CPU_DEBUG
    bool ret = (dataSize <= srcLocal.GetSize()) && (dataSize <= dstLocal.GetSize()) && (dataSize > 0);
    ASCENDC_ASSERT(
        ret, { KERNEL_LOG(KERNEL_ERROR, "DataSize must bigger than 0 and smaller than or equal to src&dst tensor."); });
#endif

    if constexpr (highPrecision && (IsSameType<T, half>::value)) {
        Internal::FastGeluV2HighPrecisionAlg(dstLocal, srcLocal, dataSize);
    } else {
        Internal::FastGeluV2Alg(dstLocal, srcLocal, dataSize);
    }
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluV2Impl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t dataSize)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    FasterGeluV2Impl<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
}
#pragma end_pipe
} // namespace AscendC
#endif // IMPL_ACTIVATION_GELU_GELU_IMPL_C310_H
