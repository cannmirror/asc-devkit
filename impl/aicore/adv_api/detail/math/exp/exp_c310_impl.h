/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file exp_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_EXP_EXP_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_EXP_EXP_C310_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
namespace ExpAPI {
constexpr MicroAPI::CastTrait castTraitF162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait castTraitS162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait castTraitF322F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename T, uint8_t taylorExpandLevel>
__aicore__ inline void ExpCompute(__local_mem__ T* dst, __local_mem__ T* src, uint32_t calCount, uint16_t repeatTimes,
    __local_mem__ float* taylorExpandTmpBuffer)
{
    constexpr float dupConstant = 2.0f;
    constexpr uint32_t floatInf = F32_INF;
    constexpr uint32_t floatNInf = F32_NEG_INF;
    MicroAPI::MaskReg mask, cmpInfMask, cmpNInfMask;
    MicroAPI::RegTensor<T> dstVreg, tempSrcVreg;
    MicroAPI::RegTensor<float> srcVreg, tempDstVreg;
    MicroAPI::RegTensor<float> intVreg, expIntVreg;
    MicroAPI::RegTensor<float> decimalVreg, expDecimalVreg;
    MicroAPI::RegTensor<float> powVreg, denominatorVreg;
    MicroAPI::RegTensor<float> factorialReg, tmpReg;
    MicroAPI::RegTensor<int16_t> iterVreg;
    MicroAPI::RegTensor<float> vReg0, vReg1;

    MicroAPI::Duplicate(vReg0, 0.0f);
    MicroAPI::Duplicate((MicroAPI::RegTensor<uint32_t>&)vReg1, floatInf);
    mask = MicroAPI::CreateMask<float>();
    MicroAPI::Duplicate<float>(tmpReg, 1.0f);
    MicroAPI::Arange<float>(factorialReg, dupConstant);
    MicroAPI::Div(factorialReg, tmpReg, factorialReg, mask);
    MicroAPI::DataCopy(taylorExpandTmpBuffer, factorialReg, mask);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(float);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        if constexpr (IsSameType<T, half>::value) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempSrcVreg, src + i * oneRepSize);
            MicroAPI::Cast<float, T, castTraitF162F32>(srcVreg, tempSrcVreg, mask);
        } else {
            MicroAPI::DataCopy(srcVreg, src + i * oneRepSize);
        }
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(
            cmpInfMask, (MicroAPI::RegTensor<uint32_t>&)srcVreg, floatInf, mask);
        MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(
            cmpNInfMask, (MicroAPI::RegTensor<uint32_t>&)srcVreg, floatNInf, mask);
        // intX = floor(x)
        MicroAPI::Truncate<float, RoundMode::CAST_FLOOR>(intVreg, srcVreg, mask);
        // decimalX = x - intX
        MicroAPI::Sub(decimalVreg, srcVreg, intVreg, mask);
        // expIntX = exp(intX)
        MicroAPI::Exp(expIntVreg, intVreg, mask);
        // expDecimalX = sum((decimalX ^ n) / n!) n is taylorExpandLevel
        MicroAPI::Adds(expDecimalVreg, decimalVreg, 1.0f, mask);
        if constexpr (taylorExpandLevel > 1) {
            powVreg = decimalVreg;
            constexpr uint16_t vloopEnd = taylorExpandLevel - 1;
            for (uint16_t j = 0; j < vloopEnd; ++j) {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(denominatorVreg, taylorExpandTmpBuffer + j);
                MicroAPI::Mul(powVreg, powVreg, decimalVreg, mask);
                MicroAPI::Mul(powVreg, powVreg, denominatorVreg, mask);
                MicroAPI::Add(expDecimalVreg, expDecimalVreg, powVreg, mask);
            }
        }
        // exp(x) = expIntX * expDecimalX
        MicroAPI::Mul(tempDstVreg, expIntVreg, expDecimalVreg, mask);
        MicroAPI::Select(tempDstVreg, vReg0, tempDstVreg, cmpNInfMask);
        MicroAPI::Select(tempDstVreg, vReg1, tempDstVreg, cmpInfMask);
        if constexpr (IsSameType<T, half>::value) {
            MicroAPI::Cast<T, float, castTraitF322F16>(dstVreg, tempDstVreg, mask);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstVreg, mask);
        } else {
            MicroAPI::DataCopy(dst + i * oneRepSize, tempDstVreg, mask);
        }
    }
}

template <typename T, uint8_t taylorExpandLevel, bool isReuseSource>
__aicore__ inline void ExpImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(dstLocal, "dstLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcLocal, "srcLocal", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", dstLocal, "dstLocal", "Exp");
    CheckCalCount(calCount, "calCount", srcLocal, "srcLocal", "Exp");
    static_assert((SupportType<T, half, float>(), "current data type is not supported on current device!"));

    if constexpr (taylorExpandLevel == 0) {
        Exp<T>(dstLocal, srcLocal, calCount);
        return;
    }
    __local_mem__ T* dst = (__local_mem__ T*)dstLocal.GetPhyAddr();
    __local_mem__ T* src = (__local_mem__ T*)srcLocal.GetPhyAddr();
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);
    __local_mem__ float* sharedTmpBufferAddr = (__local_mem__ float*)sharedTmpBuffer.GetPhyAddr();
    VF_CALL<ExpAPI::ExpCompute<T, taylorExpandLevel>>(dst, src, calCount, repeatTimes, sharedTmpBufferAddr);
}

template <typename T, uint8_t taylorExpandLevel, bool isReuseSource>
__aicore__ inline void ExpImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    ExpImpl<T, taylorExpandLevel, isReuseSource>(dstLocal, srcLocal, sharedTmpBuffer, calCount);
}

} // namespace ExpAPI
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATH_EXP_EXP_C310_IMPL_H
