/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef IMPL_NORMALIZATION_RMSNORM_RMSNORM_C310_IMPL_H
#define IMPL_NORMALIZATION_RMSNORM_RMSNORM_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
namespace RmsNormAPI {
constexpr int32_t oneRepSize = GetVecLen() / sizeof(float);

template <typename T>
__simd_callee__ inline void LoadDataWithT(
    __ubuf__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> srcOrigin;
        DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcOrigin, src + offset);
        Cast<float, T, layoutZMrgZ>(dstReg, srcOrigin, mask);
    } else {
        DataCopy(dstReg, src + offset);
    }
}

template <typename T>
__simd_callee__ inline void SaveDataWithT(
    __ubuf__ T* dst, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> regT;
        MicroAPI::Cast<T, float, LayoutZMrgZRndRSatNS>(regT, srcReg, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, regT, mask);
    } else {
        MicroAPI::DataCopy(dst + offset, srcReg, mask);
    }
}

template <typename T>
__simd_callee__ inline void ComputeSum(__ubuf__ float* dstLocal, __ubuf__ T* srcLocal,
    uint32_t bsLength, uint32_t hLength, uint32_t oriHLength)
{
    uint16_t mainRepeatTime = static_cast<uint16_t>(oriHLength / oneRepSize);
    uint32_t tailCount = oriHLength % oneRepSize;
    uint16_t tailRepeatTime = static_cast<uint16_t>(CeilDivision(tailCount, oneRepSize));
    MicroAPI::RegTensor<float> srcReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> dstTailReg;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float>(tailCount);
    for (uint16_t bsIdx = 0; bsIdx < bsLength; bsIdx++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        for (uint16_t i = 0; i < mainRepeatTime; i++) {
            LoadDataWithT(srcLocal, srcReg, maskFull, bsIdx * hLength + i * oneRepSize);
            // step 1: x²
            MicroAPI::Mul(srcReg, srcReg, srcReg, maskFull);
            // step 2: ∑x²
            MicroAPI::Add(dstReg, dstReg, srcReg, maskFull);
        }
        for (uint16_t i = 0; i < tailRepeatTime; i++) {
            LoadDataWithT(srcLocal, srcReg, maskReg, bsIdx * hLength + mainRepeatTime * oneRepSize);
            // step 1: x²
            MicroAPI::Mul(srcReg, srcReg, srcReg, maskReg);
            // step 2: ∑x²
            MicroAPI::Add(dstTailReg, dstReg, srcReg, maskReg);
            MicroAPI::Select(dstReg, dstTailReg, dstReg, maskReg);
        }
        MicroAPI::ReduceSum(dstReg, dstReg, maskFull);
        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dstLocal + bsIdx, dstReg, maskOne);
    }
}
template <typename T>
__simd_callee__ inline void ComputeY(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* gammaLocal, __ubuf__ float* tmpLocal,
    uint32_t bsLength, uint32_t hLength, uint32_t oriHLength, const float epsilon, float reciprocalOfHLength)
{
    constexpr float rsqrtExponent = -0.5;
    MicroAPI::RegTensor<float> srcReg;
    MicroAPI::RegTensor<float> src2Reg;
    MicroAPI::RegTensor<float> gammaReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> dstTailReg;

    static constexpr MicroAPI::LnSpecificMode lnMode = {MicroAPI::MaskMergeMode::ZEROING, LnAlgo::INTRINSIC};
    static constexpr MicroAPI::ExpSpecificMode expMode = {MicroAPI::MaskMergeMode::ZEROING, ExpAlgo::INTRINSIC};
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
    uint16_t mainRepeatTime = static_cast<uint16_t>(oriHLength / oneRepSize);
    uint32_t tailCount = oriHLength % oneRepSize;
    uint16_t tailRepeatTime = static_cast<uint16_t>(CeilDivision(tailCount, oneRepSize));
    MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float>(tailCount);
    for (uint16_t bsIdx = 0; bsIdx < bsLength; bsIdx++) {
        for (uint16_t i = 0; i < mainRepeatTime; i++) {
            LoadDataWithT(srcLocal, srcReg, maskFull, bsIdx * hLength + i * oneRepSize);
            LoadDataWithT(gammaLocal, gammaReg, maskFull, i * oneRepSize);
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(src2Reg, tmpLocal + bsIdx);
            // step 3: rms = 1/n*∑
            MicroAPI::Muls(src2Reg, src2Reg, reciprocalOfHLength, maskFull);
            // step 4: rms + e
            MicroAPI::Adds(src2Reg, src2Reg, epsilon, maskFull);
            // step 5: rsqrt: ln + muls + exp
            MicroAPI::Ln<float, &lnMode>(src2Reg, src2Reg, maskFull);
            MicroAPI::Muls(src2Reg, src2Reg, rsqrtExponent, maskFull);
            MicroAPI::Exp<float, &expMode>(src2Reg, src2Reg, maskFull);
            // step 6: rms = xi * rsqrt
            MicroAPI::Mul(src2Reg, srcReg, src2Reg, maskFull);
            // step 7: rms = rms * gamma
            MicroAPI::Mul(src2Reg, src2Reg, gammaReg, maskFull);
            // save
            SaveDataWithT(dstLocal, src2Reg, maskFull, bsIdx * hLength + i * oneRepSize);
        }
        for (uint16_t i = 0; i < tailRepeatTime; i++) {
            LoadDataWithT(srcLocal, srcReg, maskReg, bsIdx * hLength + mainRepeatTime * oneRepSize);
            LoadDataWithT(gammaLocal, gammaReg, maskReg, mainRepeatTime * oneRepSize);
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(src2Reg, tmpLocal + bsIdx);
            // step 3: rms = 1/n*∑
            MicroAPI::Muls(src2Reg, src2Reg, reciprocalOfHLength, maskReg);
            // step 4: rms + e
            MicroAPI::Adds(src2Reg, src2Reg, epsilon, maskReg);
            // step 5: rsqrt: ln + muls + exp
            MicroAPI::Ln<float, &lnMode>(src2Reg, src2Reg, maskReg);
            MicroAPI::Muls(src2Reg, src2Reg, rsqrtExponent, maskReg);
            MicroAPI::Exp<float, &expMode>(src2Reg, src2Reg, maskReg);
            // step 6: rms = xi * rsqrt
            MicroAPI::Mul(src2Reg, srcReg, src2Reg, maskReg);
            // step 7: rms = rms * gamma
            MicroAPI::Mul(src2Reg, src2Reg, gammaReg, maskReg);
            // save
            SaveDataWithT(dstLocal, src2Reg, maskReg, bsIdx * hLength + mainRepeatTime * oneRepSize);
        }
    }
}

template <typename T>
__simd_vf__ inline void RmsNormImplVf(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal,
    __ubuf__ T* gammaLocal, __ubuf__ float* tmpLocal, const float epsilon, const RmsNormTiling tiling)
{
    uint32_t bLength = tiling.bLength;
    uint32_t sLength = tiling.sLength;
    uint32_t hLength = tiling.hLength;
    uint32_t oriHLength = tiling.originalHLength;
    float reciprocalOfHLength = tiling.reciprocalOfHLength;
    uint16_t loopRound = static_cast<uint16_t>(tiling.loopRound);
    uint32_t mainBsLength = tiling.mainBsLength;
    uint32_t mainBshLength = tiling.mainBshLength;

    for (uint16_t i = 0; i < loopRound; i++) {
        ComputeSum(tmpLocal, srcLocal + i * mainBshLength, mainBsLength, hLength, oriHLength);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        ComputeY(dstLocal + i * mainBshLength, srcLocal + i * mainBshLength, gammaLocal, tmpLocal, mainBsLength, hLength, oriHLength, epsilon, reciprocalOfHLength); 
    }
    uint32_t inputTailPos = tiling.inputTailPos;
    uint32_t tailBsLength = tiling.tailBsLength;
    uint16_t tailRound = static_cast<uint16_t>(CeilDivision(tailBsLength, mainBsLength));
    for (uint16_t i = 0; i < tailRound; i++) {
        ComputeSum(tmpLocal, srcLocal + inputTailPos, tailBsLength, hLength, oriHLength);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        ComputeY(dstLocal + inputTailPos, srcLocal + inputTailPos, gammaLocal, tmpLocal, tailBsLength, hLength, oriHLength, epsilon, reciprocalOfHLength); 
    }
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void RmsNormImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon,
    const RmsNormTiling& tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CHECK_FUNC_HIGHLEVEL_API(RmsNorm, (T, isBasicBlock), (dstLocal, srcLocal, gammaLocal, sharedTmpBuffer, epsilon, tiling));
    LocalTensor<float> tmpLocal = sharedTmpBuffer.ReinterpretCast<float>();
    float eps = static_cast<float>(epsilon);
    RmsNormImplVf<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)gammaLocal.GetPhyAddr(), (__ubuf__ float*)tmpLocal.GetPhyAddr(), eps, tiling);
}
} // namespace RmsNormAPI
} // namespace AscendC
#endif // IMPL_NORMALIZATION_RMSNORM_RMSNORM_C310_IMPL_H