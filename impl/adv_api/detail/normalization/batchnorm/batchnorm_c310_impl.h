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
 * \file batchnorm_check_c310.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_C310_IMPL_H
#define IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#ifdef ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_check/normalization/batchnorm/batchnorm_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
namespace BatchNormAPI {
constexpr int32_t oneRepSize = GetVecLen() / sizeof(float);

template <typename T>
__simd_callee__ inline void LoadDataWithT(
    __ubuf__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> srcOrigin;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcOrigin, src + offset);
        Cast<float, T, layoutZMrgZ>(dstReg, srcOrigin, mask);
    } else {
        MicroAPI::LoadAlign(dstReg, src + offset);
    }
}

template <typename T>
__simd_callee__ inline void LoadDataWithGammBeta(
    __ubuf__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> srcOrigin;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B16>(srcOrigin, src + offset);
        Cast<float, T, layoutZMrgZ>(dstReg, srcOrigin, mask);
    } else {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(dstReg, src + offset);
    }
}

template <typename T>
__simd_callee__ inline void SaveDataWithT(
    __ubuf__ T* dst, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> regT;
        MicroAPI::Cast<T, float, LayoutZMrgZRndRSatNS>(regT, srcReg, mask);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, regT, mask);
    } else {
        MicroAPI::StoreAlign(dst + offset, srcReg, mask);
    }
}

template <typename T>
__simd_callee__ inline void ComputeOutputMean(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, uint32_t oriBLength,
    uint32_t featureLength, float firstDimValueBack)
{
    MicroAPI::RegTensor<float> srcReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> dstTailReg;
    uint16_t mainRepeatTime = static_cast<uint16_t>(featureLength / oneRepSize);
    uint32_t tailCount = featureLength % oneRepSize;
    uint16_t tailRepeatTime = static_cast<uint16_t>(CeilDivision(tailCount, oneRepSize));
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float>(tailCount);
    for (uint16_t i = 0; i < mainRepeatTime; i++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithT(srcLocal, srcReg, maskFull, bIdx * featureLength + i * oneRepSize);
            // x / N
            MicroAPI::Muls(srcReg, srcReg, firstDimValueBack, maskFull);
            // ∑(x / N)
            MicroAPI::Add(dstReg, dstReg, srcReg, maskFull);
        }
        SaveDataWithT(dstLocal, dstReg, maskFull, i * oneRepSize);
    }
    for (uint16_t i = 0; i < tailRepeatTime; i++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithT(srcLocal, srcReg, maskReg, bIdx * featureLength + mainRepeatTime * oneRepSize);
            // x / N
            MicroAPI::Muls(srcReg, srcReg, firstDimValueBack, maskReg);
            // ∑(x / N)
            MicroAPI::Add(dstReg, dstReg, srcReg, maskReg);
        }
        SaveDataWithT(dstLocal, dstReg, maskReg, mainRepeatTime * oneRepSize);
    }
}

template <typename T>
__simd_callee__ inline void ComputeFloatMean(__ubuf__ float* dstLocal, __ubuf__ T* srcLocal, uint32_t oriBLength,
    uint32_t featureLength, float firstDimValueBack)
{
    MicroAPI::RegTensor<float> srcReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> dstTailReg;
    uint16_t mainRepeatTime = static_cast<uint16_t>(featureLength / oneRepSize);
    uint32_t tailCount = featureLength % oneRepSize;
    uint16_t tailRepeatTime = static_cast<uint16_t>(CeilDivision(tailCount, oneRepSize));
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float>(tailCount);
    for (uint16_t i = 0; i < mainRepeatTime; i++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithT(srcLocal, srcReg, maskFull, bIdx * featureLength + i * oneRepSize);
            // x / N
            MicroAPI::Muls(srcReg, srcReg, firstDimValueBack, maskFull);
            // ∑(x / N)
            MicroAPI::Add(dstReg, dstReg, srcReg, maskFull);
        }
        MicroAPI::StoreAlign(dstLocal + i * oneRepSize, dstReg, maskFull);
    }
    for (uint16_t i = 0; i < tailRepeatTime; i++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithT(srcLocal, srcReg, maskReg, bIdx * featureLength + mainRepeatTime * oneRepSize);
            // x / N
            MicroAPI::Muls(srcReg, srcReg, firstDimValueBack, maskReg);
            // ∑(x / N)
            MicroAPI::Add(dstReg, dstReg, srcReg, maskReg);
        }
        MicroAPI::StoreAlign(dstLocal + mainRepeatTime * oneRepSize, dstReg, maskReg);
    }
}

template <typename T>
__simd_callee__ inline void ComputeOutputVariance(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal,
    __ubuf__ float* meanLocal, uint32_t oriBLength, uint32_t featureLength, float firstDimValueBack)
{
    MicroAPI::RegTensor<float> srcReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> meanReg;
    MicroAPI::RegTensor<float> diffReg;
    MicroAPI::RegTensor<float> sqrReg;
    MicroAPI::RegTensor<float> dstTailReg;
    uint16_t mainRepeatTime = static_cast<uint16_t>(featureLength / oneRepSize);
    uint32_t tailCount = featureLength % oneRepSize;
    uint16_t tailRepeatTime = static_cast<uint16_t>(CeilDivision(tailCount, oneRepSize));
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float>(tailCount);
    for (uint16_t i = 0; i < mainRepeatTime; i++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        MicroAPI::LoadAlign(meanReg, meanLocal + i * oneRepSize);
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithT(srcLocal, srcReg, maskFull, bIdx * featureLength + i * oneRepSize);
            // step 1: x - u
            MicroAPI::Sub(diffReg, srcReg, meanReg, maskFull);
            // step 2: (x - u)²
            MicroAPI::Mul(sqrReg, diffReg, diffReg, maskFull);
            // step 3: ∑(x - u)²
            MicroAPI::Add(dstReg, dstReg, sqrReg, maskFull);
        }
        // step 4: ∑(x - u)² / N
        MicroAPI::Muls(dstReg, dstReg, firstDimValueBack, maskFull);
        SaveDataWithT(dstLocal, dstReg, maskFull, i * oneRepSize);
    }
    for (uint16_t i = 0; i < tailRepeatTime; i++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        MicroAPI::LoadAlign(meanReg, meanLocal + mainRepeatTime * oneRepSize);
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithT(srcLocal, srcReg, maskReg, bIdx * featureLength + mainRepeatTime * oneRepSize);
            // step 1: x - u
            MicroAPI::Sub(diffReg, srcReg, meanReg, maskReg);
            // step 2: (x - u)²
            MicroAPI::Mul(sqrReg, diffReg, diffReg, maskReg);
            // step 3: ∑(x - u)²
            MicroAPI::Add(dstReg, dstReg, sqrReg, maskReg);
        }
        // step 4: ∑(x - u)² / N
        MicroAPI::Muls(dstReg, dstReg, firstDimValueBack, maskReg);
        SaveDataWithT(dstLocal, dstReg, maskReg, mainRepeatTime * oneRepSize);
    }
}

template <typename T>
__simd_callee__ inline void ComputeFloatVariance(__ubuf__ float* dstLocal, __ubuf__ T* srcLocal,
    __ubuf__ float* meanLocal, uint32_t oriBLength, uint32_t featureLength, float firstDimValueBack)
{
    MicroAPI::RegTensor<float> srcReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> meanReg;
    MicroAPI::RegTensor<float> diffReg;
    MicroAPI::RegTensor<float> sqrReg;
    MicroAPI::RegTensor<float> dstTailReg;
    uint16_t mainRepeatTime = static_cast<uint16_t>(featureLength / oneRepSize);
    uint32_t tailCount = featureLength % oneRepSize;
    uint16_t tailRepeatTime = static_cast<uint16_t>(CeilDivision(tailCount, oneRepSize));
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float>(tailCount);
    for (uint16_t i = 0; i < mainRepeatTime; i++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        MicroAPI::LoadAlign(meanReg, meanLocal + i * oneRepSize);
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithT(srcLocal, srcReg, maskFull, bIdx * featureLength + i * oneRepSize);
            // step 1: x - u
            MicroAPI::Sub(diffReg, srcReg, meanReg, maskFull);
            // step 2: (x - u)²
            MicroAPI::Mul(sqrReg, diffReg, diffReg, maskFull);
            // step 3: ∑(x - u)²
            MicroAPI::Add(dstReg, dstReg, sqrReg, maskFull);
        }
        // step 4: ∑(x - u)² / N
        MicroAPI::Muls(dstReg, dstReg, firstDimValueBack, maskFull);
        MicroAPI::StoreAlign(dstLocal + i * oneRepSize, dstReg, maskFull);
    }
    for (uint16_t i = 0; i < tailRepeatTime; i++) {
        MicroAPI::Duplicate(dstReg, static_cast<float>(0), maskFull);
        MicroAPI::LoadAlign(meanReg, meanLocal + mainRepeatTime * oneRepSize);
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithT(srcLocal, srcReg, maskReg, bIdx * featureLength + mainRepeatTime * oneRepSize);
            // step 1: x - u
            MicroAPI::Sub(diffReg, srcReg, meanReg, maskReg);
            // step 2: (x - u)²
            MicroAPI::Mul(sqrReg, diffReg, diffReg, maskReg);
            // step 3: ∑(x - u)²
            MicroAPI::Add(dstReg, dstReg, sqrReg, maskReg);
        }
        // step 4: ∑(x - u)² / N
        MicroAPI::Muls(dstReg, dstReg, firstDimValueBack, maskReg);
        MicroAPI::StoreAlign(dstLocal + mainRepeatTime * oneRepSize, dstReg, maskReg);
    }
}

template <typename T>
__simd_callee__ inline void ComputeY(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ float* tmpMeanLocal,
    __ubuf__ float* tmpVarLocal, __ubuf__ T* gammLocal,  __ubuf__ T* betaLocal, uint32_t oriBLength,
    uint32_t featureLength, const float epsilon)
{
    constexpr float rsqrtExponent = -0.5;
    MicroAPI::RegTensor<float> srcReg;
    MicroAPI::RegTensor<float> meanReg;
    MicroAPI::RegTensor<float> varReg;
    MicroAPI::RegTensor<float> gammReg;
    MicroAPI::RegTensor<float> betaReg;
    MicroAPI::RegTensor<float> diffReg;
    uint16_t mainRepeatTime = static_cast<uint16_t>(featureLength / oneRepSize);
    uint32_t tailCount = featureLength % oneRepSize;
    uint16_t tailRepeatTime = static_cast<uint16_t>(CeilDivision(tailCount, oneRepSize));
    static constexpr MicroAPI::LnSpecificMode lnMode = {MicroAPI::MaskMergeMode::ZEROING, LnAlgo::INTRINSIC};
    static constexpr MicroAPI::ExpSpecificMode expMode = {MicroAPI::MaskMergeMode::ZEROING, ExpAlgo::INTRINSIC};
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float>(tailCount);
    for (uint16_t i = 0; i < mainRepeatTime; i++) {
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithGammBeta(gammLocal, gammReg, maskFull, bIdx);
            LoadDataWithGammBeta(betaLocal, betaReg, maskFull, bIdx);
            MicroAPI::LoadAlign(meanReg, tmpMeanLocal + i * oneRepSize);
            MicroAPI::LoadAlign(varReg, tmpVarLocal + i * oneRepSize);
            LoadDataWithT(srcLocal, srcReg, maskFull, bIdx * featureLength + i * oneRepSize);
            // var + e
            MicroAPI::Adds(varReg, varReg, epsilon, maskFull);
            // rsqrt: ln + muls + exp
            MicroAPI::Ln<float, &lnMode>(varReg, varReg, maskFull);
            MicroAPI::Muls(varReg, varReg, rsqrtExponent, maskFull);
            MicroAPI::Exp<float, &expMode>(varReg, varReg, maskFull);
            // rsqrt * (x - mean)
            MicroAPI::Sub(diffReg, srcReg, meanReg, maskFull);
            MicroAPI::Mul(varReg, varReg, diffReg, maskFull);
            // res * gamm + beta
            MicroAPI::Mul(varReg, varReg, gammReg, maskFull);
            MicroAPI::Add(varReg, varReg, betaReg, maskFull);
            SaveDataWithT(dstLocal, varReg, maskFull, bIdx * featureLength + i * oneRepSize);
        }
    }
    for (uint16_t i = 0; i < tailRepeatTime; i++) {
        for (uint16_t bIdx = 0; bIdx < oriBLength; bIdx++) {
            LoadDataWithGammBeta(gammLocal, gammReg, maskReg, bIdx);
            LoadDataWithGammBeta(betaLocal, betaReg, maskReg, bIdx);
            MicroAPI::LoadAlign(meanReg, tmpMeanLocal + mainRepeatTime * oneRepSize);
            MicroAPI::LoadAlign(varReg, tmpVarLocal + mainRepeatTime * oneRepSize);
            LoadDataWithT(srcLocal, srcReg, maskReg, bIdx * featureLength + mainRepeatTime * oneRepSize);
            // var + e
            MicroAPI::Adds(varReg, varReg, epsilon, maskReg);
            // rsqrt: ln + muls + exp
            MicroAPI::Ln<float, &lnMode>(varReg, varReg, maskReg);
            MicroAPI::Muls(varReg, varReg, rsqrtExponent, maskReg);
            MicroAPI::Exp<float, &expMode>(varReg, varReg, maskReg);
            // rsqrt * (x - mean)
            MicroAPI::Sub(diffReg, srcReg, meanReg, maskReg);
            MicroAPI::Mul(varReg, varReg, diffReg, maskReg);
            // res * gamm + beta
            MicroAPI::Mul(varReg, varReg, gammReg, maskReg);
            MicroAPI::Add(varReg, varReg, betaReg, maskReg);
            SaveDataWithT(dstLocal, varReg, maskReg, bIdx * featureLength + mainRepeatTime * oneRepSize);
        }
    }
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__simd_vf__ inline void BatchNormImplVF(__ubuf__ T* output, __ubuf__ T* outputMean,
    __ubuf__ T* outputVariance, __ubuf__ T* inputX, __ubuf__ T* gamm, __ubuf__ T* beta,
    __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal, const float epsilon,
    const BatchNormTiling tiling, uint32_t oriBLength, uint32_t featureLength, float firstDimValueBack)
{
    ComputeOutputMean(outputMean, inputX, oriBLength, featureLength, firstDimValueBack);
    ComputeFloatMean(tmpMeanLocal, inputX, oriBLength, featureLength, firstDimValueBack);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    ComputeOutputVariance(outputVariance, inputX, tmpMeanLocal, oriBLength, featureLength, firstDimValueBack);
    ComputeFloatVariance(tmpVarLocal, inputX, tmpMeanLocal, oriBLength, featureLength, firstDimValueBack);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    ComputeY(output, inputX, tmpMeanLocal, tmpVarLocal, gamm, beta, oriBLength, featureLength, epsilon);
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void BatchNormImpl(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX,const LocalTensor<T>& gamm,
    const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, BatchNormTiling& tiling)
{
    CHECK_FUNC_HIGHLEVEL_API(BatchNorm, (T, isReuseSource, isBasicBlock), (output, outputMean, outputVariance, inputX,
        gamm, beta, sharedTmpBuffer, epsilon, tiling));
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    uint32_t oriBLength = tiling.originalBLength;
    uint32_t featureLength = tiling.meanVarSize;
    float firstDimValueBack = tiling.firstDimValueBack;
    if constexpr (isBasicBlock) {
        ASCENDC_ASSERT((oriBLength % 8 == 0),
            {KERNEL_LOG(KERNEL_ERROR, "BatchNorm buffer size error: oriBLength is %u not a multiple of 8", oriBLength);});
        ASCENDC_ASSERT((featureLength % 64 == 0 && featureLength <= 2048),
            {KERNEL_LOG(KERNEL_ERROR, "BatchNorm buffer size error: current sLength * hLength is %u not a multiple of 64"
            "AND <= 2048.", featureLength);});
    }
    float epsilonFloat = static_cast<float>(epsilon);
    LocalTensor<float> tmpLocal = sharedTmpBuffer.ReinterpretCast<float>();
    LocalTensor<float> tmpMeanLocal = tmpLocal;

    LocalTensor<float> tmpVarLocal = tmpLocal[featureLength];
    BatchNormImplVF<T, isReuseSource, isBasicBlock>((__ubuf__ T*)output.GetPhyAddr(),
        (__ubuf__ T*)outputMean.GetPhyAddr(), (__ubuf__ T*)outputVariance.GetPhyAddr(),
        (__ubuf__ T*)inputX.GetPhyAddr(), (__ubuf__ T*)gamm.GetPhyAddr(), (__ubuf__ T*)beta.GetPhyAddr(),
        (__ubuf__ float*)tmpMeanLocal.GetPhyAddr(), (__ubuf__ float*)tmpVarLocal.GetPhyAddr(), epsilonFloat, tiling,
        oriBLength, featureLength, firstDimValueBack);
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void BatchNormImpl(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
    const LocalTensor<T>& beta, const T epsilon, BatchNormTiling& tiling)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "BatchNorm failed to apply for tmp buffer!");});
    BatchNormImpl<T, isReuseSource, isBasicBlock>(output, outputMean, outputVariance, inputX, gamm, beta,
        sharedTmpBuffer, epsilon, tiling);
}
} // namespace BatchNormAPI
} // namespace AscendC
#endif // IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_C310_IMPL_H