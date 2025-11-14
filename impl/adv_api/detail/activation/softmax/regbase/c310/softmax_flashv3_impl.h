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

/* !
 * \file softmax_flashv3_c310_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_FLASHV3_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_FLASHV3_IMPL_H

#include "softmax_common_impl.h"

namespace AscendC {
template <typename T, typename U>
__simd_vf__ __aicore__ inline void SoftmaxFlashV3NDNoUpdateImpl(__local_mem__ T* dstUb,
    __local_mem__ U* meanUb, __local_mem__ U* expSumUb, __local_mem__ U* maxUb, __local_mem__ T* srcUb,
    __local_mem__ float* workUb, __local_mem__ float* newSrcUb, const uint16_t srcM, const uint16_t srcK,
    const uint16_t splitMeanCnt, const uint16_t baseK, const uint16_t tail, const uint16_t remainRepeatTime,
    const uint16_t kRepeatTime, const uint16_t baseKRepeatTime, const float scalar, const float r0, const float r1)
{
    constexpr uint32_t repeatStride = GetVecLen() / sizeof(float);
    constexpr uint32_t blockStride = GetDataBlockSizeInBytes() / sizeof(U);
    constexpr uint16_t repeatTime = static_cast<uint16_t>(repeatStride / blockStride);

    MicroAPI::MaskReg maskCnt;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOnePt = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> meanVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::RegTensor<float> minVreg;
    MicroAPI::RegTensor<T> castVreg;
    MicroAPI::UnalignReg ureg0, ureg1;
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    MicroAPI::Duplicate(minVreg, notNum.f);
    for (uint16_t i = 0; i < srcM; ++i) {
        for (uint16_t j = 0; j < repeatTime; ++j) {
            LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK + j * repeatStride, maskFull);
            MicroAPI::ReduceSumWithDataBlock(sumVreg, srcVreg, maskFull);
            MicroAPI::DataCopy<float>(workUb + i * repeatStride + j * blockStride, sumVreg, maskOneBlk);
        }
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::DataCopy<float>(sumVreg, workUb + i * repeatStride);
        for (uint16_t j = 0; j < remainRepeatTime; ++j) {
            LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK + repeatStride * splitMeanCnt + j * repeatStride, maskFull);
            MicroAPI::Add(sumVreg, srcVreg, sumVreg, maskFull);
        }
        MicroAPI::DataCopy<float>(workUb + i * repeatStride, sumVreg, maskFull);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::DataCopy<float>(sumVreg, workUb + i * repeatStride);
        MicroAPI::ReduceSumWithDataBlock(sumVreg, sumVreg, maskFull);
        MicroAPI::Muls(meanVreg, sumVreg, r0, maskOneBlk);

        MicroAPI::ReduceSum(tmpVreg, meanVreg, maskOneBlk);
        MicroAPI::Muls(tmpVreg, tmpVreg, r1, maskOnePt);
        MicroAPI::Duplicate(tmpVreg, tmpVreg, maskOneBlk);
        StoreIfNeedCast<U>(meanUb + i * blockStride, tmpVreg, maskOneBlk);
        MicroAPI::Sub(tmpVreg, tmpVreg, meanVreg, maskOneBlk);
        MicroAPI::Muls(tmpVreg, tmpVreg, scalar, maskOneBlk); // scalar = alpha / (1 - alpha)
        MicroAPI::DataCopy<float>(workUb + i * blockStride, tmpVreg, maskOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::Duplicate(maxVreg, notNum.f);
        for (uint16_t j = 0; j < splitMeanCnt; ++j) { // 8
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanVreg, workUb + i * splitMeanCnt + j);
            uint32_t sreg = baseK;
            for (uint16_t k = 0; k < baseKRepeatTime; ++k) { // baseK / 64
                maskCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
                __local_mem__ T *srcUbTmp = srcUb + i * srcK + j * baseK + k * repeatStride;
                MicroAPI::DataCopyUnAlignPre(ureg0, srcUbTmp);
                MicroAPI::DataCopyUnAlign(castVreg, ureg0, srcUbTmp, repeatStride);
                MicroAPI::UnPack<uint32_t, uint16_t>(
                    (MicroAPI::RegTensor<uint32_t>&)castVreg, (MicroAPI::RegTensor<uint16_t>&)castVreg);
                MicroAPI::Cast<float, T, Internal::castTraitB16ToB32>(srcVreg, castVreg, maskCnt);
                MicroAPI::Sub(srcVreg, srcVreg, meanVreg, maskCnt);
                __local_mem__ float *newSrcUbTmp = newSrcUb + i * srcK + j * baseK + k * repeatStride;
                MicroAPI::DataCopyUnAlign(newSrcUbTmp, srcVreg, ureg1, repeatStride);
                MicroAPI::DataCopyUnAlignPost(newSrcUbTmp, ureg1, 0);
                MicroAPI::Max(maxVreg, maxVreg, srcVreg, maskFull);
            }
            maskCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            __local_mem__ T *srcUbTmp = srcUb + i * srcK + j * baseK + baseKRepeatTime * repeatStride;
            MicroAPI::DataCopyUnAlignPre(ureg0, srcUbTmp);
            MicroAPI::DataCopyUnAlign(castVreg, ureg0, srcUbTmp, tail);
            MicroAPI::UnPack<uint32_t, uint16_t>(
                (MicroAPI::RegTensor<uint32_t>&)castVreg, (MicroAPI::RegTensor<uint16_t>&)castVreg);
            MicroAPI::Cast<float, T, Internal::castTraitB16ToB32>(srcVreg, castVreg, maskCnt);
            MicroAPI::Sub(srcVreg, srcVreg, meanVreg, maskCnt);
            __local_mem__ float *newSrcUbTmp = newSrcUb + i * srcK + j * baseK + baseKRepeatTime * repeatStride;
            MicroAPI::DataCopyUnAlign(newSrcUbTmp, srcVreg, ureg1, tail);
            MicroAPI::DataCopyUnAlignPost(newSrcUbTmp, ureg1, 0);
            MicroAPI::Select(srcVreg, srcVreg, minVreg, maskCnt);
            MicroAPI::Max(maxVreg, maxVreg, srcVreg, maskFull);
        }
        MicroAPI::ReduceMax(maxVreg, maxVreg, maskFull);
        MicroAPI::Duplicate(maxVreg, maxVreg, maskOneBlk);
        StoreIfNeedCast<U>(maxUb + i * blockStride, maxVreg, maskOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg, maxUb + i * blockStride);
        MicroAPI::Duplicate(sumVreg, 0);
        for (uint16_t k = 0; k < kRepeatTime; ++k) { // k / 64
            MicroAPI::DataCopy<float>(srcVreg, newSrcUb + i * srcK + k * repeatStride);
            MicroAPI::FusedExpSub(dstVreg, srcVreg, maxVreg, maskFull);
            StoreIfNeedCast<T>(dstUb + i * srcK + k * repeatStride, dstVreg, maskFull);
            MicroAPI::Add(sumVreg, sumVreg, dstVreg, maskFull);
        }
        MicroAPI::ReduceSum(sumVreg, sumVreg, maskFull);
        MicroAPI::Duplicate(sumVreg, sumVreg, maskOneBlk);
        StoreIfNeedCast<U>(expSumUb + i * blockStride, sumVreg, maskOneBlk);
    }
}

template <typename T, typename U>
__simd_vf__ __aicore__ inline void SoftmaxFlashV3NDUpdateImpl(__local_mem__ T* dstUb,
    __local_mem__ U* meanUb, __local_mem__ U* expSumUb, __local_mem__ U* maxUb,
    __local_mem__ T* srcUb, __local_mem__ T* expMaxUb, __local_mem__ U* inMeanUb,
    __local_mem__ U* inExpSumUb, __local_mem__ U* inMaxUb, __local_mem__ float* workUb,
    __local_mem__ float* newSrcUb, __local_mem__ float* tmpUb, const uint16_t srcM,
    const uint16_t srcK, const uint16_t splitMeanCnt, const uint16_t baseK, const uint16_t tail,
    const uint16_t remainRepeatTime, const uint16_t kRepeatTime, const uint16_t baseKRepeatTime,
    const uint32_t loopCnt, const float scalar, const float r0, const float r1,
    const float r2, const float r3)
{
    constexpr uint32_t repeatStride = GetVecLen() / sizeof(float);
    constexpr uint32_t blockStride = GetDataBlockSizeInBytes() / sizeof(U);
    constexpr uint16_t repeatTime = static_cast<uint16_t>(repeatStride / blockStride);
    
    MicroAPI::MaskReg maskCnt;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOnePt = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    MicroAPI::MaskReg maskOut = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL16>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> meanVreg;
    MicroAPI::RegTensor<float> inputVreg;
    MicroAPI::RegTensor<float> shiftVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::RegTensor<float> minVreg;
    MicroAPI::RegTensor<T> castVreg;
    MicroAPI::UnalignReg ureg0, ureg1;
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    MicroAPI::Duplicate(minVreg, notNum.f);
    for (uint16_t i = 0; i < srcM; ++i) {
        for (uint16_t j = 0; j < repeatTime; ++j) {
            LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK + j * repeatStride, maskFull);
            MicroAPI::ReduceSumWithDataBlock(sumVreg, srcVreg, maskFull);
            MicroAPI::DataCopy<float>(workUb + i * repeatStride + j * blockStride, sumVreg, maskOneBlk);
        }
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::DataCopy<float>(sumVreg, workUb + i * repeatStride);
        for (uint16_t j = 0; j < remainRepeatTime; ++j) {
            LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK + repeatStride * splitMeanCnt + j * repeatStride, maskFull);
            MicroAPI::Add(sumVreg, srcVreg, sumVreg, maskFull);
        }
        MicroAPI::DataCopy<float>(workUb + i * repeatStride, sumVreg, maskFull);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::DataCopy<float>(sumVreg, workUb + i * repeatStride);
        MicroAPI::ReduceSumWithDataBlock(sumVreg, sumVreg, maskFull);
        MicroAPI::Muls(meanVreg, sumVreg, r0, maskOneBlk);

        MicroAPI::ReduceSum(tmpVreg, meanVreg, maskOneBlk);
        MicroAPI::Muls(tmpVreg, tmpVreg, r1, maskOnePt);
        MicroAPI::Duplicate(tmpVreg, tmpVreg, maskOneBlk);
        MicroAPI::DataCopy<float>(tmpUb + i * blockStride, tmpVreg, maskOneBlk);
        MicroAPI::Sub(tmpVreg, tmpVreg, meanVreg, maskOneBlk);
        MicroAPI::Muls(tmpVreg, tmpVreg, scalar, maskOneBlk); // scalar = alpha / (1 - alpha)
        MicroAPI::DataCopy<float>(workUb + i * blockStride, tmpVreg, maskOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        LoadIfNeedCast<U>(inputVreg, inMeanUb + i * blockStride, maskOneBlk);
        MicroAPI::DataCopy<float>(tmpVreg, tmpUb + i * blockStride);
        MicroAPI::Muls(shiftVreg, inputVreg, r2, maskOneBlk);
        MicroAPI::Add(shiftVreg, shiftVreg, tmpVreg, maskOneBlk);
        MicroAPI::Muls(shiftVreg, shiftVreg, r3, maskOneBlk);
        StoreIfNeedCast<U>(meanUb + i * blockStride, shiftVreg, maskOneBlk);

        MicroAPI::Duplicate(maxVreg, notNum.f);
        for (uint16_t j = 0; j < splitMeanCnt; ++j) { // 8
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanVreg, workUb + i * splitMeanCnt + j);
            uint32_t sreg = baseK;
            for (uint16_t k = 0; k < baseKRepeatTime; ++k) { // baseK / 64
                maskCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
                __local_mem__ T *srcUbTmp = srcUb + i * srcK + j * baseK + k * repeatStride;
                MicroAPI::DataCopyUnAlignPre(ureg0, srcUbTmp);
                MicroAPI::DataCopyUnAlign(castVreg, ureg0, srcUbTmp, repeatStride);
                MicroAPI::UnPack<uint32_t, uint16_t>(
                    (MicroAPI::RegTensor<uint32_t>&)castVreg, (MicroAPI::RegTensor<uint16_t>&)castVreg);
                MicroAPI::Cast<float, T, Internal::castTraitB16ToB32>(srcVreg, castVreg, maskCnt);
                MicroAPI::Sub(srcVreg, srcVreg, meanVreg, maskCnt);
                __local_mem__ float *newSrcUbTmp = newSrcUb + i * srcK + j * baseK + k * repeatStride;
                MicroAPI::DataCopyUnAlign(newSrcUbTmp, srcVreg, ureg1, repeatStride);
                MicroAPI::DataCopyUnAlignPost(newSrcUbTmp, ureg1, 0);
                MicroAPI::Max(maxVreg, maxVreg, srcVreg, maskFull);
            }
            maskCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            __local_mem__ T *srcUbTmp = srcUb + i * srcK + j * baseK + baseKRepeatTime * repeatStride;
            MicroAPI::DataCopyUnAlignPre(ureg0, srcUbTmp);
            MicroAPI::DataCopyUnAlign(castVreg, ureg0, srcUbTmp, tail);
            MicroAPI::UnPack<uint32_t, uint16_t>(
                (MicroAPI::RegTensor<uint32_t>&)castVreg, (MicroAPI::RegTensor<uint16_t>&)castVreg);
            MicroAPI::Cast<float, T, Internal::castTraitB16ToB32>(srcVreg, castVreg, maskCnt);
            MicroAPI::Sub(srcVreg, srcVreg, meanVreg, maskCnt);
            __local_mem__ float *newSrcUbTmp = newSrcUb + i * srcK + j * baseK + baseKRepeatTime * repeatStride;
            MicroAPI::DataCopyUnAlign(newSrcUbTmp, srcVreg, ureg1, tail);
            MicroAPI::DataCopyUnAlignPost(newSrcUbTmp, ureg1, 0);
            MicroAPI::Select(srcVreg, srcVreg, minVreg, maskCnt);
            MicroAPI::Max(maxVreg, maxVreg, srcVreg, maskFull);
        }
        MicroAPI::ReduceMax(maxVreg, maxVreg, maskFull);
        MicroAPI::Duplicate(maxVreg, maxVreg, maskOneBlk);

        MicroAPI::Sub(dstVreg, tmpVreg, shiftVreg, maskOneBlk);
        MicroAPI::Muls(dstVreg, dstVreg, scalar, maskOneBlk); // scalar = alpha / (1 - alpha)
        MicroAPI::Sub(tmpVreg, inputVreg, shiftVreg, maskOneBlk);
        MicroAPI::Muls(tmpVreg, tmpVreg, scalar, maskOneBlk); // scalar = alpha / (1 - alpha)
        MicroAPI::Add(maxVreg, dstVreg, maxVreg, maskOneBlk);
        LoadIfNeedCast<U>(inputVreg, inMaxUb + i * blockStride, maskOneBlk);
        MicroAPI::Add(tmpVreg, inputVreg, tmpVreg, maskOneBlk);
        MicroAPI::Max(maxVreg, tmpVreg, maxVreg, maskOneBlk);
        StoreIfNeedCast<U>(maxUb + i * blockStride, maxVreg, maskOneBlk);
        MicroAPI::Sub(maxVreg, maxVreg, dstVreg, maskOneBlk);
        MicroAPI::DataCopy<float>(tmpUb + i * blockStride, maxVreg, maskOneBlk);
        MicroAPI::FusedExpSub(tmpVreg, tmpVreg, maxVreg, maskFull);
        LoadIfNeedCast<U>(inputVreg, inExpSumUb + i * blockStride, maskOneBlk);
        MicroAPI::Mul(sumVreg, tmpVreg, inputVreg, maskOneBlk);
        MicroAPI::DataCopy<float>(expSumUb + i * blockStride, sumVreg, maskOneBlk);
        MicroAPI::Interleave(tmpVreg, dstVreg, tmpVreg, tmpVreg);
        StoreIfNeedCast<T>(expMaxUb + i * blockStride * 2, tmpVreg, maskOut);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::Duplicate(sumVreg, 0);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg, tmpUb + i * blockStride);
        for (uint16_t k = 0; k < kRepeatTime; ++k) { // k / 64
            MicroAPI::DataCopy<float>(srcVreg, newSrcUb + i * srcK + k * repeatStride);
            MicroAPI::FusedExpSub(dstVreg, srcVreg, maxVreg, maskFull);
            StoreIfNeedCast<T>(dstUb + i * srcK + k * repeatStride, dstVreg, maskFull);
            MicroAPI::Add(sumVreg, sumVreg, dstVreg, maskFull);
        }
        MicroAPI::ReduceSum(sumVreg, sumVreg, maskFull);
        MicroAPI::Duplicate(sumVreg, sumVreg, maskOneBlk);
        MicroAPI::DataCopy<float>(tmpVreg, expSumUb + i * blockStride);
        MicroAPI::Add(sumVreg, tmpVreg, sumVreg, maskOneBlk);
        StoreIfNeedCast<U>(expSumUb + i * blockStride, sumVreg, maskOneBlk);
    }
}

template <typename T, typename U, bool isUpdate = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV3Process(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor,
    const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inExpSumTensor,
    const LocalTensor<U>& inMaxTensor, const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape,
    const SoftMaxTiling& tiling, const SoftMaxParams& params)
{
    constexpr uint16_t repeatStride = GetVecLen() / sizeof(float);
    uint16_t srcM = tiling.srcM;
    uint16_t srcK = tiling.srcK;
    uint16_t splitMeanCnt = static_cast<uint16_t>(params.splitMeanCnt);
    uint16_t baseK = static_cast<uint16_t>(srcK / splitMeanCnt);
    uint16_t kRepeatTime = static_cast<uint16_t>(srcK / repeatStride);
    uint16_t remainRepeatTime = kRepeatTime - splitMeanCnt;
    uint16_t baseKRepeatTime = CeilDivision(baseK, repeatStride) - 1;
    uint16_t tail = baseK - baseKRepeatTime * repeatStride;
    uint32_t loopCnt = params.loopCnt;
    float scalar = params.alpha / (1 - params.alpha);
    float r0 = static_cast<float>(1.0f / baseK);
    float r1 = static_cast<float>(1.0f / splitMeanCnt);

    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    __local_mem__ U* meanUb = (__local_mem__ U*)meanTensor.GetPhyAddr();
    __local_mem__ U* inMeanUb = (__local_mem__ U*)inMeanTensor.GetPhyAddr();
    __local_mem__ U* expSumUb = (__local_mem__ U*)expSumTensor.GetPhyAddr();
    __local_mem__ U* inExpSumUb = (__local_mem__ U*)inExpSumTensor.GetPhyAddr();
    __local_mem__ U* maxUb = (__local_mem__ U*)maxTensor.GetPhyAddr();
    __local_mem__ U* inMaxUb = (__local_mem__ U*)inMaxTensor.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)srcTensor.GetPhyAddr();
    __local_mem__ T* expMaxUb = (__local_mem__ T*)expMaxTensor.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* newSrcUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * repeatStride);

    if constexpr (!isUpdate) {
        SoftmaxFlashV3NDNoUpdateImpl<T, U>(dstUb, meanUb, expSumUb, maxUb, srcUb, workUb, newSrcUb,
            srcM, srcK, splitMeanCnt, baseK, tail, remainRepeatTime, kRepeatTime, baseKRepeatTime, scalar, r0, r1);
    } else {
        float r2 = static_cast<float>(loopCnt - 1.0f);
        float r3 = static_cast<float>(1.0f / loopCnt);
        __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * repeatStride + srcM * srcK);
        SoftmaxFlashV3NDUpdateImpl<T, U>(dstUb, meanUb, expSumUb, maxUb, srcUb, expMaxUb,
                inMeanUb, inExpSumUb, inMaxUb, workUb, newSrcUb, tmpUb, srcM, srcK, splitMeanCnt, baseK,
                tail, remainRepeatTime, kRepeatTime, baseKRepeatTime, loopCnt, scalar, r0, r1, r2, r3);
    }
}
} // namespace AscendC
#endif // IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_FLASHV3_IMPL_H
