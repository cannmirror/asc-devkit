/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file simple_softmax_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SIMPLE_SOFTMAX_IMPL_H
#define AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SIMPLE_SOFTMAX_IMPL_H

#include "softmax_common_impl.h"

namespace AscendC {
template <typename T1, typename T2>
__aicore__ inline void SimpleSoftMaxGenericNZImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, const uint16_t mRepeatTimes, const uint16_t kRepeatTimes,
    const uint16_t outNum, const uint16_t dataBlock)
{
    MicroAPI::MaskReg maskCnt;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::RegTensor<float> maxVreg1;
    MicroAPI::RegTensor<float> maxVreg2;
    MicroAPI::RegTensor<float> sumVreg1;
    MicroAPI::RegTensor<float> sumVreg2;

    for (uint16_t j = 0; j < kRepeatTimes; ++j) {
        uint32_t sreg = outNum;
        for (uint16_t i = 0; i < mRepeatTimes; ++i) {
            maskCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadIfNeedCast<T2>(maxVreg, maxUb + i * FLOAT_REPEAT_SIZE, maskFull);
            LoadIfNeedCast<T2>(sumVreg, sumUb + i * FLOAT_REPEAT_SIZE, maskFull);
            if constexpr (SupportType<T2, float>()) {
                MicroAPI::Interleave(maxVreg1, maxVreg2, maxVreg, maxVreg);
                MicroAPI::Interleave(sumVreg1, sumVreg2, sumVreg, sumVreg);
                LoadIfNeedCast<T1>(srcVreg, srcUb + 2 * i * FLOAT_REPEAT_SIZE + j * dataBlock, maskFull);
                MicroAPI::Sub(dstVreg, srcVreg, maxVreg1, maskCnt);
                MicroAPI::Exp(tmpVreg, dstVreg, maskCnt);
                MicroAPI::Div(dstVreg, tmpVreg, sumVreg1, maskCnt);
                StoreIfNeedCast<T1>(dstUb + 2 * i * FLOAT_REPEAT_SIZE + j * dataBlock, dstVreg, maskCnt);
                maskCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
                LoadIfNeedCast<T1>(srcVreg, srcUb + (2 * i + 1) * FLOAT_REPEAT_SIZE + j * dataBlock, maskFull);
                MicroAPI::Sub(dstVreg, srcVreg, maxVreg2, maskCnt);
                MicroAPI::Exp(tmpVreg, dstVreg, maskCnt);
                MicroAPI::Div(dstVreg, tmpVreg, sumVreg2, maskCnt);
                StoreIfNeedCast<T1>(dstUb + (2 * i + 1) * FLOAT_REPEAT_SIZE + j * dataBlock, dstVreg, maskCnt);
            } else {
                LoadIfNeedCast<T1>(srcVreg, srcUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, maskFull);
                MicroAPI::Sub(dstVreg, srcVreg, maxVreg, maskCnt);
                MicroAPI::Exp(tmpVreg, dstVreg, maskCnt);
                MicroAPI::Div(dstVreg, tmpVreg, sumVreg, maskCnt);
                StoreIfNeedCast<T1>(dstUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, dstVreg, maskCnt);
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void SimpleSoftMaxNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<float> workLocal,
    const SoftMaxTiling& tiling, const LastAxisShapeND& originalSrcShape)
{
    static_assert((SupportType<Tuple<T1, T2>, Tuple<half, float>, Tuple<half, half>, Tuple<float, float>>()),
        "Failed to check dtype in SimpleSoftMax, current api "
        "support dtype combination is T1 : half, T2 : float; T1 : half, T2 : half; "
        "T1 : float, T2 : float");
    uint16_t srcM = tiling.srcM;
    uint16_t srcK = tiling.srcK;
    uint16_t oriM = originalSrcShape.m;
    constexpr uint16_t nzKUnitLen =
        IsSameType<T2, half>::value ? SOFTMAX_SHAPE_NZ_BASIC_COUNT : SOFTMAX_SHAPE_NZ_BASIC_COUNT / 2;
    uint16_t dataBlock = srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint16_t mRepeatTimes = static_cast<uint16_t>(CeilDivision(srcM * nzKUnitLen, FLOAT_REPEAT_SIZE));
    uint16_t kRepeatTimes = srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint32_t sreg = oriM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)inSumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)inMaxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();

    VF_CALL<SimpleSoftMaxGenericNZImpl<T1, T2>>(
        dstUb, sumUb, maxUb, srcUb, mRepeatTimes, kRepeatTimes, sreg, dataBlock);
}

template <typename T1, typename T2>
__aicore__ inline void SimpleSoftMaxGenericNDImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, const uint16_t srcM, const uint16_t srcK,
    const uint16_t repeatTimes, const uint16_t blockStride)
{
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;

    for (uint16_t i = 0; i < srcM; ++i) {
        LoadIfNeedCast<T2>(maxVreg, maxUb + i * blockStride, maskFull);
        LoadIfNeedCast<T2>(sumVreg, sumUb + i * blockStride, maskFull);
        MicroAPI::Duplicate(maxVreg, maxVreg, maskFull);
        MicroAPI::Duplicate(sumVreg, sumVreg, maskFull);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, maskFull);
            MicroAPI::Sub(dstVreg, srcVreg, maxVreg, maskFull);
            MicroAPI::Exp(tmpVreg, dstVreg, maskFull);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, maskFull);
            StoreIfNeedCast<T1>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, dstVreg, maskFull);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void SimpleSoftMaxGenericNDWithTailImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, const uint16_t srcM, const uint16_t srcK,
    const uint16_t repeatTimes, const uint16_t blockStride)
{
    MicroAPI::MaskReg maskCnt;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;

    for (uint16_t i = 0; i < srcM; ++i) {
        LoadIfNeedCast<T2>(maxVreg, maxUb + i * blockStride, maskFull);
        LoadIfNeedCast<T2>(sumVreg, sumUb + i * blockStride, maskFull);
        MicroAPI::Duplicate(maxVreg, maxVreg, maskFull);
        MicroAPI::Duplicate(sumVreg, sumVreg, maskFull);
        uint32_t sreg = srcK;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            maskCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, maskFull);
            MicroAPI::Sub(dstVreg, srcVreg, maxVreg, maskCnt);
            MicroAPI::Exp(tmpVreg, dstVreg, maskCnt);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, maskCnt);
            StoreIfNeedCast<T1>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, dstVreg, maskCnt);
        }
    }
}

template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxNDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<float> workLocal,
    const SoftMaxTiling& tiling)
{
    static_assert((SupportType<Tuple<T1, T2>, Tuple<half, float>, Tuple<half, half>, Tuple<float, float>>()),
        "Failed to check dtype in SimpleSoftMax, current api "
        "support dtype combination is T1 : half, T2 : float; T1 : half, T2 : half; "
        "T1 : float, T2 : float");
    uint16_t srcM = tiling.srcM;
    uint16_t srcK = tiling.srcK;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(srcK, FLOAT_REPEAT_SIZE));
    constexpr uint16_t blockStride = GetDataBlockSizeInBytes() / sizeof(T2);

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)inSumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)inMaxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();

    if constexpr (isBasicBlock) {
        VF_CALL<SimpleSoftMaxGenericNDImpl<T1, T2>>(dstUb, sumUb, maxUb, srcUb, srcM, srcK, repeatTimes, blockStride);
    } else {
        if constexpr (config.oriSrcM == 0 || config.oriSrcK == 0) {
            if (tiling.srcK % FLOAT_REPEAT_SIZE != 0) {
                VF_CALL<SimpleSoftMaxGenericNDWithTailImpl<T1, T2>>(
                    dstUb, sumUb, maxUb, srcUb, srcM, srcK, repeatTimes, blockStride);
            } else {
                VF_CALL<SimpleSoftMaxGenericNDImpl<T1, T2>>(
                    dstUb, sumUb, maxUb, srcUb, srcM, srcK, repeatTimes, blockStride);
            }
        } else if constexpr (config.oriSrcK % FLOAT_REPEAT_SIZE != 0) {
            VF_CALL<SimpleSoftMaxGenericNDWithTailImpl<T1, T2>>(
                dstUb, sumUb, maxUb, srcUb, srcM, srcK, repeatTimes, blockStride);
        } else {
            VF_CALL<SimpleSoftMaxGenericNDImpl<T1, T2>>(
                dstUb, sumUb, maxUb, srcUb, srcM, srcK, repeatTimes, blockStride);
        }
    }
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SIMPLE_SOFTMAX_IMPL_H
