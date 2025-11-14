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
 * \file mean_c310_impl.h
 * \brief
 */
#ifndef IMPL_REDUCE_MEAN_C310_IMPL_H
#define IMPL_REDUCE_MEAN_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "include/adv_api/reduce/mean_utils.h"

namespace AscendC {
namespace Internal {

template <typename T, typename accType>
struct GetConvType {
    using type = T;
};

template <>
struct GetConvType<half, float> {
    using type = float;
};

template <typename T, typename ConvType>
__simd_callee__ inline void LoadSrcData(
    MicroAPI::RegTensor<ConvType>& srcReg, __ubuf__ T* src, uint16_t index, uint32_t offset, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> srcTmpReg;
    if constexpr (std::is_same<T, half>::value && std::is_same<ConvType, float>::value) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcTmpReg, src + index * offset);
        MicroAPI::Cast<float, T, castTraitB16ToB32>(srcReg, srcTmpReg, mask);
    } else {
        MicroAPI::DataCopy(srcReg, src + index * offset);
    }
}

template <typename T, typename U, typename ConvType>
__simd_vf__ inline void MeanForOneRepeatTime(
    __ubuf__ T* dstUb, __ubuf__ U* srcUb, const MeanParams meanParams, uint32_t calCount, uint32_t offset)
{
    uint32_t count;
    ConvType scalarValue = static_cast<ConvType>(1.0f / meanParams.n);
    MicroAPI::MaskReg mask;
    MicroAPI::UnalignReg uregOut;
    MicroAPI::RegTensor<ConvType> srcTmpReg, dstTmpReg;
    MicroAPI::RegTensor<T> dstReg;

    for (int i = 0; i < meanParams.outter; i++) {
        count = calCount;
        mask = MicroAPI::UpdateMask<ConvType>(count);
        LoadSrcData(srcTmpReg, srcUb, i, offset, mask);
        MicroAPI::ReduceSum(dstTmpReg, srcTmpReg, mask);
        MicroAPI::Muls(dstTmpReg, dstTmpReg, scalarValue, mask);
        if constexpr (sizeof(T) == sizeof(half) && sizeof(ConvType) == sizeof(float)) {
            MicroAPI::Cast<T, float, castTraitB32ToB16>(dstReg, dstTmpReg, mask);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)dstReg, (MicroAPI::RegTensor<uint32_t>&)dstReg);
        } else {
            dstReg = dstTmpReg;
        }
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUb, dstReg, uregOut, 1);
    }
    MicroAPI::DataCopyUnAlignPost(dstUb, uregOut, 0);
}

template <typename T, typename ConvType, bool isFirstRepeat>
__simd_vf__ inline void ReduceSumNextN(__ubuf__ ConvType* dstUb, __ubuf__ T* srcUb, const MeanParams meanParams,
    uint32_t calCount, uint32_t repeatTimes, uint32_t offset)
{
    uint32_t count;
    MicroAPI::MaskReg mask;
    MicroAPI::UnalignReg uregIn;
    MicroAPI::RegTensor<ConvType> srcReg, dstReg;
    constexpr int32_t eleCountPerVL = GetVecLen() / sizeof(ConvType);

    for (uint16_t i = 0; i < meanParams.outter; i++) {
        count = calCount;
        auto dstTmpUb = dstUb + i * offset;
        for (uint16_t j = 0; j < repeatTimes; j++) {
            mask = MicroAPI::UpdateMask<ConvType>(count);
            if constexpr (isFirstRepeat) {
                LoadSrcData(srcReg, srcUb + i * meanParams.inner, j, eleCountPerVL, mask);
            } else {
                LoadSrcData(srcReg, srcUb + i * offset, j, eleCountPerVL, mask);
            }
            MicroAPI::ReduceSum(dstReg, srcReg, mask);
            MicroAPI::DataCopyUnAlign<ConvType, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstTmpUb, dstReg, uregIn, 1);
        }
        MicroAPI::DataCopyUnAlignPost(dstTmpUb, uregIn, 0);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}
} // namespace Internal

template <typename T, typename accType, bool isReuseSource, bool isBasicBlock, int32_t reduceDim>
__aicore__ inline void MeanCheckParams(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const MeanParams& meanParams)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "Mean");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "Mean");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "Mean");
    constexpr uint32_t meanInnerAlignLen = 32;
    ASCENDC_ASSERT((1 <= meanParams.n) && (meanParams.n <= meanParams.inner), {
        KERNEL_LOG(KERNEL_ERROR, "The value of n must be greater than or equal to 1 and less than or equal to inner.");
    });
    ASCENDC_ASSERT((meanParams.inner * sizeof(T) % meanInnerAlignLen == 0),
        { KERNEL_LOG(KERNEL_ERROR, "The value of inner * sizeof(T) must be an integer multiple of 32."); });
}

template <typename T, typename accType = T, bool isReuseSource = false, bool isBasicBlock = false,
    int32_t reduceDim = -1>
__aicore__ inline void MeanImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const MeanParams& meanParams)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    MeanCheckParams<T, accType, isReuseSource, isBasicBlock, reduceDim>(
        dstTensor, srcTensor, sharedTmpBuffer, meanParams);

    using ConvType = typename Internal::GetConvType<T, accType>::type;
    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)srcTensor.GetPhyAddr();
    __local_mem__ ConvType* sharedTmpBufferUb = (__local_mem__ ConvType*)sharedTmpBuffer.GetPhyAddr();

    constexpr int32_t eleCountPerVL = GetVecLen() / sizeof(ConvType);
    uint16_t repeatTimes = CeilDivision(meanParams.n, eleCountPerVL);
    uint32_t loopRepeatTimes;
    uint32_t calCount = meanParams.n;

    uint32_t totalCnt = 1;
    uint32_t dataSize = repeatTimes;
    uint32_t offset = AlignUp(CeilDivision(meanParams.inner, eleCountPerVL), 32);
    while (dataSize > 1) {
        ++totalCnt;
        dataSize = CeilDivision(dataSize, eleCountPerVL);
    }

    if (repeatTimes == 1) {
        Internal::MeanForOneRepeatTime<T, T, ConvType>(dstUb, srcUb, meanParams, meanParams.n, meanParams.inner);
        return;
    }

    Internal::ReduceSumNextN<T, ConvType, true>(
        sharedTmpBufferUb, srcUb, meanParams, calCount, repeatTimes, offset);

    --totalCnt;
    loopRepeatTimes = repeatTimes;
    while (totalCnt != 0) {
        calCount = loopRepeatTimes;
        loopRepeatTimes = CeilDivision(loopRepeatTimes, eleCountPerVL);
        if (totalCnt == 1) {
            Internal::MeanForOneRepeatTime<T, ConvType, ConvType>(
                dstUb, sharedTmpBufferUb, meanParams, calCount, offset);
        } else {
            Internal::ReduceSumNextN<ConvType, ConvType, false>(
                sharedTmpBufferUb, sharedTmpBufferUb, meanParams, calCount, loopRepeatTimes, offset);
        }
        --totalCnt;
    }
}
} // namespace AscendC

#endif // IMPL_REDUCE_MEAN_C310_IMPL_H
