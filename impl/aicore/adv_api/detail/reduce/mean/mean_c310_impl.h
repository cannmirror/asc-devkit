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
 * \file mean_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_REDUCE_MEAN_MEAN_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_REDUCE_MEAN_MEAN_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "reduce/mean_utils.h"

namespace AscendC {
namespace MeanAPI {
constexpr MicroAPI::CastTrait castTraitF16F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait castTraitF32F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T, typename accType>
struct GetConvType {
    using type = T;
};

template <>
struct GetConvType<half, float> {
    using type = float;
};

template <typename T, typename ConvType>
__aicore__ inline void LoadSrcData(
    MicroAPI::RegTensor<ConvType>& srcReg, __ubuf__ T* src, uint16_t index, MicroAPI::MaskReg& mask)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(ConvType);
    MicroAPI::RegTensor<T> srcTmpReg;
    if constexpr (std::is_same<T, half>::value && std::is_same<ConvType, float>::value) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcTmpReg, src + index * eleCountPerVL);
        MicroAPI::Cast<float, T, castTraitF16F32>(srcReg, srcTmpReg, mask);
    } else {
        MicroAPI::DataCopy(srcReg, src + index * eleCountPerVL);
    }
}

template <typename T, typename ConvType>
__aicore__ inline void MeanCoreImpl(__ubuf__ T* dstUb, __ubuf__ T* srcUb, const MeanParams& meanParams)
{
    MicroAPI::MaskReg mask;
    MicroAPI::UnalignReg ureg;
    MicroAPI::RegTensor<ConvType> srcReg, tmpReg;
    MicroAPI::RegTensor<ConvType> zeroReg;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T>();

    uint32_t calCount;
    uint32_t hLength = meanParams.inner;
    ConvType scalarValue = static_cast<ConvType>(1.0f / meanParams.n);
    constexpr int32_t eleCountPerVL = GetVecLen() / sizeof(ConvType);
    uint16_t repeatTimes = CeilDivision(meanParams.n, eleCountPerVL);

    for (uint16_t i = 0; i < meanParams.outter; i++) {
        calCount = meanParams.n;
        MicroAPI::Duplicate(zeroReg, 0, fullMask);
        for (uint16_t j = 0; j < repeatTimes; j++) {
            mask = MicroAPI::UpdateMask<ConvType>(calCount);
            LoadSrcData<T, ConvType>(srcReg, srcUb + i * hLength, j, mask);
            MicroAPI::ReduceSum(tmpReg, srcReg, mask);
            MicroAPI::Add(zeroReg, zeroReg, tmpReg, mask);
        }
        MicroAPI::Muls(zeroReg, zeroReg, scalarValue, mask);
        if constexpr (sizeof(T) == sizeof(half) && sizeof(ConvType) == sizeof(float)) {
            MicroAPI::Cast<T, float, castTraitF32F16>(dstReg, zeroReg, fullMask);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)dstReg, (MicroAPI::RegTensor<uint32_t>&)dstReg);
        } else {
            dstReg = zeroReg;
        }
        MicroAPI::DataCopyUnAlign(dstUb, dstReg, ureg, 1);
    }
    MicroAPI::DataCopyUnAlignPost(dstUb, ureg, 0);
}
} // namespace MeanAPI

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

    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)srcTensor.GetPhyAddr();
    using ConvType = typename MeanAPI::GetConvType<T, accType>::type;
    VF_CALL<MeanAPI::MeanCoreImpl<T, ConvType>>(dstUb, srcUb, meanParams);
}
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_REDUCE_MEAN_MEAN_C310_IMPL_H
