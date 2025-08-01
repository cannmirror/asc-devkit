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
 * \file sum_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_REDUCE_SUM_SUM_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_REDUCE_SUM_SUM_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "reduce/sum_utils.h"

namespace AscendC {
namespace SumAPI {
template <typename T>
__aicore__ inline void SumCoreImpl(__ubuf__ T* dstUb, __ubuf__ T* srcUb, const SumParams& sumParams)
{
    MicroAPI::MaskReg mask;
    MicroAPI::UnalignReg ureg;
    MicroAPI::RegTensor<T> srcReg, dstReg, zeroReg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T>();

    uint32_t calCount;
    uint32_t rLength = sumParams.outter;
    uint32_t hLength = sumParams.inner;
    constexpr int32_t eleCountPerVL = GetVecLen() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(sumParams.n, eleCountPerVL);
    for (uint16_t i = 0; i < sumParams.outter; i++) {
        calCount = sumParams.n;
        MicroAPI::Duplicate(zeroReg, 0, fullMask);
        for (uint16_t j = 0; j < repeatTimes; j++) {
            mask = MicroAPI::UpdateMask<T>(calCount);
            MicroAPI::DataCopy(srcReg, srcUb + i * hLength + j * eleCountPerVL);
            MicroAPI::ReduceSum(dstReg, srcReg, mask);
            MicroAPI::Add(zeroReg, zeroReg, dstReg, mask);
        }
        MicroAPI::DataCopyUnAlign(dstUb, zeroReg, ureg, 1);
    }
    MicroAPI::DataCopyUnAlignPost(dstUb, ureg, 0);
}
} // namespace SumAPI

template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SumCheckParams(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SumParams& sumParams)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "Sum");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "Sum");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "Sum");
    constexpr uint32_t sumInnerAlignLen = 32;
    ASCENDC_ASSERT((1 <= sumParams.n) && (sumParams.n <= sumParams.inner), {
        KERNEL_LOG(KERNEL_ERROR, "The value of n must be greater than or equal to 1 and less than or equal to inner.");
    });
    ASCENDC_ASSERT((sumParams.inner * sizeof(T) % sumInnerAlignLen == 0),
        { KERNEL_LOG(KERNEL_ERROR, "The value of inner * sizeof(T) must be an integer multiple of 32."); });
}

template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SumCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SumParams& sumParams)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    SumCheckParams<T, reduceDim, isReuseSource, isBasicBlock>(dstTensor, srcTensor, sharedTmpBuffer, sumParams);
    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)srcTensor.GetPhyAddr();
    VF_CALL<SumAPI::SumCoreImpl<T>>(dstUb, srcUb, sumParams);
}
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_REDUCE_SUM_SUM_C310_IMPL_H
