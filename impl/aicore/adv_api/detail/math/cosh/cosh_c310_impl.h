/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cosh_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_COSH_COSH_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_COSH_COSH_C310_IMPL_H

#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
namespace CoshInternal {
// Computes cosh values based on input types.
// According formula: cosh(x) = (e^x + e^(-x))/2 = e^(x-ln2) + 0.25/(e^(x-ln2)).
template <typename T>
__aicore__ inline void CoshCompute(__ubuf__ T* dstUb, __ubuf__ T* srcUb, uint32_t calCount, uint16_t repeatTimes)
{
    constexpr float scalarNegLnTwo = -0.6931472;
    constexpr float scalarBrc = 0.25;
    constexpr uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));
    static constexpr MicroAPI::CastTrait coshCastTraitUpper = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait coshCastTraitLower = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    MicroAPI::MaskReg coshMask;
    MicroAPI::RegTensor<float> brcReg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castReg;
    MicroAPI::RegTensor<float> computeReg0;
    MicroAPI::RegTensor<float> computeReg1;
    MicroAPI::RegTensor<float> resReg;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::Duplicate(brcReg, scalarBrc);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        coshMask = MicroAPI::UpdateMask<float>(calCount);
        if constexpr (SupportBytes<T, 2>()) {
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * vlSize);
            MicroAPI::Cast<float, half, coshCastTraitUpper>(castReg, srcReg, coshMask);
        } else {
            MicroAPI::DataCopy(castReg, srcUb + i * vlSize);
        }
        MicroAPI::Adds(castReg, castReg, scalarNegLnTwo, coshMask);
        MicroAPI::Exp(computeReg0, castReg, coshMask);
        MicroAPI::Div(computeReg1, brcReg, computeReg0, coshMask);
        MicroAPI::Add(resReg, computeReg0, computeReg1, coshMask);
        if constexpr (SupportBytes<T, 2>()) {
            MicroAPI::Cast<half, float, coshCastTraitLower>(dstReg, resReg, coshMask);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * vlSize, dstReg, coshMask);
        } else {
            MicroAPI::DataCopy(dstUb + i * vlSize, resReg, coshMask);
        }
    }
}
} // namespace CoshInternal

template <typename T, bool isReuseSource = false>
__aicore__ inline void CoshImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    CoshImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void CoshImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    static_assert(SupportType<T, half, float>(), "Cosh only support half/float data type on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Cosh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Cosh");

    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)srcTensor.GetPhyAddr();
    constexpr int32_t vlSize = static_cast<int32_t>(GetVecLen() / sizeof(float));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));
    VF_CALL<CoshInternal::CoshCompute<T>>(dstUb, srcUb, calCount, repeatTimes);
}
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_MATH_COSH_COSH_C310_IMPL_H