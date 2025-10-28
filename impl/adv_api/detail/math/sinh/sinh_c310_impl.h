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
 * \file sinh_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_SINH_SINH_C310_IMPL_H
#define IMPL_MATH_SINH_SINH_C310_IMPL_H

#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
namespace SinhInternal {
// Computes sinh values based on input types.
// According formula: sinh(x) = (e^x - e^(-x))/2 = e^(x-ln2) - 0.25/(e^(x-ln2)).
template <typename T>
__aicore__ inline void SinhCompute(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t calCount, uint16_t repeatTimes)
{
    constexpr float scalarNegLnTwo = -0.6931472;
    constexpr float scalarBrc = 0.25;
    constexpr uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(float));
    static constexpr MicroAPI::CastTrait sinhCastTraitUpper = { MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN };
    static constexpr MicroAPI::CastTrait sinhCastTraitLower = { MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
    MicroAPI::MaskReg sinhMask;
    MicroAPI::RegTensor<float> dupReg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castUpperReg;
    MicroAPI::RegTensor<float> computeReg0;
    MicroAPI::RegTensor<float> computeReg1;
    MicroAPI::RegTensor<float> resReg;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::Duplicate(dupReg, scalarBrc);
    for (uint16_t i = 0; i < repeatTimes; i++) {
        sinhMask = MicroAPI::UpdateMask<float>(calCount);
        if constexpr (SupportBytes<T, 2>()) {
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * vlSize);
            MicroAPI::Cast<float, half, sinhCastTraitUpper>(castUpperReg, srcReg, sinhMask);
        } else {
            MicroAPI::DataCopy(castUpperReg, srcUb + i * vlSize);
        }
        MicroAPI::Adds(castUpperReg, castUpperReg, scalarNegLnTwo, sinhMask);
        MicroAPI::Exp(computeReg0, castUpperReg, sinhMask);
        MicroAPI::Div(computeReg1, dupReg, computeReg0, sinhMask);
        MicroAPI::Sub(resReg, computeReg0, computeReg1, sinhMask);
        if constexpr (SupportBytes<T, 2>()) {
            MicroAPI::Cast<half, float, sinhCastTraitLower>(dstReg, resReg, sinhMask);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * vlSize, dstReg, sinhMask);
        } else {
            MicroAPI::DataCopy(dstUb + i * vlSize, resReg, sinhMask);
        }
    }
}
} // namespace SinhInternal

template <typename T, bool isReuseSource = false>
__aicore__ inline void SinhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    SinhImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SinhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    static_assert(SupportType<T, half, float>(), "Sinh only support half/float data type on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Sinh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Sinh");

    __local_mem__ T *dstUb = (__local_mem__ T *)dstTensor.GetPhyAddr();
    __local_mem__ T *srcUb = (__local_mem__ T *)srcTensor.GetPhyAddr();
    constexpr int32_t vlSize = static_cast<int32_t>(GetVecLen() / sizeof(float));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));
    VF_CALL<SinhInternal::SinhCompute<T>>(dstUb, srcUb, calCount, repeatTimes);
}
} // namespace AscendC

#endif // IMPL_MATH_SINH_SINH_C310_IMPL_H