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
 * \file sign_l300_impl.h
 * \brief
 */
#ifndef IMPL_MATH_SIGN_SIGN_L300_IMPL_H
#define IMPL_MATH_SIGN_SIGN_L300_IMPL_H

#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
namespace SignInternal {
template <typename T, typename RegT, const MicroAPI::RegTrait& trait = MicroAPI::RegTraitNumOne>
__simd_vf__ inline void SignCoreCompute(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t calCount, uint16_t repeatTime, uint32_t vlSize)
{
    MicroAPI::MaskReg signMask;
    MicroAPI::MaskReg cmpMask0;
    MicroAPI::MaskReg cmpMask1;
    RegT brcZeroReg;
    RegT brcOneReg;
    RegT brcNegOneReg;
    RegT srcReg;
    RegT selReg0;
    RegT selReg1;
    MicroAPI::Duplicate(brcZeroReg, 0);
    MicroAPI::Duplicate(brcOneReg, 1);
    MicroAPI::Duplicate(brcNegOneReg, -1);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        signMask = MicroAPI::UpdateMask<T, trait>(calCount);
        MicroAPI::DataCopy(srcReg, srcUb + i * vlSize);
        MicroAPI::CompareScalar<T, CMPMODE::LT>(cmpMask0, srcReg, 0, signMask);
        MicroAPI::CompareScalar<T, CMPMODE::GT>(cmpMask1, srcReg, 0, signMask);
        MicroAPI::Select(selReg0, brcNegOneReg, brcZeroReg, cmpMask0);
        MicroAPI::Select(selReg1, brcOneReg, selReg0, cmpMask1);
        MicroAPI::DataCopy(dstUb + i * vlSize, selReg1, signMask);
    }
}
} // namespace SignInternal

template <typename T, bool isReuseSource = false>
__aicore__ inline void SignCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    SignCompute<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SignCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    static_assert(SupportType<T, half, float, int64_t>(), "Sign only support half/float/int64_t data type on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Sign");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Sign");
    constexpr uint32_t SIGN_B64_REPEAT_STRIDE = 2;
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    if constexpr (sizeof(T) == 8) {
        using RegT = MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>;
        constexpr int32_t vlSize = static_cast<int32_t>(GetVecLen() / sizeof(T) * SIGN_B64_REPEAT_STRIDE);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, vlSize));
        SignInternal::SignCoreCompute<T, RegT, MicroAPI::RegTraitNumTwo>(dstUb, srcUb, calCount, repeatTime, vlSize);
    } else {
        using RegT = MicroAPI::RegTensor<T>;
        constexpr int32_t vlSize = static_cast<int32_t>(GetVecLen() / sizeof(T));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, vlSize));
        SignInternal::SignCoreCompute<T, RegT>(dstUb, srcUb, calCount, repeatTime, vlSize);
    }
}
} // namespace AscendC

#endif // IMPL_MATH_SIGN_SIGN_L300_IMPL_H