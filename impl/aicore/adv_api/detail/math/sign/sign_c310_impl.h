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
 * \file sign_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_SIGN_SIGN_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_SIGN_SIGN_C310_IMPL_H

#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
namespace SignInternal {
template <typename T>
__aicore__ inline void SignCoreCompute(__ubuf__ T* dstUb, __ubuf__ T* srcUb, uint32_t calCount, uint16_t repeatTimes)
{
    constexpr uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    MicroAPI::MaskReg signMask;
    MicroAPI::MaskReg cmpMask0;
    MicroAPI::MaskReg cmpMask1;
    MicroAPI::RegTensor<T> brcZeroReg;
    MicroAPI::RegTensor<T> brcOneReg;
    MicroAPI::RegTensor<T> brcNegOneReg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<T> selReg0;
    MicroAPI::RegTensor<T> selReg1;
    MicroAPI::Duplicate(brcZeroReg, 0);
    MicroAPI::Duplicate(brcOneReg, 1);
    MicroAPI::Duplicate(brcNegOneReg, -1);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        signMask = MicroAPI::UpdateMask<T>(calCount);
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
__aicore__ inline void SignCompute(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    static_assert(SupportType<T, half, float>(), "Sign only support half/float data type on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Sign");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Sign");

    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)srcTensor.GetPhyAddr();

    constexpr int32_t vlSize = static_cast<int32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, vlSize));
    VF_CALL<SignInternal::SignCoreCompute<T>>(dstUb, srcUb, calCount, repeatTimes);
}
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_MATH_SIGN_SIGN_C310_IMPL_H