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
 * \file swish_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_SWISH_SWISH_C310_IMPL_H
#define IMPL_MATH_SWISH_SWISH_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "../../common/check.h"
 
namespace AscendC {
namespace Internal {

template <typename T>
__simd_vf__ inline void SwishComputeVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t count, const T scalarValue,
    const uint16_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(count);
        MicroAPI::LoadAlign(srcVreg, src + i * oneRepElm);
        MicroAPI::Muls(vreg0, srcVreg, scalarValue, mask);
        MicroAPI::Exp(vreg0, vreg0, mask);
        MicroAPI::Adds(vreg0, vreg0, 1.0f, mask);
        MicroAPI::Div(dstVreg, srcVreg, vreg0, mask);
        MicroAPI::StoreAlign(dst + i * oneRepElm, dstVreg, mask);
    } 
}
} // namespace Internal

template <typename T, bool isReuseSource = false>
__aicore__ inline void SwishCompute(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t count, const T scalarValue)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "Swish only support half/float data type on current device!");
    CheckTensorPosition(dstLocal, "dstLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcLocal, "srcLocal", "VECIN, VECOUT, VECCALC");
    CheckCalCount(count, "count", dstLocal, "dstLocal", "Swish");
    CheckCalCount(count, "count", srcLocal, "srcLocal", "Swish");
    float negOne = -1.0;
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
    const T scalar = static_cast<T>(negOne * static_cast<float>(scalarValue));
    Internal::SwishComputeVF<T>(
        (__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), count, scalar, repeatTimes);
}

}   // namespace AscendC
#endif  // IMPL_MATH_SWISH_SWISH_C310_IMPL_H
