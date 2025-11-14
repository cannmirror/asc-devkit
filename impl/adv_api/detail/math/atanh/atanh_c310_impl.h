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
 * \file atanh_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_ATANH_ATANH_C310_IMPL_H
#define IMPL_MATH_ATANH_ATANH_C310_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"
 
namespace AscendC {
namespace Internal {
constexpr float ATANH_ONE = 1;
constexpr float ATANH_NEG_ONE = -1;
constexpr float ATANH_MULS_COSNTANT = 0.5;
/*
    atanh = 0.5 * ln((1 + x) / (1 - x))
*/
template <typename T>
__simd_vf__ inline void AtanhImplVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<float> tmpReg1;
    MicroAPI::RegTensor<float> tmpReg2;
    MicroAPI::RegTensor<float> dupReg;
    MicroAPI::MaskReg mask;
    constexpr int32_t oneRepElm = static_cast<int32_t>(GetVecLen() / sizeof(float));
    MicroAPI::Duplicate<float>(dupReg, ATANH_ONE);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, src + i * oneRepElm);
            MicroAPI::Cast<float, half, castTraitB16ToB32>(
                (MicroAPI::RegTensor<float>&)srcVreg, srcVreg, mask);
        } else {
            MicroAPI::DataCopy(srcVreg, src + i * oneRepElm);
        }
        MicroAPI::Adds(tmpReg1, (MicroAPI::RegTensor<float>&)srcVreg, ATANH_ONE, mask);
        MicroAPI::Sub(tmpReg2, dupReg, (MicroAPI::RegTensor<float>&)srcVreg, mask);
        MicroAPI::Div(tmpReg1, tmpReg1, tmpReg2, mask);
        MicroAPI::Ln(tmpReg1, tmpReg1, mask);
        MicroAPI::Muls((MicroAPI::RegTensor<float>&)dstVreg, tmpReg1, ATANH_MULS_COSNTANT, mask);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<half, float, castTraitB32ToB16>(
                dstVreg, (MicroAPI::RegTensor<float>&)dstVreg, mask);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * oneRepElm, dstVreg, mask);
        } else {
            MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
        }
    }
}
} // namespace Internal

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "Atanh only support half/float data type on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Atanh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Atanh");
    constexpr int32_t oneRepElm = static_cast<int32_t>(GetVecLen() / sizeof(float));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, oneRepElm));
    Internal::AtanhImplVF<T>((__ubuf__ T*)dstTensor.GetPhyAddr(), (__ubuf__ T*)srcTensor.GetPhyAddr(), calCount, repeatTimes);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    AtanhImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    AtanhImpl<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    AtanhImpl<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}

}   // namespace AscendC
#endif  //IMPL_MATH_ATANH_ATANH_C310_IMPL_H
