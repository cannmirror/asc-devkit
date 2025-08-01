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
 * \file acosh_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_ACOSH_ACOSH_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_ACOSH_ACOSH_C310_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
namespace AcoshInternal {
constexpr float ACOSH_NEG_ONE = -1;
constexpr MicroAPI::CastTrait ACOSH_CAST_TRAIT_F162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait ACOSH_CAST_TRAIT_F322F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
} // namespace AcoshInternal
/*
    when x < 0 : acosh = nan
    when x > 0 : acosh = ln(x + sqrt(x ^ 2 - 1))
*/
template <typename T>
__aicore__ inline void AcoshImplVF(__ubuf__ T* dst, __ubuf__ T* src, uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::MaskReg mask;
    constexpr int32_t oneRepElm = static_cast<int32_t>(GetVecLen() / sizeof(float));
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, src + i * oneRepElm);
            MicroAPI::Cast<float, half, AcoshInternal::ACOSH_CAST_TRAIT_F162F32>(
                (MicroAPI::RegTensor<float>&)srcVreg, srcVreg, mask);
        } else {
            MicroAPI::DataCopy(srcVreg, src + i * oneRepElm);
        }
        MicroAPI::Mul(tmpReg, (MicroAPI::RegTensor<float>&)srcVreg, (MicroAPI::RegTensor<float>&)srcVreg, mask);
        MicroAPI::Adds(tmpReg, tmpReg, AcoshInternal::ACOSH_NEG_ONE, mask);
        MicroAPI::Sqrt(tmpReg, tmpReg, mask);
        MicroAPI::Add(tmpReg, tmpReg, (MicroAPI::RegTensor<float>&)srcVreg, mask);
        MicroAPI::Ln((MicroAPI::RegTensor<float>&)dstVreg, tmpReg, mask);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<half, float, AcoshInternal::ACOSH_CAST_TRAIT_F322F16>(
                dstVreg, (MicroAPI::RegTensor<float>&)dstVreg, mask);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * oneRepElm, dstVreg, mask);
        } else {
            MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
        }
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcoshImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "Acosh only support half/float data type on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Acosh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Acosh");
    constexpr int32_t oneRepElm = static_cast<int32_t>(GetVecLen() / sizeof(float));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, oneRepElm));
    VF_CALL<AcoshImplVF<T>>(
        (__ubuf__ T*)dstTensor.GetPhyAddr(), (__ubuf__ T*)srcTensor.GetPhyAddr(), calCount, repeatTimes);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcoshImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcoshImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcoshImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATH_ACOSH_ACOSH_C310_IMPL_H