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
 * \file geglu_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_ACTIVATION_GEGLU_GEGLU_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_ACTIVATION_GEGLU_GEGLU_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../../common/check.h"

namespace AscendC {
namespace GeGLUInternal {
constexpr MicroAPI::CastTrait castTraitB16ToB32{
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait castTraitB32ToB16{
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr float gegluConstantA = 22.36386;
constexpr float gegluConstantB = -0.071354814;
} // namespace GeGLUInternal
template <typename T>
__aicore__ inline void GeGLUImplVF(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint32_t count, const uint16_t repeatTimes)
{
    MicroAPI::RegTensor<half> srcOrigin0;
    MicroAPI::RegTensor<half> srcOrigin1;
    MicroAPI::RegTensor<float> srcVreg0;
    MicroAPI::RegTensor<float> srcVreg1;
    MicroAPI::RegTensor<float> tmpReg0;
    MicroAPI::RegTensor<float> tmpReg1;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::MaskReg mask;
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(float));
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(count);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcOrigin0, src0 + i * oneRepElm);
            MicroAPI::Cast<float, half, GeGLUInternal::castTraitB16ToB32>(srcVreg0, srcOrigin0, mask);
            MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcOrigin1, src1 + i * oneRepElm);
            MicroAPI::Cast<float, half, GeGLUInternal::castTraitB16ToB32>(srcVreg1, srcOrigin1, mask);
        } else {
            MicroAPI::DataCopy(srcVreg0, src0 + i * oneRepElm);
            MicroAPI::DataCopy(srcVreg1, src1 + i * oneRepElm);
        }
        MicroAPI::Mul(tmpReg0, srcVreg1, srcVreg1, mask);
        MicroAPI::Adds(tmpReg0, tmpReg0, GeGLUInternal::gegluConstantA, mask);
        MicroAPI::Mul(tmpReg0, tmpReg0, srcVreg1, mask);
        MicroAPI::Muls(tmpReg0, tmpReg0, GeGLUInternal::gegluConstantB, mask);
        MicroAPI::Exp(tmpReg1, tmpReg0, mask);
        MicroAPI::Adds(tmpReg1, tmpReg1, 1.0f, mask);
        MicroAPI::Div(tmpReg1, srcVreg1, tmpReg1, mask);
        MicroAPI::Mul(dstVreg, srcVreg0, tmpReg1, mask);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<half, float, GeGLUInternal::castTraitB32ToB16>(
                (MicroAPI::RegTensor<half>&)dstVreg, dstVreg, mask);
            MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(
                dst + i * oneRepElm, (MicroAPI::RegTensor<half>&)dstVreg, mask);
        } else {
            MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
        }
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLUImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "GeGLU only support half/float data type on current device!");
    ASCENDC_ASSERT((srcTensor0.GetSize() == srcTensor1.GetSize()),
        { KERNEL_LOG(KERNEL_ERROR, "Input params.GetSize must be equal with each other!"); });
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor0, "srcTensor0", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor1, "srcTensor1", "VECIN, VECOUT, VECCALC");
    CheckCalCount(count, "count", dstTensor, "dstTensor", "GeGLU");
    CheckCalCount(count, "count", srcTensor0, "srcTensor0", "GeGLU");
    CheckCalCount(count, "count", srcTensor1, "srcTensor1", "GeGLU");
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(float));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
    VF_CALL<GeGLUImplVF<T>>((__ubuf__ T*)dstTensor.GetPhyAddr(), (__ubuf__ T*)srcTensor0.GetPhyAddr(),
        (__ubuf__ T*)srcTensor1.GetPhyAddr(), count, repeatTimes);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLUImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    GeGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, count);
}

} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_ACTIVATION_GEGLU_GEGLU_C310_IMPL_H
