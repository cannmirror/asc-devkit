/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file clamp_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_CLAMP_CLAMP_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_CLAMP_CLAMP_C310_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
template <typename T, CLAMPMODE selMode, bool isReuseSource = false>
__aicore__ inline void ClampCompute(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, const T scalar, uint32_t calCount, const uint16_t repeatTimes)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg selMask;
    MicroAPI::MaskReg nanMask;
    MicroAPI::RegTensor<T> scalarReg;
    MicroAPI::Duplicate(scalarReg, scalar);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        maskReg = MicroAPI::UpdateMask<T>(calCount);
        MicroAPI::DataCopy<T>(srcReg, srcUb + i * repeatElm);
        MicroAPI::Compare<T, CMPMODE::NE>(nanMask, srcReg, srcReg, maskReg);
        if constexpr (selMode == CLAMPMODE::CLAMP_MAX) {
            MicroAPI::CompareScalar<T, CMPMODE::LT>(selMask, srcReg, scalar, maskReg);
        } else {
            MicroAPI::CompareScalar<T, CMPMODE::GE>(selMask, srcReg, scalar, maskReg);
        }
        MicroAPI::MaskOr(selMask, nanMask, selMask, maskReg);
        MicroAPI::Select(dstReg, srcReg, scalarReg, selMask);
        MicroAPI::DataCopy<T>(dstUb + i * repeatElm, dstReg, maskReg);
    }
}
/* **************************************************************************************************
 * ClampMax                                           *
 * ************************************************************************************************* */
template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMaxImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    static_assert(SupportType<T, float, half>(), "ClampMax only support half/float data type on current device");

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "ClampMax");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "ClampMax");

    __local_mem__ T* srcUb = (__local_mem__ T*)srcTensor.GetPhyAddr();
    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, repeatElm));
    VF_CALL<ClampCompute<T, CLAMPMODE::CLAMP_MAX, isReuseSource>>(dstUb, srcUb, scalar, calCount, repeatTimes);
}

/* **************************************************************************************************
 * ClampMin                                           *
 * ************************************************************************************************* */

template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    static_assert(SupportType<T, float, half>(), "ClampMin only support half/float data type on current device");

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "ClampMin");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "ClampMin");

    __local_mem__ T* srcUb = (__local_mem__ T*)srcTensor.GetPhyAddr();
    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, repeatElm));
    VF_CALL<ClampCompute<T, CLAMPMODE::CLAMP_MIN, isReuseSource>>(dstUb, srcUb, scalar, calCount, repeatTimes);
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATH_CLAMP_CLAMP_C310_IMPL_H