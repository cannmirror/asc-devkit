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

/* !
 * \file round_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_ROUND_ROUND_C310_IMPL_H
#define IMPL_MATH_ROUND_ROUND_C310_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
template <typename T, bool isReuseSource = false>
__aicore__ inline void RoundCompute(__local_mem__ T *dstUb, __local_mem__ T *srcUb, uint32_t calCount,
    const uint16_t repeatTimes)
{
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::MaskReg maskReg;
    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
        maskReg = MicroAPI::UpdateMask<T>(calCount);
        MicroAPI::DataCopy<T>(srcReg, srcUb + i * repeatElm);
        MicroAPI::Truncate<T, RoundMode::CAST_RINT>(dstReg, srcReg, maskReg);
        MicroAPI::DataCopy<T>(dstUb + i * repeatElm, dstReg, maskReg);
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void RoundImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    RoundImpl(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void RoundImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    static_assert(SupportType<T, float, half>(), "Round only support half/float data type on current device");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Round");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Round");

    __local_mem__ T *srcUb = (__local_mem__ T *)srcTensor.GetPhyAddr();
    __local_mem__ T *dstUb = (__local_mem__ T *)dstTensor.GetPhyAddr();
    constexpr uint32_t repeatElm = GetVecLen() / sizeof(T);
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, repeatElm));

    VF_CALL<RoundCompute<T, isReuseSource>>(dstUb, srcUb, calCount, repeatTimes);
}
} // namespace AscendC
#endif // IMPL_MATH_ROUND_ROUND_C310_IMPL_H