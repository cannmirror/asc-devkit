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
 * \file sigmoid_c310_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SIGMOID_C310_IMPL_H
#define IMPL_ACTIVATION_SIGMOID_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "../../common/common.h"

namespace AscendC {
namespace Internal {
/*
 * Formula is y= 1 / (1 + exp(-x))
*/
template<typename T>
__simd_vf__ inline void SigmoidImplVF(__ubuf__ T* dstUb, __ubuf__ T* srcUb, uint32_t count, const uint16_t repeatTimes)
{
    uint32_t sreg = count;
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> dstReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = MicroAPI::UpdateMask<float>(sreg);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
            MicroAPI::Cast<float, T, castTraitB16ToB32>(castReg, srcReg, preg);
        } else {
            MicroAPI::DataCopy(castReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
        }
        MicroAPI::Muls(tmpReg, castReg, -1.0f, preg);
        MicroAPI::Exp(tmpReg, tmpReg, preg);

        MicroAPI::Adds(tmpReg, tmpReg, 1.0f, preg);
        MicroAPI::Duplicate(dstReg, 1.0f, preg);
        MicroAPI::Div(dstReg, dstReg, tmpReg, preg);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<T, float, castTraitB32ToB16>(srcReg, dstReg, preg);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * B32_DATA_NUM_PER_REPEAT, srcReg, preg);
        } else {
            MicroAPI::DataCopy(dstUb + i * B32_DATA_NUM_PER_REPEAT, dstReg, preg);
        }
    }
}
} // namespace Internal

template<typename T, bool isReuseSource = false>
__aicore__ inline void SigmoidImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "Sigmoid");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "Sigmoid");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "Sigmoid");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not be larger than srcTensor length %u", calCount,
            srcTensor.GetSize());
    });
    ASCENDC_ASSERT((calCount <= dstTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not be larger than dstTensor length %u", calCount,
            dstTensor.GetSize());
    });

    __local_mem__ T *dstUb = (__local_mem__ T *)dstTensor.GetPhyAddr();
    __local_mem__ T *srcUb = (__local_mem__ T *)srcTensor.GetPhyAddr();
    uint16_t repeatTimes = CeilDivision(calCount, B32_DATA_NUM_PER_REPEAT);
    Internal::SigmoidImplVF<T>(dstUb, srcUb, calCount, repeatTimes);
}
} // namespace AscendC

#endif // IMPL_ACTIVATION_SIGMOID_C310_IMPL_H
