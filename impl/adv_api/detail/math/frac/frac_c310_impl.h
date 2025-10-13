/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file frac_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_FRAC_FRAC_C310_IMPL_H
#define IMPL_MATH_FRAC_FRAC_C310_IMPL_H

#include "kernel_tensor.h"

namespace AscendC {
namespace FRAC {

constexpr MicroAPI::CastTrait castTraitF162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait castTraitF322F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

__aicore__ inline void FracCompute(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg mask)
{
    MicroAPI::Truncate<float, RoundMode::CAST_TRUNC>(dstReg, srcReg, mask);
    MicroAPI::Sub(dstReg, srcReg, dstReg, mask);
}

template<typename T>
__aicore__ inline void FracCoreImpl(__ubuf__ T* dstUb, __ubuf__ T* srcUb, uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> dstReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(calCount);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
            MicroAPI::Cast<float, T, castTraitF162F32>(castReg, srcReg, mask);
        } else {
            MicroAPI::DataCopy(castReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
        }
        FracCompute(dstReg, castReg, mask);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<T, float, castTraitF322F16>(srcReg, dstReg, mask);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * B32_DATA_NUM_PER_REPEAT, srcReg, mask);
        } else {
            MicroAPI::DataCopy(dstUb + i * B32_DATA_NUM_PER_REPEAT, dstReg, mask);
        }
    }
}

} // namespace Frac

template <typename T, bool isReuseSource = false>
__aicore__ inline void FracImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPos(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECOUT / VECCALC", "Frac");
    FracImpl(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void FracImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    static_assert((SupportType<T, half, float>(), "current data type is not supported on current device!"));
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "Frac");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "Frac");
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
    VF_CALL<FRAC::FracCoreImpl<T>>(dstUb, srcUb, calCount, repeatTimes);
}
} // namespace AscendC

#endif // IMPL_MATH_FRAC_FRAC_C310_IMPL_H
