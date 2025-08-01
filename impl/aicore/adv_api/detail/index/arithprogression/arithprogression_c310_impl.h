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
 * \file arithprogression_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_INDEX_ARITHPROGRESSION_ARITHPROGRESSION_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_INDEX_ARITHPROGRESSION_ARITHPROGRESSION_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_log.h"
namespace AscendC {
// Generating an underlying arithmetic sequence through scalar operations.
template <typename RegT, typename ScalarT>
__aicore__ inline void GetBaseArithProgression(RegT& dstReg, const ScalarT firstValue, const ScalarT diffValue)
{
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint8_t>();
    MicroAPI::Arange(dstReg, ScalarT(0));
    MicroAPI::Muls(dstReg, dstReg, diffValue, fullMask);
    MicroAPI::Adds(dstReg, dstReg, firstValue, fullMask);
}

template <typename T, const MicroAPI::RegTrait& regTrait>
__aicore__ inline void VfCallArithProgression(
    __ubuf__ T* dstLocalAddr, const T firstValue, const T diffValue, const int32_t count, const uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T, regTrait> tmpReg;
        MicroAPI::RegTensor<T, regTrait> stepReg;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL, regTrait>();
        GetBaseArithProgression(tmpReg, firstValue, diffValue);
        uint32_t sreg = static_cast<uint32_t>(count);
        MicroAPI::MaskReg preg;
        const uint32_t sregLower = static_cast<uint32_t>(regTrait.REG_NUM * ONE_REPEAT_BYTE_SIZE / sizeof(T));
        MicroAPI::Duplicate(stepReg, static_cast<T>(sregLower));
        MicroAPI::Muls(stepReg, stepReg, diffValue, fullMask);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = MicroAPI::UpdateMask<T, regTrait>(sreg);
            MicroAPI::DataCopy(dstLocalAddr + i * sregLower, tmpReg, preg);
            MicroAPI::Add(tmpReg, tmpReg, stepReg, fullMask);
        }
    }
}

template <typename T>
__aicore__ inline void ArithProgressionImpl(
    const LocalTensor<T>& dstLocal, const T firstValue, const T diffValue, const int32_t count)
{
    ASCENDC_ASSERT((dstLocal.GetSize() >= count),
        { KERNEL_LOG(KERNEL_ERROR, "dst length must equal with ArithProgression length"); });
    ASCENDC_ASSERT((static_cast<float>(diffValue) >= static_cast<float>(0)),
        { KERNEL_LOG(KERNEL_ERROR, "diff value mast bigger then 0"); });
    static_assert(SupportType<T, int16_t, int32_t, half, float, int64_t>(),
        "current data type is not supported on current device!");

    __ubuf__ T* dstLocalAddr = (__ubuf__ T*)dstLocal.GetPhyAddr();
    if constexpr (sizeof(T) != 8) {
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, ONE_REPEAT_BYTE_SIZE / sizeof(T)));
        VfCallArithProgression<T, MicroAPI::RegTraitNumOne>(dstLocalAddr, firstValue, diffValue, count, repeatTimes);
    } else {
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, 2 * ONE_REPEAT_BYTE_SIZE / sizeof(T)));
        VfCallArithProgression<T, MicroAPI::RegTraitNumTwo>(dstLocalAddr, firstValue, diffValue, count, repeatTimes);
    }
}
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_INDEX_ARITHPROGRESSION_ARITHPROGRESSION_C310_IMPL_H
