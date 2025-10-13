/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/* !
 * \file fma_common_impl.h
 * \brief
 */

#ifndef IMPL_MATH_FMA_FMA_COMMON_IMPL_H
#define IMPL_MATH_FMA_FMA_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
struct FmaConfig {
    bool isReuseSource;
};
constexpr FmaConfig DEFAULT_FMA_CONFIG = { false };

template <typename T>
__aicore__ inline void FmaImplVF(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* src2,
    uint32_t count, uint16_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    MicroAPI::RegTensor<T> srcVreg0;
    MicroAPI::RegTensor<T> srcVreg1;
    MicroAPI::RegTensor<T> srcVreg2;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(count);
        MicroAPI::DataCopy(srcVreg0, src0 + i * oneRepElm);
        MicroAPI::DataCopy(srcVreg1, src1 + i * oneRepElm);
        MicroAPI::DataCopy(srcVreg2, src2 + i * oneRepElm);
        MicroAPI::FusedMulDstAdd(srcVreg0, srcVreg1, srcVreg2, mask);
        MicroAPI::DataCopy(dst + i * oneRepElm, srcVreg0, mask);
    }
}

template <const FmaConfig& config, typename T>
__aicore__ inline void FmaImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, const LocalTensor<T>& src2, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "Fma only support half/float data type on current device!");
    CHECK_FUNC_HIGHLEVEL_API(Fma, (T, config.isReuseSource), (dst, src0, src1, src2, sharedTmpBuffer, count));
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
    VF_CALL<FmaImplVF<T>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ T*)src0.GetPhyAddr(),
        (__ubuf__ T*)src1.GetPhyAddr(), (__ubuf__ T*)src2.GetPhyAddr(), count, repeatTimes);
}

template <const FmaConfig& config, typename T>
__aicore__ inline void FmaImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, const LocalTensor<T>& src2, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    FmaImpl<config, T>(dst, src0, src1, src2, sharedTmpBuffer, count);
}
} // namespace AscendC
#endif // IMPL_MATH_FMA_FMA_COMMON_IMPL_H