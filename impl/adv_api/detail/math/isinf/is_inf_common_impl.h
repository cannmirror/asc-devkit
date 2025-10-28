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
 * \file if_inf_common_impl.h
 * \brief
 */

#ifndef LIB_MATH_IS_INF_IMPL_H
#define LIB_MATH_IS_INF_IMPL_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"

namespace AscendC {
struct IsInfConfig {
    bool isReuseSource;
};
constexpr IsInfConfig DEFAULT_IS_INF_CONFIG = { false };

template <typename T, typename U>
__aicore__ inline void IsInfImplVF(__ubuf__ T* dst, __ubuf__ U* src, uint32_t count, uint16_t repeatTimes)
{
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(U));
    constexpr uint32_t floatInf = F32_INF;
    constexpr uint32_t floatNInf = F32_NEG_INF;
    constexpr uint32_t halfInf = 0x7c00;
    constexpr uint32_t halfNInf = 0xfc00;

    MicroAPI::RegTensor<U> srcVreg;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<T> vReg0;
    MicroAPI::RegTensor<T> vReg1;
    MicroAPI::MaskReg mask;
    MicroAPI::MaskReg cmpMaskReg;
    MicroAPI::MaskReg cmpInfMask;
    MicroAPI::MaskReg cmpNInfMask;
    if constexpr (Std::is_same_v<T, bool>) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint8_t>&)vReg0, 0u);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint8_t>&)vReg1, 1u);
    } else {
        MicroAPI::Duplicate(vReg0, 0.0);
        MicroAPI::Duplicate(vReg1, 1.0);
    }
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<U>(count);
        MicroAPI::DataCopy(srcVreg, src + i * oneRepElm);
        if constexpr (Std::is_same_v<U, float>) {
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(cmpInfMask, (MicroAPI::RegTensor<uint32_t>&)srcVreg, floatInf, mask);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(cmpNInfMask, (MicroAPI::RegTensor<uint32_t>&)srcVreg, floatNInf, mask);
        } else if constexpr (Std::is_same_v<U, half>) {
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(cmpInfMask, (MicroAPI::RegTensor<uint16_t>&)srcVreg, halfInf, mask);
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(cmpNInfMask, (MicroAPI::RegTensor<uint16_t>&)srcVreg, halfNInf, mask);
        }
        MicroAPI::MaskOr(cmpMaskReg, cmpInfMask, cmpNInfMask, mask);
        if constexpr (Std::is_same_v<T, bool>) {
            if constexpr (Std::is_same_v<U, float>) {
                MicroAPI::MaskPack(cmpMaskReg, cmpMaskReg);
                MicroAPI::MaskPack(cmpMaskReg, cmpMaskReg);
                MicroAPI::Select(dstVreg, vReg1, vReg0, cmpMaskReg);
                MicroAPI::MaskPack(mask, mask);
                MicroAPI::MaskPack(mask, mask);
            } else if constexpr (Std::is_same_v<U, half>) {
                MicroAPI::MaskPack(cmpMaskReg, cmpMaskReg);
                MicroAPI::Select(dstVreg, vReg1, vReg0, cmpMaskReg);
                MicroAPI::MaskPack(mask, mask);
            }
        } else {
            MicroAPI::Select(dstVreg, vReg1, vReg0, cmpMaskReg);
        }
        MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
    }
}

template <const IsInfConfig& config, typename T, typename U>
__aicore__ inline void IsInfImpl(const LocalTensor<T>& dst, const LocalTensor<U>& src, 
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert((SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<bool, half>,
        Tuple<float, float>, Tuple<bool, float>>()), "Failed to check dtype in IsInf, current api "
        "support dtype combination is T : half/bool, U : half; T : float/bool, U : float");
    CHECK_FUNC_HIGHLEVEL_API(IsInf, (T, U, config.isReuseSource), (dst, src, sharedTmpBuffer, count));
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(U));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
    VF_CALL<IsInfImplVF<T, U>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ U*)src.GetPhyAddr(), count,
        repeatTimes);
}

template <const IsInfConfig& config, typename T, typename U>
__aicore__ inline void IsInfImpl(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    IsInfImpl<config, T, U>(dst, src, sharedTmpBuffer, count);
}
}  // namespace AscendC
#endif
#endif  // IMPL_MATH_ISINF_ISINF_COMMON_IMPL_H
