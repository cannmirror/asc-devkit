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
 * \file logical_not_common_impl.h
 * \brief
 */

#ifndef LIB_MATH_LOGICAL_NOT_IMPL_H
#define LIB_MATH_LOGICAL_NOT_IMPL_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"

namespace AscendC {
struct LogicalNotConfig {
    bool isReuseSource;
};
constexpr LogicalNotConfig DEFAULT_LOGICAL_NOT_CONFIG = { false };

template <typename T, typename U, typename RegT, typename RegU, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__aicore__ inline void LogicalNotVF(__ubuf__ T* dst, __ubuf__ U* src, uint16_t repeatTime, uint32_t count, uint32_t oneRepElm)
{
    RegT dstVreg;
    RegT brcZeroReg;
    RegT brcOneReg;
    RegU srcVreg;
    MicroAPI::MaskReg mask;
    MicroAPI::MaskReg cmpMask;

    MicroAPI::Duplicate(brcOneReg, 1u);
    MicroAPI::Duplicate(brcZeroReg, 0u);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<U, Trait>(count);
        MicroAPI::DataCopy(srcVreg, src + i * oneRepElm);
        MicroAPI::CompareScalar<U, CMPMODE::EQ>(cmpMask, srcVreg, static_cast<U>(0), mask);
        if constexpr (sizeof(U) == 2) {
            MicroAPI::MaskPack(cmpMask, cmpMask);
            MicroAPI::MaskPack(mask, mask);
        } else if constexpr (sizeof(U) == 4 || sizeof(U) == 8) {
            MicroAPI::MaskPack(cmpMask, cmpMask);
            MicroAPI::MaskPack(cmpMask, cmpMask);
            MicroAPI::MaskPack(mask, mask);
            MicroAPI::MaskPack(mask, mask);
        }
        MicroAPI::Select(dstVreg, brcOneReg, brcZeroReg, cmpMask);
        MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
    }
}

template <const LogicalNotConfig& config = DEFAULT_LOGICAL_NOT_CONFIG, typename T, typename U>
__aicore__ inline void LogicalNotImpl(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, bool>(), "LogicalNot only support bool data type on current device!");
    static_assert(SupportType<U, bool, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, 
        uint32_t, int32_t, uint64_t, int64_t>(), "LogicalNot only support bool/uint8_t/int8_t/half/bfloat16_t/"
        "uint16_t/int16_t/float/uint32_t/int32_t/uint64_t/int64_t data type on current device!");
    CHECK_FUNC_HIGHLEVEL_API(LogicalNot, (T, U, config.isReuseSource), (dst, src, count));
    using RegT = MicroAPI::RegTensor<T>;
    constexpr uint32_t LOGICAL_NOT_B64_REPEAT_STRIDE = 2;
    if constexpr (sizeof(U) == 8) {
        using RegU = MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>;
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(U) * LOGICAL_NOT_B64_REPEAT_STRIDE);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        VF_CALL<LogicalNotVF<T, U, RegT, RegU, MicroAPI::RegTraitNumTwo>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ U*)src.GetPhyAddr(), repeatTime, count, oneRepElm);
    } else {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(U));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        if constexpr (Std::is_same_v<U, bool>) {
            using RegU = MicroAPI::RegTensor<uint8_t>;
            VF_CALL<LogicalNotVF<T, uint8_t, RegT, RegU>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ uint8_t*)src.GetPhyAddr(), repeatTime, count, oneRepElm);
        } else {
            using RegU = MicroAPI::RegTensor<U>;
            VF_CALL<LogicalNotVF<T, U, RegT, RegU>>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ U*)src.GetPhyAddr(), repeatTime, count, oneRepElm);
        }
    }
}

}
#endif
#endif  // IMPL_MATH_LOGICAL_NOT_LOGICAL_NOT_COMMON_IMPL_H
