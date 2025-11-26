/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file bitwise_not_c310_impl.h
 * \brief
 */

#ifndef IMPL_MATH_BITWISE_NOT_BITWISE_NOT_C310_IMPL_H
#define IMPL_MATH_BITWISE_NOT_BITWISE_NOT_C310_IMPL_H
namespace AscendC {
struct BitwiseNotConfig {
    bool isReuseSource;
};
constexpr BitwiseNotConfig DEFAULT_BITWISE_NOT_CONFIG = {false};
template <typename T, typename RegT, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__simd_vf__ inline void BitwiseNotCompute(__ubuf__ T* dst, __ubuf__ T* src, uint32_t count,
                                         uint16_t repeatTime, uint32_t oneRepElm, uint32_t offset)
{
    MicroAPI::MaskReg mask;
    RegT srcVreg;
    RegT dstVreg;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T, Trait>(count);
        MicroAPI::DataCopy(srcVreg, src + i * oneRepElm);
        MicroAPI::Not(dstVreg, srcVreg, mask);
        MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
        mask = MicroAPI::UpdateMask<T, Trait>(count);
        MicroAPI::DataCopy(srcVreg, src + i * oneRepElm + offset);
        MicroAPI::Not(dstVreg, srcVreg, mask);
        MicroAPI::DataCopy(dst + i * oneRepElm + offset, dstVreg, mask);
    }
    mask = MicroAPI::UpdateMask<T, Trait>(count);
    MicroAPI::DataCopy(srcVreg, src + repeatTime * oneRepElm * 2);
    MicroAPI::Not(dstVreg, srcVreg, mask);
    MicroAPI::DataCopy(dst + repeatTime * oneRepElm * 2, dstVreg, mask);
}

template <const BitwiseNotConfig& config, typename T>
__aicore__ inline void BitwiseNotImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
{
    if ASCEND_IS_AIC {
        return;
    }

    static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t>(),
                  "current data type is not supported on current device!");
    CHECK_FUNC_HIGHLEVEL_API(BitwiseNot, (T, config.isReuseSource), (dst, src, count));

    __ubuf__ T* dstTensor = (__ubuf__ T*)dst.GetPhyAddr();
    __ubuf__ T* srcTensor = (__ubuf__ T*)src.GetPhyAddr();

    if constexpr (sizeof(T) == 8) {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T) * 2);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm) / 2);
        uint32_t offset = repeatTime * oneRepElm;
        BitwiseNotCompute<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>, MicroAPI::RegTraitNumTwo>(
            dstTensor, srcTensor, count, repeatTime, oneRepElm, offset);
    } else {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm) / 2);
        uint32_t offset = repeatTime * oneRepElm;
        BitwiseNotCompute<T, MicroAPI::RegTensor<T>>(dstTensor, srcTensor, count, repeatTime, oneRepElm,
                                                              offset);
    }
}
} // namespace AscendC
#endif // IMPL_MATH_BITWISE_NOT_BITWISE_NOT_C310_IMPL_H