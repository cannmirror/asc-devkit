/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_micro_maskreg_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_MASKREG_IMPL_H
#define ASCENDC_MODULE_MICRO_MASKREG_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {

template <typename T, const RegTrait &regTrait = RegTraitNumOne>
__aicore__ inline MaskReg UpdateMaskImpl(uint32_t &scalar)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "UpdateMask only support type b8/b16/b32 on current device");
    MaskReg reg;
    if constexpr (sizeof(T) == 1) {
        reg = plt_b8(scalar);
    } else if constexpr (sizeof(T) == 2) {
        reg = plt_b16(scalar);
    } else if constexpr (sizeof(T) == 4) {
        reg = plt_b32(scalar);
    }
    return reg;
}

template <typename T, MaskPattern mode = MaskPattern::ALL, const RegTrait &regTrait = RegTraitNumOne>
__aicore__ inline MaskReg CreateMaskImpl()
{
    (void)regTrait;
    static_assert(SupportBytes<T, 1, 2, 4>(), "CreateMask only support type b8/b16/b32 on current device");
    constexpr auto modeValue = std::integral_constant<::Pat, static_cast<::Pat>(mode)>();
    MaskReg reg;
    // pset instruction can only appear in exe phase, so we use pge instruction here
    if constexpr (sizeof(T) == 1) {
        reg = pge_b8(modeValue);
    } else if constexpr (sizeof(T) == 2) {
        reg = pge_b16(modeValue);
    } else if constexpr (sizeof(T) == 4) {
        reg = pge_b32(modeValue);
    }
    return reg;
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void MaskGenWithRegTensorImpl(MaskReg &dstMask, RegT &srcReg, int16_t bitOffset)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "MaskGenWithRegTensor is not supported on current device!"); });
}

__aicore__ inline void MaskNotImpl(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask)
{
    pnot(dstMask, srcMask, mask);
}

__aicore__ inline void MaskAndImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    pand(dstMask, srcMask0, srcMask1, mask);
}

__aicore__ inline void MaskOrImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    por(dstMask, srcMask0, srcMask1, mask);
}

__aicore__ inline void MaskXorImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    pxor(dstMask, srcMask0, srcMask1, mask);
}

__aicore__ inline void MaskMovImpl(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask)
{
    pmov(dstMask, srcMask, mask);
}

__aicore__ inline void MaskMovImpl(MaskReg &dstMask, MaskReg &srcMask)
{
    pmov(dstMask, srcMask);
}

template <typename T>
__aicore__ inline void MaskSlideImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, const int16_t slideAmount)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "MaskSlide only support type b8/b16/b32 ");
    if constexpr (sizeof(T) == 1) {
        pslide_b8(dstMask, srcMask0, srcMask1, slideAmount);
    } else if constexpr (sizeof(T) == 2) {
        pslide_b16(dstMask, srcMask0, srcMask1, slideAmount);
    } else if constexpr (sizeof(T) == 4) {
        pslide_b32(dstMask, srcMask0, srcMask1, slideAmount);
    }
}

template <typename T>
__aicore__ inline void MaskInterleaveImpl(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "MaskInterleave only support type b8/b16/b32 on current device");
    if constexpr (sizeof(T) == 1) {
        pintlv_b8(dstMask0, dstMask1, srcMask0, srcMask1);
    } else if constexpr (sizeof(T) == 2) {
        pintlv_b16(dstMask0, dstMask1, srcMask0, srcMask1);
    } else if constexpr (sizeof(T) == 4) {
        pintlv_b32(dstMask0, dstMask1, srcMask0, srcMask1);
    }
}

template <typename T>
__aicore__ inline void MaskDeInterleaveImpl(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "MaskDeInterleave only support type b8/b16/b32 on current device");
    if constexpr (sizeof(T) == 1) {
        pdintlv_b8(dstMask0, dstMask1, srcMask0, srcMask1);
    } else if constexpr (sizeof(T) == 2) {
        pdintlv_b16(dstMask0, dstMask1, srcMask0, srcMask1);
    } else if constexpr (sizeof(T) == 4) {
        pdintlv_b32(dstMask0, dstMask1, srcMask0, srcMask1);
    }
}

__aicore__ inline void MaskSelImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    psel(dstMask, srcMask0, srcMask1, mask);
}

template <HighLowPart part = HighLowPart::LOWEST>
__aicore__ inline void MaskPackImpl(MaskReg &dstMask, MaskReg &srcMask)
{
    constexpr auto partValue = std::integral_constant<::HiloPart, static_cast<::HiloPart>(part)>();
    ppack(dstMask, srcMask, partValue);
}

template <HighLowPart part = HighLowPart::LOWEST>
__aicore__ inline void MaskUnPackImpl(MaskReg &dstMask, MaskReg &srcMask)
{
    constexpr auto partValue = std::integral_constant<::HiloPart, static_cast<::HiloPart>(part)>();
    punpack(dstMask, srcMask, partValue);
}

template <typename T> __aicore__ inline MaskReg MoveMaskImpl()
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "MoveMask is not supported on current device!"); });
}

} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_MASKREG_IMPL_H