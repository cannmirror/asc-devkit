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
__simd_callee__ inline MaskReg UpdateMaskImpl(uint32_t &scalarValue)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "UpdateMask only support type b8/b16/b32");
    MaskReg reg;
    if constexpr (sizeof(T) == 1) {
        reg = plt_b8(scalarValue, POST_UPDATE);
    } else if constexpr (sizeof(T) == 2) {
        reg = plt_b16(scalarValue, POST_UPDATE);
    } else if constexpr (sizeof(T) == 4) {
        reg = plt_b32(scalarValue, POST_UPDATE);
    }
    return reg;
}

template <typename RegT>
__simd_callee__ inline void MaskGenWithRegTensorImpl(MaskReg &dstMask, RegT &srcReg, int16_t bitOffset)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "MaskGenWithRegTensor is not supported on current device!");});
}

template <typename T = DefaultType, int16_t Offset, typename RegT>
__simd_callee__ inline void MaskGenWithRegTensorImpl(MaskReg &dstMask, RegT &srcReg)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 2, 4>(), "MaskGenWithRegTensor only support type b16/b32 on current device");
    if constexpr (sizeof(ActualT) == 2) {
        static_assert((Offset >= 0) && (Offset <= 15), "MaskGenWithRegTensor Offset must be in 0~15 when T is b16");
        movvp(dstMask, (RegTensor<uint16_t> &)srcReg, Offset);
    } else if constexpr (sizeof(ActualT) == 4) {
        static_assert((Offset >= 0) && (Offset <= 31), "MaskGenWithRegTensor Offset must be in 0~31 when T is b32");
        movvp(dstMask, (RegTensor<uint32_t> &)srcReg, Offset);
    }
}

template <typename T, MaskPattern mode = MaskPattern::ALL, const RegTrait &regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg CreateMaskImpl()
{
    static_assert(SupportBytes<T, 1, 2, 4, 8>(), "CreateMask only support type b8/b16/b32/b64 on current device");
    constexpr auto modeValue = std::integral_constant<::Pat, static_cast<::Pat>(mode)>();
    MaskReg reg;
    if constexpr (sizeof(T) == 1) {
        reg = pset_b8(modeValue);
    } else if constexpr (sizeof(T) == 2) {
        reg = pset_b16(modeValue);
    } else if constexpr (sizeof(T) == 4) {
        reg = pset_b32(modeValue);
    } else if constexpr (sizeof(T) == 8) {
        reg = pset_b32(modeValue);
        if constexpr (regTrait.REG_NUM == 1) {
            punpack(reg, reg, LOWER);
        }
    }
    return reg;
}

__simd_callee__ inline void MaskNotImpl(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask)
{
    pnot(dstMask, srcMask, mask);
}

__simd_callee__ inline void MaskAndImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    pand(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskOrImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    por(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskXorImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    pxor(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskMovImpl(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask)
{
    pmov(dstMask, srcMask, mask);
}

__simd_callee__ inline void MaskMovImpl(MaskReg &dstMask, MaskReg &srcMask)
{
    pmov(dstMask, srcMask);
}

template <typename T>
__simd_callee__ inline void MaskSlideImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, const int16_t slideAmount)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "MaskSlide only support type b8/b16/b32");
    if constexpr (sizeof(T) == 1) {
        pslide_b8(dstMask, srcMask0, srcMask1, slideAmount);
    } else if constexpr (sizeof(T) == 2) {
        pslide_b16(dstMask, srcMask0, srcMask1, slideAmount);
    } else if constexpr (sizeof(T) == 4) {
        pslide_b32(dstMask, srcMask0, srcMask1, slideAmount);
    }
}

template <typename T>
__simd_callee__ inline void MaskInterleaveImpl(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "MaskInterleave only support type b8/b16/b32");
    if constexpr (sizeof(T) == 1) {
        pintlv_b8(dstMask0, dstMask1, srcMask0, srcMask1);
    } else if constexpr (sizeof(T) == 2) {
        pintlv_b16(dstMask0, dstMask1, srcMask0, srcMask1);
    } else if constexpr (sizeof(T) == 4) {
        pintlv_b32(dstMask0, dstMask1, srcMask0, srcMask1);
    }
}

template <typename T>
__simd_callee__ inline void MaskDeInterleaveImpl(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "MaskDeInterleave only support type b8/b16/b32");
    if constexpr (sizeof(T) == 1) {
        pdintlv_b8(dstMask0, dstMask1, srcMask0, srcMask1);
    } else if constexpr (sizeof(T) == 2) {
        pdintlv_b16(dstMask0, dstMask1, srcMask0, srcMask1);
    } else if constexpr (sizeof(T) == 4) {
        pdintlv_b32(dstMask0, dstMask1, srcMask0, srcMask1);
    }
}

__simd_callee__ inline void MaskSelImpl(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    psel(dstMask, srcMask0, srcMask1, mask);
}

template <HighLowPart part = HighLowPart::LOWEST>
__simd_callee__ inline void MaskPackImpl(MaskReg &dstMask, MaskReg &srcMask)
{
    constexpr auto partValue = std::integral_constant<::HiloPart, static_cast<::HiloPart>(part)>();
    ppack(dstMask, srcMask, partValue);
}

template <HighLowPart part = HighLowPart::LOWEST>
__simd_callee__ inline void MaskUnPackImpl(MaskReg &dstMask, MaskReg &srcMask)
{
    constexpr auto partValue = std::integral_constant<::HiloPart, static_cast<::HiloPart>(part)>();
    punpack(dstMask, srcMask, partValue);
}

template <typename T> __simd_callee__ inline MaskReg MoveMaskImpl()
{
    static_assert(SupportBytes<T, 2, 4>(), "MoveMask only support type b16/b32");

    if constexpr (sizeof(T) == 2) {
        return movp_b16();
    } else if constexpr (sizeof(T) == 4) {
        return movp_b32();
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_MASKREG_IMPL_H