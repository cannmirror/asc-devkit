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
 * \file kernel_micro_maskreg_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_MASKREG_IMPL_H
#define ASCENDC_MODULE_MICRO_MASKREG_IMPL_H

namespace AscendC {
namespace MicroAPI {
template <typename T, const RegTrait &regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg UpdateMaskImpl(uint32_t &scalar)
{
    static_assert(SupportBytes<T, 1, 2, 4, 8>(), "UpdateMask only support type b8/b16/b32/b64 on current device");
    MaskReg reg;
    if constexpr (sizeof(T) == 1) {
        reg = plt_b8(scalar, POST_UPDATE);
    } else if constexpr (sizeof(T) == 2) {
        reg = plt_b16(scalar, POST_UPDATE);
    } else if constexpr (sizeof(T) == 4) {
        if constexpr (SupportType<T, complex32>()) {
            if constexpr (regTrait.REG_NUM == 2) {
                reg = plt_b16(scalar, POST_UPDATE);
            } else {
                uint32_t updateScalar = scalar;
                reg = plt_b16(scalar, POST_UPDATE);
                punpack(reg, reg, LOWER);
                constexpr uint32_t one_repeat_num = VECTOR_REG_WIDTH / sizeof(T);
                scalar = (updateScalar > one_repeat_num) ? (updateScalar - one_repeat_num) : 0;
            }
        } else {
            reg = plt_b32(scalar, POST_UPDATE);
        }
    } else if constexpr (sizeof(T) == 8) {
        if constexpr (regTrait.REG_NUM == 2) {
            reg = plt_b32(scalar, POST_UPDATE);
        } else {
            uint32_t updateScalar = scalar;
            reg = plt_b32(scalar, POST_UPDATE);
            punpack(reg, reg, LOWER);
            constexpr uint32_t one_repeat_num = VECTOR_REG_WIDTH / sizeof(T);
            scalar = (updateScalar > one_repeat_num) ? (updateScalar - one_repeat_num) : 0;
        }
    }
    return reg;
}

template <typename T, MaskPattern mode, const RegTrait &regTrait>
__simd_callee__ inline constexpr MaskPattern GetMaskPattern()
{
    if constexpr ((regTrait.REG_NUM == 1) && (sizeof(T) == 8)) {
        if constexpr (mode == MaskPattern::H) {
            return MaskPattern::VL16;
        } else if constexpr (mode == MaskPattern::Q) {
            return MaskPattern::VL8;
        }
    } else if constexpr ((regTrait.REG_NUM == 1) && SupportType<T, complex32>()) {
        if constexpr (mode == MaskPattern::H) {
            return MaskPattern::VL32;
        } else if constexpr (mode == MaskPattern::Q) {
            return MaskPattern::VL16;
        }
    }
    return mode;
}

template <typename T, MaskPattern mode = MaskPattern::ALL, const RegTrait &regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg CreateMaskImpl()
{
    static_assert(SupportBytes<T, 1, 2, 4, 8>(), "CreateMask only support type b8/b16/b32/b64 on current device");
    constexpr auto modeValue = std::integral_constant<::Pat, static_cast<::Pat>(GetMaskPattern<T, mode, regTrait>())>();
    MaskReg reg;
    if constexpr (sizeof(T) == 1) {
        reg = pset_b8(modeValue);
    } else if constexpr (sizeof(T) == 2) {
        reg = pset_b16(modeValue);
    } else if constexpr (sizeof(T) == 4) {
        if constexpr (SupportType<T, complex32>()) {
            reg = pset_b16(modeValue);
            if constexpr (regTrait.REG_NUM == 1) {
                punpack(reg, reg, LOWER);
            }
        } else {
            reg = pset_b32(modeValue);
        }
    } else if constexpr (sizeof(T) == 8) {
        reg = pset_b32(modeValue);
        if constexpr (regTrait.REG_NUM == 1) {
            punpack(reg, reg, LOWER);
        }
    }
    return reg;
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
__simd_callee__ inline void MaskInterleaveImpl(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1)
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
__simd_callee__ inline void MaskDeInterleaveImpl(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1)
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
    static_assert(SupportBytes<T, 2, 4>(), "MoveMask only support type b16/b32 on current device");

    if constexpr (sizeof(T) == 2) {
        return movp_b16();
    } else if constexpr (sizeof(T) == 4) {
        return movp_b32();
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_MASKREG_IMPL_H