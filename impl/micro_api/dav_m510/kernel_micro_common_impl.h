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
 * \file kernel_micro_common_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_COMMON_IMPL_H
#define ASCENDC_MODULE_MICRO_COMMON_IMPL_H

#include "micro_api/kernel_micro_utils.h"
#include "../../../include/micro_api/kernel_micro_struct_intf.h"
#include "../../../include/micro_api/kernel_micro_maskreg_intf.h"

namespace AscendC {
namespace MicroAPI {
template <typename T, StoreDist dist> __aicore__ inline constexpr StoreDist GetStoreDist()
{
    if constexpr (dist == StoreDist::DIST_NORM) {
        static_assert(SupportBytes<T, 1, 2, 4, 8>(),
                      "StoreDist DIST_NORM only support type b8/b16/b32/b64 on current device");
        if constexpr (sizeof(T) == 1) {
            return StoreDist::DIST_NORM_B8;
        } else if constexpr (sizeof(T) == 2) {
            return StoreDist::DIST_NORM_B16;
        } else if constexpr (sizeof(T) == 4) {
            return StoreDist::DIST_NORM_B32;
        } else if constexpr (sizeof(T) == 8) {
            return StoreDist::DIST_NORM_B32;
        }
    }
    return dist;
}

template <typename T, const RegTrait& otherTrait = RegTraitNumOne> constexpr __aicore__ inline bool CheckRegTrait()
{
    constexpr RegTrait regTrait = T::trait;
    return regTrait.REG_NUM == otherTrait.REG_NUM;
}

#ifndef __ASC_NPU_HOST__
template <RoundMode mode> __aicore__ inline constexpr ::ROUND GetRound()
{
// To avoid naming conflicts of ROUND member variables in cpu debug,
// the names of the returned member variables are changed to be different from those of
// the ROUND enumeration class of the compiler.
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if constexpr (mode == RoundMode::CAST_RINT) {
        return ::ROUND::CAST_RINT;
    } else if constexpr (mode == RoundMode::CAST_ROUND) {
        return ::ROUND::CAST_ROUND;
    } else if constexpr (mode == RoundMode::CAST_FLOOR) {
        return ::ROUND::CAST_FLOOR;
    } else if constexpr (mode == RoundMode::CAST_CEIL) {
        return ::ROUND::CAST_CEIL;
    } else if constexpr (mode == RoundMode::CAST_TRUNC) {
        return ::ROUND::CAST_TRUNC;
    } else if constexpr (mode == RoundMode::CAST_ODD) {
        return ::ROUND::CAST_ODD;
    } else {
        return ::ROUND::CAST_HYBRID;
    }
#else
    if constexpr (mode == RoundMode::CAST_RINT) {
        return ::ROUND::R;
    } else if constexpr (mode == RoundMode::CAST_ROUND) {
        return ::ROUND::A;
    } else if constexpr (mode == RoundMode::CAST_FLOOR) {
        return ::ROUND::F;
    } else if constexpr (mode == RoundMode::CAST_CEIL) {
        return ::ROUND::C;
    } else if constexpr (mode == RoundMode::CAST_TRUNC) {
        return ::ROUND::Z;
    } else if constexpr (mode == RoundMode::CAST_ODD) {
        return ::ROUND::O;
    } else {
        return ::ROUND::H;
    }
#endif
}
#endif

#ifndef __ASC_NPU_HOST__
template <MaskMergeMode mode> __aicore__ inline constexpr auto GetMaskMergeMode()
{
// To avoid naming conflicts of mode struct in cpu debug.
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    return std::integral_constant<::CpuMode, static_cast<::CpuMode>(mode)>();
#else
    return std::integral_constant<::Mode, static_cast<::Mode>(mode)>();
#endif
}
#endif  // #ifndef __ASC_NPU_HOST__

template <typename T, typename U, typename ShortType>
__simd_callee__ inline void TraitOneToTaitTwoTmpl(T& dstReg, U& srcReg)
{
    using ActualT1 = typename U::ActualT;
    using ActualT2 = typename T::ActualT;
    static_assert(CheckRegTrait<T, RegTraitNumTwo>() && CheckRegTrait<U, RegTraitNumOne>(),
                  "T should be RegTraitNumTwo and U should be RegTraitNumOne");
    static_assert(sizeof(ActualT2) == (sizeof(ShortType) * 2) && sizeof(ActualT1) == (sizeof(ShortType) * 2),
                  "T and U should be 2 times of shortType lenth");
    RegTensor<ShortType> zeroReg;
    MaskReg maskFull = CreateMask<ShortType, MaskPattern::ALL>();
    Duplicate(zeroReg, 0, maskFull);
    DeInterleave((RegTensor<ShortType>&)dstReg.reg[0], (RegTensor<ShortType>&)dstReg.reg[1],
                 (RegTensor<ShortType>&)srcReg, zeroReg);
}

template <typename T, typename U, typename ShortType>
__simd_callee__ inline void TraitTwoToTaitOneTmpl(T& dstReg, U& srcReg)
{
    using ActualT1 = typename T::ActualT;
    using ActualT2 = typename U::ActualT;
    static_assert(CheckRegTrait<T, RegTraitNumOne>() && CheckRegTrait<U, RegTraitNumTwo>(),
                  "T should be RegTraitNumOne and U should be RegTraitNumTwo");
    static_assert(sizeof(ActualT2) == (sizeof(ShortType) * 2) && sizeof(ActualT1) == (sizeof(ShortType) * 2),
                  "U and T should be 2 times of shortType lenth");
    RegTensor<ShortType> dstRegShortFake;
    Interleave((RegTensor<ShortType>&)dstReg, dstRegShortFake, (RegTensor<ShortType>&)srcReg.reg[0],
               (RegTensor<ShortType>&)srcReg.reg[1]);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::MERGING, typename RegT>
__simd_callee__ inline void CopyMerging(RegT& dstReg, RegT& srcReg, MaskReg& mask)
{
    using ActualT = typename RegT::ActualT;
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (IsSameType<ActualT, bool>::value) {
        vmov((RegTensor<int8_t>&)dstReg, (RegTensor<int8_t>&)srcReg, mask, modeValue);
    } else if constexpr (sizeof(ActualT) == 1) {
        vmov((RegTensor<uint8_t>&)dstReg, (RegTensor<uint8_t>&)srcReg, mask, modeValue);
    } else if constexpr (sizeof(ActualT) == 2) {
        vmov((RegTensor<uint16_t>&)dstReg, (RegTensor<uint16_t>&)srcReg, mask, modeValue);
    } else if constexpr (sizeof(ActualT) == 4) {
        vmov((RegTensor<uint32_t>&)dstReg, (RegTensor<uint32_t>&)srcReg, mask, modeValue);
    } else if constexpr (sizeof(ActualT) == 8) {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            constexpr auto lowerDist =
                std::integral_constant<::HiloPart, static_cast<::HiloPart>(HighLowPart::LOWEST)>();
            MaskReg dstMask;
            MaskReg tmpMask;
            MaskReg dumpMask;
            ppack(tmpMask, mask, lowerDist);
            pintlv_b32(dstMask, dumpMask, tmpMask, tmpMask);
            vmov((RegTensor<uint32_t> &)dstReg, (RegTensor<uint32_t> &)srcReg, dstMask, modeValue);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            vmov((RegTensor<uint32_t> &)dstReg.reg[0], (RegTensor<uint32_t> &)srcReg.reg[0], mask, modeValue);
            vmov((RegTensor<uint32_t> &)dstReg.reg[1], (RegTensor<uint32_t> &)srcReg.reg[1], mask, modeValue);
        }
    }
}

template <typename T, typename U>
__simd_callee__ inline void B64TraitOneToTaitTwo(T& dstReg, U& srcReg)
{
    TraitOneToTaitTwoTmpl<T, U, uint32_t>(dstReg, srcReg);
}

template <typename T, typename U>
__simd_callee__ inline void B64TraitTwoToTaitOne(T& dstReg, U& srcReg)
{
    TraitTwoToTaitOneTmpl<T, U, uint32_t>(dstReg, srcReg);
}

template <typename T, typename U>
__simd_callee__ inline void B32TraitOneToTaitTwo(T& dstReg, U& srcReg)
{
    TraitOneToTaitTwoTmpl<T, U, uint16_t>(dstReg, srcReg);
}

template <typename T, typename U>
__simd_callee__ inline void B32TraitTwoToTaitOne(T& dstReg, U& srcReg)
{
    TraitTwoToTaitOneTmpl<T, U, uint16_t>(dstReg, srcReg);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_COMMON_IMPL_H
