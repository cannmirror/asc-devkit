/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file kernel_micro_maskreg_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_H
#define ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_H

#include "kernel_micro_common_intf.h"

namespace AscendC {
namespace MicroAPI {
template <typename T, const RegTrait &regTrait = RegTraitNumOne> __simd_callee__ inline MaskReg UpdateMask(uint32_t &scalar);

template <typename T, MaskPattern mode = MaskPattern::ALL, const RegTrait &regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg CreateMask();

__simd_callee__ inline void MaskNot(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask);

__simd_callee__ inline void MaskAnd(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask);

template <typename T = DefaultType, int16_t Offset, typename U>
__simd_callee__ inline void MaskGenWithRegTensor(MaskReg &dstMask, U &srcReg);

__simd_callee__ inline void MaskOr(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask);

__simd_callee__ inline void MaskXor(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask);

__simd_callee__ inline void MaskMov(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask);

__simd_callee__ inline void MaskMov(MaskReg &dstMask, MaskReg &srcMask);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2103 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3103 || \
    __NPU_ARCH__ == 3113)
template <typename T>
__aicore__ inline void MaskSlide(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, const int16_t slideAmount);
#endif

template <typename T>
__simd_callee__ inline void MaskInterleave(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1);

template <typename T>
__simd_callee__ inline void MaskDeInterleave(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1);

__simd_callee__ inline void MaskSel(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask);

template <HighLowPart part = HighLowPart::LOWEST> __simd_callee__ inline void MaskPack(MaskReg &dstMask, MaskReg &srcMask);

template <HighLowPart part = HighLowPart::LOWEST> __simd_callee__ inline void MaskUnPack(MaskReg &dstMask, MaskReg &srcMask);

template <typename T> __simd_callee__ inline MaskReg MoveMask();
} // namespace MicroAPI
} // namespace AscendC

#include "impl/micro_api/kernel_micro_maskreg_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_H