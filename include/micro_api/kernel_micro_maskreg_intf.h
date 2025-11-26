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
 * \file kernel_micro_maskreg_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_H
#define ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_H

#include "kernel_micro_common_intf.h"

namespace AscendC {
namespace MicroAPI {
template <typename T, const RegTrait& regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg UpdateMask(uint32_t& scalarValue);

template <typename T, MaskPattern mode = MaskPattern::ALL, const RegTrait& regTrait = RegTraitNumOne>
__simd_callee__ inline MaskReg CreateMask();

__simd_callee__ inline void MaskNot(MaskReg& dst, MaskReg& src, MaskReg& mask);

__simd_callee__ inline void MaskAnd(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask);

template <typename T = DefaultType, int16_t offset, typename U>
__simd_callee__ inline void MaskGenWithRegTensor(MaskReg& dst, U& srcReg);

__simd_callee__ inline void MaskOr(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask);

__simd_callee__ inline void MaskXor(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask);

__simd_callee__ inline void MaskMov(MaskReg& dst, MaskReg& src, MaskReg& mask);

__simd_callee__ inline void MaskMov(MaskReg& dst, MaskReg& src);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2103 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3103 || \
    __NPU_ARCH__ == 3113)
template <typename T>
__aicore__ inline void MaskSlide(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, const int16_t slideAmount);
#endif

template <typename T>
__simd_callee__ inline void MaskInterleave(MaskReg& dst0, MaskReg& dst1, MaskReg& src0, MaskReg& src1);

template <typename T>
__simd_callee__ inline void MaskDeInterleave(MaskReg& dst0, MaskReg& dst1, MaskReg& src0, MaskReg& src1);

__simd_callee__ inline void MaskSel(MaskReg &dst, MaskReg &src0, MaskReg& src1, MaskReg& mask);

template <HighLowPart part = HighLowPart::LOWEST>
__simd_callee__ inline void MaskPack(MaskReg& dst, MaskReg& src);

template <HighLowPart part = HighLowPart::LOWEST>
__simd_callee__ inline void MaskUnPack(MaskReg& dst, MaskReg& src);

template <typename T>
__simd_callee__ inline MaskReg MoveMask();
} // namespace MicroAPI
} // namespace AscendC

#include "impl/micro_api/kernel_micro_maskreg_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_H