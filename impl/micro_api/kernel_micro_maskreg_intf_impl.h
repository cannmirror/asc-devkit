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
 * \file kernel_micro_maskreg_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_IMPL_H
#define ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_IMPL_H

#if __NPU_ARCH__ == 2103
#include "micro_api/dav_l210/kernel_micro_maskreg_impl.h"
#elif __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_maskreg_impl.h"
#elif __NPU_ARCH__ == 3103
#include "micro_api/dav_l310/kernel_micro_maskreg_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_maskreg_impl.h"
#else 
#include "micro_api/dav_c310/kernel_micro_maskreg_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T, const RegTrait& regTrait>
__simd_callee__ inline MaskReg UpdateMask(uint32_t &scalar)
{
    return UpdateMaskImpl<T, regTrait>(scalar);
}

template <typename T, MaskPattern mode, const RegTrait& regTrait>
__simd_callee__ inline MaskReg CreateMask()
{
    return CreateMaskImpl<T, mode, regTrait>();
}

__simd_callee__ inline void MaskNot(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask)
{
    MaskNotImpl(dstMask, srcMask, mask);
}

template <typename T, int16_t Offset, typename RegT>
__simd_callee__ inline void MaskGenWithRegTensor(MaskReg &dstMask, RegT &srcReg)
{
    MaskGenWithRegTensorImpl<T, Offset, RegT>(dstMask, srcReg);
}

__simd_callee__ inline void MaskAnd(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    MaskAndImpl(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskOr(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    MaskOrImpl(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskXor(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    MaskXorImpl(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskMov(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask)
{
    MaskMovImpl(dstMask, srcMask, mask);
}

__simd_callee__ inline void MaskMov(MaskReg &dstMask, MaskReg &srcMask)
{
    MaskMovImpl(dstMask, srcMask);
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2103 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3103 || \
    __NPU_ARCH__ == 3113)
template <typename T>
__aicore__ inline void MaskSlide(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, const int16_t slideAmount)
{
    MaskSlideImpl<T>(dstMask, srcMask0, srcMask1, slideAmount);
}
#endif

template <typename T>
__simd_callee__ inline void MaskInterleave(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1)
{
    MaskInterleaveImpl<T>(dstMask0, dstMask1, srcMask0, srcMask1);
}

template <typename T>
__simd_callee__ inline void MaskDeInterleave(MaskReg &dstMask0, MaskReg &dstMask1, MaskReg &srcMask0, MaskReg &srcMask1)
{
    MaskDeInterleaveImpl<T>(dstMask0, dstMask1, srcMask0, srcMask1);
}

__simd_callee__ inline void MaskSel(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    MaskSelImpl(dstMask, srcMask0, srcMask1, mask);
}

template <HighLowPart part>
__simd_callee__ inline void MaskPack(MaskReg &dstMask, MaskReg &srcMask)
{
    MaskPackImpl<part>(dstMask, srcMask);
}

template <HighLowPart part>
__simd_callee__ inline void MaskUnPack(MaskReg &dstMask, MaskReg &srcMask)
{
    MaskUnPackImpl<part>(dstMask, srcMask);
}

template <typename T>
__simd_callee__ inline MaskReg MoveMask()
{
    return MoveMaskImpl<T>();
}
}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_IMPL_H