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

#if __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_maskreg_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_maskreg_impl.h"
#elif __NPU_ARCH__ == 5102
#include "micro_api/dav_m510/kernel_micro_maskreg_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_maskreg_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T, const RegTrait& regTrait>
__simd_callee__ inline MaskReg UpdateMask(uint32_t& scalarValue)
{
    return UpdateMaskImpl<T, regTrait>(scalarValue);
}

template <typename T, MaskPattern mode, const RegTrait& regTrait>
__simd_callee__ inline MaskReg CreateMask()
{
    return CreateMaskImpl<T, mode, regTrait>();
}

__simd_callee__ inline void MaskNot(MaskReg& dst, MaskReg& src, MaskReg& mask)
{
    MaskNotImpl(dst, src, mask);
}
__simd_callee__ inline void Not(MaskReg& dst, MaskReg& src, MaskReg& mask)
{
    MaskNotImpl(dst, src, mask);
}

template <typename T, int16_t offset, typename U>
__simd_callee__ inline void MaskGenWithRegTensor(MaskReg& dst, U& srcReg)
{
    MaskGenWithRegTensorImpl<T, offset, U>(dst, srcReg);
}

__simd_callee__ inline void MaskAnd(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask)
{
    MaskAndImpl(dst, src0, src1, mask);
}
__simd_callee__ inline void And(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask)
{
    MaskAndImpl(dst, src0, src1, mask);
}

__simd_callee__ inline void MaskOr(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask)
{
    MaskOrImpl(dst, src0, src1, mask);
}
__simd_callee__ inline void Or(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask)
{
    MaskOrImpl(dst, src0, src1, mask);
}

__simd_callee__ inline void MaskXor(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask)
{
    MaskXorImpl(dst, src0, src1, mask);
}
__simd_callee__ inline void Xor(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask)
{
    MaskXorImpl(dst, src0, src1, mask);
}

__simd_callee__ inline void MaskMov(MaskReg& dst, MaskReg& src, MaskReg& mask)
{
    MaskMovImpl(dst, src, mask);
}
__simd_callee__ inline void Move(MaskReg& dst, MaskReg& src, MaskReg& mask)
{
    MaskMovImpl(dst, src, mask);
}

__simd_callee__ inline void MaskMov(MaskReg& dst, MaskReg& src)
{
    MaskMovImpl(dst, src);
}
__simd_callee__ inline void Move(MaskReg& dst, MaskReg& src)
{
    MaskMovImpl(dst, src);
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || \
    __NPU_ARCH__ == 3113)
template <typename T>
__aicore__ inline void MaskSlide(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, const int16_t slideAmount)
{
    MaskSlideImpl<T>(dstMask, srcMask0, srcMask1, slideAmount);
}
#endif

template <typename T>
__simd_callee__ inline void MaskInterleave(MaskReg& dst0, MaskReg& dst1, MaskReg& src0, MaskReg& src1)
{
    MaskInterleaveImpl<T>(dst0, dst1, src0, src1);
}

template <typename T>
__simd_callee__ inline void MaskDeInterleave(MaskReg& dst0, MaskReg& dst1, MaskReg& src0, MaskReg& src1)
{
    MaskDeInterleaveImpl<T>(dst0, dst1, src0, src1);
}

__simd_callee__ inline void MaskSel(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg &mask)
{
    MaskSelImpl(dst, src0, src1, mask);
}
__simd_callee__ inline void Select(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg &mask)
{
    MaskSelImpl(dst, src0, src1, mask);
}

template <HighLowPart part = HighLowPart::LOWEST>
__simd_callee__ inline void MaskPack(MaskReg& dst, MaskReg& src)
{
    MaskPackImpl<part>(dst, src);
}
template <HighLowPart part>
__simd_callee__ inline void Pack(MaskReg& dst, MaskReg& src)
{
    MaskPackImpl<part>(dst, src);
}

template <HighLowPart part = HighLowPart::LOWEST>
__simd_callee__ inline void MaskUnPack(MaskReg& dst, MaskReg& src)
{
    MaskUnPackImpl<part>(dst, src);
}
template <HighLowPart part>
__simd_callee__ inline void UnPack(MaskReg& dst, MaskReg& src)
{
    MaskUnPackImpl<part>(dst, src);
}

template <typename T>
__simd_callee__ inline MaskReg MoveMask()
{
    return MoveMaskImpl<T>();
}
} // namespace MicroAPI
} // namespace AscendC

#endif // ASCENDC_MODULE_MICRO_MASKREG_INTERFACE_IMPL_H