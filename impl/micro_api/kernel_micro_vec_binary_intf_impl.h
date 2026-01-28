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
 * \file kernel_micro_vec_binary_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_VEC_BINARY_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_VEC_BINARY_INTERFACE_IMPL_H

#include "micro_api/kernel_micro_vec_unary_intf.h"
#include "micro_api/kernel_micro_vec_binary_scalar_intf.h"

#if __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_vec_binary_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_vec_binary_impl.h"
#elif __NPU_ARCH__ == 5102
#include "micro_api/dav_m510/kernel_micro_vec_binary_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_vec_binary_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Add(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    AddImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Sub(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    SubImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Mul(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    MulImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, auto mode, typename U>
__simd_callee__ inline void Div(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    DivImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Max(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    MaxImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Min(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    MinImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, typename U, MaskMergeMode mode, typename S, typename V>
__simd_callee__ inline void ShiftLeft(S& dstReg, S& srcReg0, V& srcReg1, MaskReg& mask)
{
    ShiftLeftImpl<T, U, mode, S, V>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, typename U, MaskMergeMode mode, typename S, typename V>
__simd_callee__ inline void ShiftRight(S& dstReg, S& srcReg0, V& srcReg1, MaskReg& mask)
{
    ShiftRightImpl<T, U, mode, S, V>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void And(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    AndImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Or(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    OrImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Xor(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    XorImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Prelu(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    PreluImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, typename U>
__simd_callee__ inline void Mull(U& dstReg0, U& dstReg1, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    MullImpl<T, U>(dstReg0, dstReg1, srcReg0, srcReg1, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void MulAddDst(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    MulAddDstImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Mula(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    MulAddDstImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T = DefaultType, typename U>
__simd_callee__ inline void AddCarryOut(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    AddCarryOutImpl<T, U>(carry, dstReg, srcReg0, srcReg1, mask);
}
template <typename T, typename U>
__simd_callee__ inline void Add(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    AddCarryOutImpl<T, U>(carry, dstReg, srcReg0, srcReg1, mask);
}

template <typename T = DefaultType, typename U>
__simd_callee__ inline void SubCarryOut(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    SubCarryOutImpl<T, U>(carry, dstReg, srcReg0, srcReg1, mask);
}
template <typename T, typename U>
__simd_callee__ inline void Sub(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    SubCarryOutImpl<T, U>(carry, dstReg, srcReg0, srcReg1, mask);
}

template <typename T = DefaultType, typename U>
__simd_callee__ inline void AddCarryOuts(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& carrySrc,
                                         MaskReg& mask)
{
    AddCarryOutsImpl<T, U>(carry, dstReg, srcReg0, srcReg1, carrySrc, mask);
}
template <typename T, typename U>
__simd_callee__ inline void AddC(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& carrySrc,
                                 MaskReg& mask)
{
    AddCarryOutsImpl<T, U>(carry, dstReg, srcReg0, srcReg1, carrySrc, mask);
}

template <typename T = DefaultType, typename U>
__simd_callee__ inline void SubCarryOuts(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& carrySrc,
                                         MaskReg& mask)
{
    SubCarryOutsImpl<T, U>(carry, dstReg, srcReg0, srcReg1, carrySrc, mask);
}
template <typename T, typename U>
__simd_callee__ inline void SubC(MaskReg& carry, U& dstReg, U& srcReg0, U& srcReg1, MaskReg& carrySrc,
                                 MaskReg& mask)
{
    SubCarryOutsImpl<T, U>(carry, dstReg, srcReg0, srcReg1, carrySrc, mask);
}
} // namespace MicroAPI
} // namespace AscendC

#endif // ASCENDC_KERNEL_MICRO_VEC_BINARY_INTERFACE_IMPL_H