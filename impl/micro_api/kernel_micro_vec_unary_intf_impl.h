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
 * \file kernel_micro_vec_unary_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_VEC_SINGLE_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_VEC_SINGLE_INTERFACE_IMPL_H

#if __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_vec_unary_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_vec_unary_impl.h"
#elif __NPU_ARCH__ == 5102
#include "micro_api/dav_m510/kernel_micro_vec_unary_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_vec_unary_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Abs(U& dstReg, U& srcReg, MaskReg& mask)
{
    AbsImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, typename U, MaskMergeMode mode, typename S, typename V>
__simd_callee__ inline void Abs(S& dstReg, V& srcReg, MaskReg& mask)
{
    AbsImpl<T, U, mode, S, V>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Relu(U& dstReg, U& srcReg, MaskReg& mask)
{
    ReluImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, auto mode, typename U>
__simd_callee__ inline void Exp(U& dstReg, U& srcReg, MaskReg& mask)
{
    ExpImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, auto mode, typename U>
__simd_callee__ inline void Sqrt(U& dstReg, U& srcReg, MaskReg& mask)
{
    SqrtImpl<T, mode, U>(dstReg, srcReg, mask);
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || \
    __NPU_ARCH__ == 3113)
template <typename T, MaskMergeMode mode, typename RegT>
__aicore__ inline void Rsqrt(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    RsqrtImpl<T, mode, RegT>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename RegT>
__aicore__ inline void Rec(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    RecImpl<T, mode, RegT>(dstReg, srcReg, mask);
}

template <typename T, typename SrcT, MaskMergeMode mode, typename RegT, typename RegSrcT>
__aicore__ inline void CountBit(RegT &dstReg, RegSrcT &srcReg, MaskReg &mask)
{
    CountBitImpl<T, SrcT, mode, RegT, RegSrcT>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename RegT>
__aicore__ inline void CountLeadingSignBits(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    CountLeadingSignBitsImpl<T, mode, RegT>(dstReg, srcReg, mask);
}
#endif

template <typename T, auto mode, typename U>
__simd_callee__ inline void Ln(U& dstReg, U& srcReg, MaskReg& mask)
{
    LogImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, auto mode, typename U>
__simd_callee__ inline void Log(U& dstReg, U& srcReg, MaskReg& mask)
{
    LogImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, auto mode, typename U>
__simd_callee__ inline void Log2(U& dstReg, U& srcReg, MaskReg& mask)
{
    Log2Impl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, auto mode, typename U>
__simd_callee__ inline void Log10(U& dstReg, U& srcReg, MaskReg& mask)
{
    Log10Impl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Neg(U& dstReg, U& srcReg, MaskReg& mask)
{
    NegImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void Not(U& dstReg, U& srcReg, MaskReg& mask)
{
    NotImpl<T, mode, U>(dstReg, srcReg, mask);
}
} // namespace MicroAPI
} // namespace AscendC

#endif // ASCENDC_KERNEL_MICRO_VEC_SINGLE_INTERFACE_IMPL_H