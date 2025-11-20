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
 * \file kernel_micro_vec_binary_scalar_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_VEC_BINARY_SCALAR_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_VEC_BINARY_SCALAR_INTERFACE_IMPL_H

#include "kernel_micro_vec_binary_intf_impl.h"

#if __NPU_ARCH__ == 2103
#include "micro_api/dav_l210/kernel_micro_vec_binary_scalar_impl.h"
#elif __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_vec_binary_scalar_impl.h"
#elif __NPU_ARCH__ == 3103
#include "micro_api/dav_l310/kernel_micro_vec_binary_scalar_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_vec_binary_scalar_impl.h"
#else 
#include "micro_api/dav_c310/kernel_micro_vec_binary_scalar_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T, typename U, MaskMergeMode mode, typename S>
__simd_callee__ inline void Adds(S &dstReg, S &srcReg, U scalar, MaskReg &mask)
{
    AddsImpl<T, U, mode, S>(dstReg, srcReg, scalar, mask);
}

template <typename T, typename U, MaskMergeMode mode, typename S>
__simd_callee__ inline void Muls(S &dstReg, S &srcReg, U scalar, MaskReg &mask)
{
    MulsImpl<T, U, mode, S>(dstReg, srcReg, scalar, mask);
}

template <typename T, typename U, MaskMergeMode mode, typename S>
__simd_callee__ inline void Maxs(S &dstReg, S &srcReg, U scalar, MaskReg &mask)
{
    MaxsImpl<T, U, mode, S>(dstReg, srcReg, scalar, mask);
}

template <typename T, typename U, MaskMergeMode mode, typename S>
__simd_callee__ inline void Mins(S &dstReg, S &srcReg, U scalar, MaskReg &mask)
{
    MinsImpl<T, U, mode, S>(dstReg, srcReg, scalar, mask);
}

template <typename T, typename U, MaskMergeMode mode, typename S>
__simd_callee__ inline void ShiftLefts(S &dstReg, S &srcReg, U scalar, MaskReg &mask)
{
    ShiftLeftsImpl<T, U, mode, S>(dstReg, srcReg, scalar, mask);
}

template <typename T, typename U, MaskMergeMode mode, typename S>
__simd_callee__ inline void ShiftRights(S &dstReg, S &srcReg, U scalar, MaskReg &mask)
{
    ShiftRightsImpl<T, U, mode, S>(dstReg, srcReg, scalar, mask);
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2103 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3103 || \
    __NPU_ARCH__ == 3113)
template <typename T, typename U, MaskMergeMode mode, typename S>
__aicore__ inline void Rounds(S &dstReg, S &srcReg0, U scalar, MaskReg &mask)
{
    RoundsImpl<T, U, mode, S>(dstReg, srcReg0, scalar, mask);
}
#endif

template <typename T, typename U, MaskMergeMode mode, typename S>
__simd_callee__ inline void LeakyRelu(S &dstReg, S &srcReg, U scalar, MaskReg &mask)
{
    LeakyReluImpl<T, U, mode, S>(dstReg, srcReg, scalar, mask);
}
}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_KERNEL_MICRO_VEC_BINARY_SCALAR_INTERFACE_IMPL_H