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
 * \file kernel_micro_vec_fused_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_VEC_FUSED_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_VEC_FUSED_INTERFACE_IMPL_H

#if __NPU_ARCH__ == 2103
#include "micro_api/dav_l210/kernel_micro_vec_fused_impl.h"
#elif __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_vec_fused_impl.h"
#elif __NPU_ARCH__ == 3103
#include "micro_api/dav_l310/kernel_micro_vec_fused_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_vec_fused_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_vec_fused_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T0, typename T1, typename T2, RegLayout layout, typename T3, typename T4>
__simd_callee__ inline void FusedMulsCast(T3 &dstReg, T4 &srcReg, T2 scalar, MaskReg &mask)
{
    FusedMulsCastImpl<T0, T1, T2, layout, T3, T4>(dstReg, srcReg, scalar, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void FusedAbsSub(U &dstReg, U &srcReg0, U &srcReg1, MaskReg &mask)
{
    FusedAbsSubImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

template <typename T, typename U, RegLayout layout, MaskMergeMode mode, typename S, typename V>
__simd_callee__ inline void FusedExpSub(S &dstReg, V &srcReg0, V &srcReg1, MaskReg &mask)
{
    FusedExpSubImpl<T, U, layout, mode, S, V>(dstReg, srcReg0, srcReg1, mask);
}
template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void FusedMulDstAdd(U &dstReg, U &srcReg0, U &srcReg1, MaskReg &mask)
{
    FusedMulDstAddImpl<T, mode, U>(dstReg, srcReg0, srcReg1, mask);
}

}
}  // namespace AscendC
#endif  // ASCENDC_KERNEL_MICRO_VEC_FUSED_INTERFACE_IMPL_H