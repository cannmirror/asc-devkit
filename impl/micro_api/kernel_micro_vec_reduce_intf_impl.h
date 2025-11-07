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
 * \file kernel_micro_vec_reduce_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_VEC_REDUCE_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_VEC_REDUCE_INTERFACE_IMPL_H

#if __NPU_ARCH__ == 2103
#include "micro_api/dav_l210/kernel_micro_vec_reduce_impl.h"
#elif __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_vec_reduce_impl.h"
#elif __NPU_ARCH__ == 3103
#include "micro_api/dav_l310/kernel_micro_vec_reduce_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_vec_reduce_impl.h"
#else 
#include "micro_api/dav_c310/kernel_micro_vec_reduce_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T, typename U, MaskMergeMode mode, typename S, typename V>
__simd_callee__ inline void ReduceSum(S &dstReg, V srcReg, MaskReg mask)
{
    ReduceSumImpl<T, U, mode, S, V>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void ReduceMax(U &dstReg, U srcReg, MaskReg mask)
{
    ReduceMaxImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void ReduceMin(U &dstReg, U srcReg, MaskReg mask)
{
    ReduceMinImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void ReduceSumWithDataBlock(U &dstReg, U srcReg, MaskReg mask)
{
    ReduceSumWithDataBlockImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void ReduceMaxWithDataBlock(U &dstReg, U srcReg, MaskReg mask)
{
    ReduceMaxWithDataBlockImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void ReduceMinWithDataBlock(U &dstReg, U srcReg, MaskReg mask)
{
    ReduceMinWithDataBlockImpl<T, mode, U>(dstReg, srcReg, mask);
}

template <typename T, MaskMergeMode mode, typename U>
__simd_callee__ inline void PairReduceSum(U &dstReg, U srcReg, MaskReg mask)
{
    PairReduceSumImpl<T, mode, U>(dstReg, srcReg, mask);
}
}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_KERNEL_MICRO_VEC_REDUCE_INTERFACE_IMPL_H