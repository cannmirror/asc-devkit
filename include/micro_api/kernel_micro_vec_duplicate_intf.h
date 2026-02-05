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
 * \file kernel_micro_vec_duplicate_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_DUPLICATE_INTERFACE_H
#define ASCENDC_MODULE_MICRO_VEC_DUPLICATE_INTERFACE_H

#include "kernel_micro_common_intf.h"
#include "kernel_micro_utils.h"

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename U, typename S>
__simd_callee__ inline void Duplicate(S& dstReg, U scalarValue);

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U, typename S>
__simd_callee__ inline void Duplicate(S& dstReg, U scalarValue, MaskReg& mask);

template <typename T = DefaultType, HighLowPart pos = HighLowPart::LOWEST, MaskMergeMode mode = MaskMergeMode::ZEROING,
          typename S>
__simd_callee__ inline void Duplicate(S& dstReg, S& srcReg, MaskReg& mask);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void Interleave(U& dstReg0, U& dstReg1, U& srcReg0, U& srcReg1);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void DeInterleave(U& dstReg0, U& dstReg1, U& srcReg0, U& srcReg1);
} // namespace MicroAPI
} // namespace AscendC

#include "impl/micro_api/kernel_micro_vec_duplicate_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_VEC_DUPLICATE_INTERFACE_H