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
 * \file kernel_micro_vec_vconv_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_VCONV_INTERFACE_H
#define ASCENDC_MODULE_MICRO_VEC_VCONV_INTERFACE_H

#include "micro_api_inc/kernel_micro_common_intf.h"
namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename U = DefaultType, const CastTrait& trait = castTrait, typename S,
          typename V>
__simd_callee__ inline void Cast(S& dstReg, V& srcReg, MaskReg& mask);

// truncate f162f16/f322f32/bf162bf16
template <typename T = DefaultType, RoundMode roundMode = RoundMode::CAST_NONE,
          MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void Truncate(S& dstReg, S& srcReg, MaskReg& mask);
} // namespace MicroAPI
} // namespace AscendC

#include "impl/micro_api/kernel_micro_vec_vconv_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_VEC_VCONV_INTERFACE_H