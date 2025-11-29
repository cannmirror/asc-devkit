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
 * \file kernel_micro_gather_mask_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_GATHER_MASK_INTERFACE_H
#define ASCENDC_MODULE_MICRO_GATHER_MASK_INTERFACE_H

#include "kernel_micro_common_intf.h"

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, GatherMaskMode store = GatherMaskMode::NO_STORE_REG, typename U>
__simd_callee__ inline void Squeeze(U& dstReg, U& srcReg, MaskReg& mask);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void Unsqueeze(U& dstReg, MaskReg& mask);

template <SpecialPurposeReg spr>
__aicore__ inline int64_t GetSpr();

template <SpecialPurposeReg spr>
__simd_callee__ inline void ClearSpr();

template <typename T = DefaultType, typename U = DefaultType, typename S, typename V>
__simd_callee__ inline void Gather(S& dstReg, S& srcReg, V& indexReg);
} // namespace MicroAPI
} // namespace AscendC

#include "impl/micro_api/kernel_micro_gather_mask_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_COPY_INTERFACE_H