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

/*!
 * \file kernel_operator_vec_gather_mask_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_gather.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

namespace AscendC {
#pragma begin_pipe(V)
template <typename T, typename U, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<U>& src1Pattern, const bool reduceMode, const uint32_t mask,
    const GatherMaskParams& gatherMaskParams, uint64_t& rsvdCnt);

template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const uint8_t src1Pattern, const bool reduceMode, const uint32_t mask, const GatherMaskParams& gatherMaskParams,
    uint64_t& rsvdCnt);
#pragma end_pipe
} // namespace AscendC

#include "../../impl/basic_api/kernel_operator_vec_gather_mask_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_INTERFACE_H