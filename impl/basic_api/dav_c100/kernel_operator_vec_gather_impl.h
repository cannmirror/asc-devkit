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
 * \file kernel_operator_vec_gather_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#include "kernel_struct_gather.h"

namespace AscendC {
/* **************************************************************************************************
 * Gather                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void GatherbImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* offset,
    const uint32_t srcLength, const uint8_t repeatTime, const GatherRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Gatherb");
}

template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ uint32_t* srcOffsetLocal,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTime,
    const uint16_t dstRepStride)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Gather");
}

template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ uint32_t* srcOffsetLocal,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask[], const uint8_t repeatTime,
    const uint16_t dstRepStride)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Gather");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H