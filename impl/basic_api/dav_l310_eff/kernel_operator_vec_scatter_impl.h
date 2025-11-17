/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_vec_scatter_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H


namespace AscendC {
/* **************************************************************************************************
 * Scatter                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void ScatterImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ uint32_t* dstOffset,
    const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask, const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Don't support Scatter"); });
    return;
}

template <typename T>
__aicore__ inline void ScatterImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ uint32_t* dstOffset,
    const uint32_t dstLength, const uint32_t dstBaseAddr, const uint64_t mask[2], const uint8_t repeatTime,
    const uint8_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Don't support Scatter"); });
    return;
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_SCATTER_IMPL_H