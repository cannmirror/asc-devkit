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
 * \file kernel_operator_vec_scatter_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_SCATTER_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_SCATTER_INTERFACE_H
#include "kernel_tensor.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup scatter Level 0
 * @brief scatter element from dst according to dstOffset
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] dstOffset input LocalTensor
 * @param [in] mask valid element count
 * @param [in] repeatTime repeat times
 * @param [in] srcRepStride src repeat stride
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint64_t mask,
    const uint8_t repeatTime, const uint8_t srcRepStride);

/*
 * @ingroup scatter Level 0
 * @brief scatter element from dst according to dstOffset
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] dstOffset input LocalTensor
 * @param [in] mask valid element count(bit mode)
 * @param [in] repeatTime repeat times
 * @param [in] srcRepStride src repeat stride
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint64_t mask[],
    const uint8_t repeatTime, const uint8_t srcRepStride);

/*
 * @ingroup scatter Level 2
 * @brief scatter element from dst according to dstOffset
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] dstOffset input LocalTensor
 * @param [in] count element count
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint32_t>& dstOffset, const uint32_t dstBaseAddr, const uint32_t count);
} // namespace AscendC
#pragma end_pipe

#include "../../impl/basic_api/kernel_operator_vec_scatter_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_SCATTER_INTERFACE_H