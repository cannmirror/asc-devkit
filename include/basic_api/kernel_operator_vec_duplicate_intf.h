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
 * \file kernel_operator_vec_duplicate_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_INTERFACE_H
#include "kernel_tensor.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Duplicate                                            *
 * ************************************************************************************************* */
/*
 * @ingroup Duplicate Level 0
 * @brief dst[i] = scalar
 * @param [out] dst output LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] dstBlockStride dst block stride
 * @param [in] dstRepeatStride dst repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask,
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, uint64_t mask[],
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride);

/*
 * @ingroup Duplicate Level 2
 * @brief dst = dst[i] = scalar
 * @param [out] dst output LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count);
} // namespace AscendC
#pragma end_pipe

#include "../../impl/basic_api/kernel_operator_vec_duplicate_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_INTERFACE_H
