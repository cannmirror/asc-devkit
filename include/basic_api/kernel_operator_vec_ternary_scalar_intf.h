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
 * \file kernel_operator_vec_ternary_scalar_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_unary.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup Axpy Level 0
 * @brief dst[i] = src[i]*scalar + dst[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalarValue input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Axpy Level 2
 * @brief dst[i] = src[i]*scalar + dst[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalarValue input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void Axpy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const U& scalarValue,
    const int32_t& count);
} // namespace AscendC
#pragma end_pipe

#include "../../impl/basic_api/kernel_operator_vec_ternary_scalar_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_INTERFACE_H
