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
 * \file kernel_operator_vec_binary_continuous_impl.h
 * \brief AscendC l510 support vec binary continuous data api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * Add                                             *
 * ************************************************************************************************* */
// Add::Level 2
template <typename T>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

__aicore__ inline void AddImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void AddImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void AddImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void AddImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * Sub                                             *
 * ************************************************************************************************* */
// Sub::Level 2
template <typename T>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

__aicore__ inline void SubImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void SubImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void SubImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void SubImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * Mul                                             *
 * ************************************************************************************************* */
// Mul::Level 2
template <typename T>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MulImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MulImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MulImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MulImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * Div                                             *
 * ************************************************************************************************* */
// Div::Level 2
template <typename T>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void DivImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void DivImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * Max                                             *
 * ************************************************************************************************* */
// Max::Level 2
template <typename T>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MaxImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MaxImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MaxImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MaxImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * Min                                             *
 * ************************************************************************************************* */
// Min::Level 2
template <typename T>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MinImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MinImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MinImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void MinImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * And                                             *
 * ************************************************************************************************* */
// And::Level 2
template <typename T>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void AndImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void AndImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * Or                                             *
 * ************************************************************************************************* */
// Or::Level 2
template <typename T>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void OrImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

__aicore__ inline void OrImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
// FusedMulAdd::Level 2
template <typename T>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
// FusedMulAddRelu::Level 2
template <typename T>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "currfunc is not supported on current device!"); });
}
/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
// MulAddDst::Level 2
template <typename T, typename U>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "MulAddDst is not supported on current device!"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
