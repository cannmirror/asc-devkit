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
 * \file kernel_operator_vec_unary_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_UNARY_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_UNARY_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_unary.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Unary                                              *
 * ************************************************************************************************* */

/* **************************************** Relu ****************************************** */
/*
 * @ingroup Relu Level 0
 * @brief dst[i] = (src[i] < 0) ? 0 : src[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Relu(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Relu(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Relu Level 2
 * @brief dst[i] = (src[i] < 0) ? 0 : src[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Relu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);

/* **************************************** Exp ****************************************** */
/*
 * @ingroup Exp Level 0
 * @brief dst[i] = exp(src[i])
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Exp(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Exp(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Exp Level 2
 * @brief dst[i] = exp(src[i])
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Exp(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);

/* **************************************** Ln ****************************************** */
/*
 * @ingroup Ln Level 0
 * @brief dst[i] = Ln(src[i])
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Ln(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Ln(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Ln Level 2
 * @brief dst[i] = Ln(src[i])
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Ln(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);

/* **************************************** Abs ****************************************** */
/*
 * @ingroup Abs Level 0
 * @brief dst[i] = abs(src[i])
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Abs(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Abs(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Abs Level 2
 * @brief dst[i] = abs(src[i])
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Abs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);

/* **************************************** Reciprocal ****************************************** */
/*
 * @ingroup Rec Level 0
 * @brief dst[i] = 1/src[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Rec Level 2
 * @brief dst[i] = 1/src[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const int32_t& count);

/* **************************************** Rsqrt ****************************************** */
/*
 * @ingroup Rsqrt Level 0
 * @brief dst[i] = 1/sqrt(src[i])
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Rsqrt Level 2
 * @brief dst[i] = 1/sqrt(src[i])
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);

/* **************************************** Sqrt ****************************************** */
/*
 * @ingroup Sqrt Level 0
 * @brief dst[i] = src[i]^(0.5)
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Sqrt(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Sqrt(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Sqrt Level 2
 * @brief dst[i] = src[i]^(0.5)
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Sqrt(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);

/* **************************************** Not ****************************************** */
/*
 * @ingroup Not Level 0
 * @brief dst[i] = ~src[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Not(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Not(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Not Level 2
 * @brief dst[i] = ~src[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Not(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count);
} // namespace AscendC
#pragma end_pipe

#include "../../impl/basic_api/kernel_operator_vec_unary_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_UNARY_INTERFACE_H
