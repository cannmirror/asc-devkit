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
 * \file kernel_operator_vec_binary_scalar_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_unary.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Adds                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Adds Level 0
 * @brief dst[i] = src[i] + sacalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Adds Level 2
 * @brief dst = src[i] + sacalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    const int32_t& count);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    const int32_t& count);

/* **************************************************************************************************
 * Muls                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Muls Level 0
 * @brief dst[i] = src[i] * sacalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Muls Level 2
 * @brief dst = src[i] * sacalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    const int32_t& count);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    const int32_t& count);

/* **************************************************************************************************
 * Maxs                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Maxs Level 0
 * @brief dst[i] = src[i] > sacalar ? src[0] : scalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Maxs Level 2
 * @brief dst = src[i] > sacalar ? src[0] : scalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    const int32_t& count);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    const int32_t& count);

/* **************************************************************************************************
 * Mins                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Mins Level 0
 * @brief dst[i] = src[i] < sacalar ? src[0] : scalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Mins Level 2
 * @brief dst = src[i] < sacalar ? src[0] : scalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    const int32_t& count);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Mins(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    const int32_t& count);

/* **************************************************************************************************
 * ShiftLeft                                             *
 * ************************************************************************************************* */
/*
 * @ingroup ShiftLeft Level 0
 * @brief dst[i] = src[i] << sacalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Shiftleft Level 2
 * @brief dst = src[i] << sacalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    const int32_t& count);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    const int32_t& count);

/* **************************************************************************************************
 * ShiftRight                                             *
 * ************************************************************************************************* */
/*
 * @ingroup ShiftRight Level 0
 * @brief dst[i] = src[i] >> sacalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn);

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn = false);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams, bool roundEn);

/*
 * @ingroup ShiftRight Level 2
 * @brief dst = src[i] >> sacalar
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    const int32_t& count);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    const int32_t& count);

/* **************************************************************************************************
 * LeakyRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup LeakyRelu Level 0
 * @brief dst[i] = src[i] < 0 ? (scalar * src[i]) : src[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup LeakyRelu Level 2
 * @brief dst = src[i] < 0 ? (scalar * src[i]) : src[i]
 * @param [out] dst output LocalTensor
 * @param [in] src input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue,
    const int32_t& count);

template <typename T, typename U, bool isSetMask = true,
    typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const U& scalarValue,
    const int32_t& count);
} // namespace AscendC
#pragma end_pipe

#include "../../impl/basic_api/kernel_operator_vec_binary_scalar_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
