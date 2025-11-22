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
 * \file kernel_operator_vec_binary_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_binary.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Add                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Add Level 0
 * @brief dst = src0 + src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Add Level 2
 * @brief dst = src0 + src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * Sub                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Sub Level 0
 * @brief dst = src0 - src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Sub(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Sub(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Sub Level 2
 * @brief dst = src0 - src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Sub(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * Mul                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Mul Level 0
 * @brief dst = src0 * src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Mul(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Mul(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Mul Level 2
 * @brief dst = src0 * src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Mul(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * Div                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Div Level 0
 * @brief dst = src0 / src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Div(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Div(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Div Level 2
 * @brief dst = src0 / src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Div(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, const int32_t& count);


/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
/*
 * @ingroup MulAddDst Level 0
 * @brief dst = src0 * src1 + dst
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDst(const LocalTensor<T>& dst, const LocalTensor<U>& src0,
                                 const LocalTensor<U>& src1, const uint64_t mask[], const uint8_t repeatTime,
                                 const BinaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDst(const LocalTensor<T>& dst, const LocalTensor<U>& src0,
                                 const LocalTensor<U>& src1, uint64_t mask, const uint8_t repeatTime,
                                 const BinaryRepeatParams& repeatParams);

/*
 * @ingroup MulAddDst Level 2
 * @brief dst = src0 * src1 + dst
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void MulAddDst(const LocalTensor<T>& dst, const LocalTensor<U>& src0,
                                 const LocalTensor<U>& src1, const int32_t& count);

/* **************************************************************************************************
 * Max                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Max Level 0
 * @brief dst = src0 > src1 ? src0 : src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Max Level 2
 * @brief dst = src0 > src1 ? src0 : src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * Min                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Min Level 0
 * @brief dst = src0 > src1 ? src1 : src0
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Min(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Min(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Min Level 2
 * @brief dst = src0 > src1 ? src1 : src0
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Min(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * And                                             *
 * ************************************************************************************************* */
/*
 * @ingroup And Level 0
 * @brief dst = src0 & src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void And(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void And(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup And Level 2
 * @brief dst = src0 & src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void And(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                           const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * Or                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Or Level 0
 * @brief dst = src0 | src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Or(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                          const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                          const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Or(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                          const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                          const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Or Level 2
 * @brief dst = src0 | src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Or(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                          const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * AddRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup AddRelu Level 0
 * @brief dst = Relu(src0 + src1)
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void AddRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                               const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                               const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void AddRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                               const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                               const BinaryRepeatParams& repeatParams);

/*
 * @ingroup AddRelu Level 2
 * @brief dst = Relu(src0 + src1)
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void AddRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                               const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup AddDeqRelu Level 0
 * @brief dst = DeqRelu(src0 + src1)
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <bool isSetMask = true>
__aicore__ inline void AddDeqRelu(const LocalTensor<half>& dst, const LocalTensor<int32_t>& src0,
                                  const LocalTensor<int32_t>& src1, uint64_t mask[], const uint8_t repeatTime,
                                  const BinaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AddDeqRelu(const LocalTensor<T>& dst, const LocalTensor<U>& src0,
                                  const LocalTensor<U>& src1, uint64_t mask[], const uint8_t repeatTime,
                                  const BinaryRepeatParams& repeatParams);

template <bool isSetMask = true>
__aicore__ inline void AddDeqRelu(const LocalTensor<half>& dst, const LocalTensor<int32_t>& src0,
                                  const LocalTensor<int32_t>& src1, uint64_t mask, const uint8_t repeatTime,
                                  const BinaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AddDeqRelu(const LocalTensor<T>& dst, const LocalTensor<U>& src0,
                                  const LocalTensor<U>& src1, uint64_t mask, const uint8_t repeatTime,
                                  const BinaryRepeatParams& repeatParams);
/*
 * @ingroup AddDeqRelu Level 2
 * @brief dst = DeqRelu(src0 + src1)
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
__aicore__ inline void AddDeqRelu(const LocalTensor<half>& dst, const LocalTensor<int32_t>& src0,
                                  const LocalTensor<int32_t>& src1, const int32_t& count);

template <typename T, typename U>
__aicore__ inline void AddDeqRelu(const LocalTensor<T>& dst, const LocalTensor<U>& src0,
                                  const LocalTensor<U>& src1, const int32_t& count);

/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
/*
 * @ingroup FusedMulAdd Level 0
 * @brief dst = src0 * dst + src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAdd(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                   const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                                   const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAdd(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                   const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                                   const BinaryRepeatParams& repeatParams);

/*
 * @ingroup FusedMulAdd Level 2
 * @brief dst = src0 * dst + src1
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void FusedMulAdd(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                   const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup FusedMulAddRelu Level 0
 * @brief dst = relu(src0 * dst + src1)
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                       const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                                       const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                       const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                                       const BinaryRepeatParams& repeatParams);

/*
 * @ingroup FusedMulAddRelu Level 2
 * @brief dst = relu(src0 * dst + src1)
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] count number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                                       const LocalTensor<T>& src1, const int32_t& count);

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SubRelu Level 0
 * @brief dst = Relu(src0 - src1)
 * @param [out] dst output LocalTensor
 * @param [in] src0 input LocalTensor
 * @param [in] src1 input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTime repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void SubRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                               const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
                               const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void SubRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                               const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
                               const BinaryRepeatParams& repeatParams);

template <typename T>
__aicore__ inline void SubRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
                               const LocalTensor<T>& src1, const int32_t& count);
}  // namespace AscendC
#pragma end_pipe
#include "../../impl/basic_api/kernel_operator_vec_binary_intf_impl.h"
#endif  // ASCENDC_MODULE_OPERATOR_VEC_BINARY_INTERFACE_H