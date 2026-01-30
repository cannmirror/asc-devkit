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
 * \file kernel_operator_fixpipe_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_FIXPIPE_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_FIXPIPE_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_fixpipe.h"
#include "tile_api/kernel_tensor_tile_intf_utils.h"

namespace AscendC {
/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Fixpipe
 * @brief After calculation, process the results
 * @param [out] dst output GlobalTensor
 * @param [in] src input LocalTensor
 * @param [in] intriParams.cburstNum number of burst
 * @param [in] intriParams.burstLen burst length
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstStride dst block stride
 * @param [in] intriParams.biasParams contains isBias flag and bias LocalTensor
 * @param [in] intriParams.quantParams contains quant mode and quant params
 * @param [in] intriParams.reluEn indicates whether to enable the relu function
 * @param [in] intriParams.nz2ndParams contains the input params for enable the nz2nd function
 */

template <typename T>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
    bool isUnitFlag = false);

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T> &preData, bool isUnitFlag = false);

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
__aicore__ inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint32_t dstNdStride);
#else
__aicore__ inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride);
#endif

__aicore__ inline void SetFixpipePreQuantFlag(uint64_t config);

__aicore__ inline void SetFixPipeClipRelu(uint64_t config);

template <typename T>
__aicore__ inline void SetFixPipeAddr(const LocalTensor<T> &eleWiseData, uint16_t c0ChStride);

#if defined(__NPU_ARCH__) &&                                                                            \
    (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3003 ||    \
     __NPU_ARCH__ == 3113 || __NPU_ARCH__ == 3101 || __NPU_ARCH__ == 5102)
// L0C->L1
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParamsV220& intriParams);

// L0C->L1 deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t,
    typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<S>& cbufWorkspace, const FixpipeParamsV220& intriParams);

// L0C->GM
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParamsV220& intriParams);

// L0C->GM deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t,
    typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<S>& cbufWorkspace, const FixpipeParamsV220& intriParams);
#endif

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3002
// L0C->L1/UB
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParamsM300& intriParams);

// L0C->L1/UB deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t,
    typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<S>& cbufWorkspace, const FixpipeParamsM300& intriParams);

// L0C->GM
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParamsM300& intriParams);

// L0C->GM deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t,
    typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<S>& cbufWorkspace, const FixpipeParamsM300& intriParams);
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 3102
// L0C->L1/UB
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParamsM310& intriParams);

// L0C->L1/UB deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t,
    typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<S>& cbufWorkspace, const FixpipeParamsM310& intriParams);

// L0C->GM
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParamsM310& intriParams);

// L0C->GM deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t,
    typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<S>& cbufWorkspace, const FixpipeParamsM310& intriParams);
#elif (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
// L0C->L1
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParamsC310<config.format>& intriParams);

// L0C->L1/UB deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParamsC310<config.format>& intriParams);

// L0C->GM
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParamsC310<config.format>& intriParams);

// L0C->GM deq tensor quant
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParamsC310<config.format>& intriParams);
#endif
} // namespace AscendC

namespace AscendC {

template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeTemplate<T, U>, void>::type 
Fixpipe(const T& dst, const U& src);

template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename V>
__aicore__ inline typename Std::enable_if<VerifyingFixpipeQuantTemplate<T, U, V>, void>::type 
Fixpipe(const T& dst, const U& src, const V& quant);

} // namespace AscendC

#include "../../impl/basic_api/kernel_operator_fixpipe_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_INTERFACE_H
