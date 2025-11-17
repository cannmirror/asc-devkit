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
 * \file kernel_operator_vec_transpose_impl.h
 * \brief AscendC l510 support vector transpose api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H

namespace AscendC {
template <typename T>
__aicore__ inline void TransDataTo5HDImpl(__ubuf__ T* dstList[16], __ubuf__ T* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    ASSERT(false && "TransDataTo5HD is not supported on current device!");
}

template <typename T>
__aicore__ inline void TransDataTo5HDImpl(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE],
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& transDataTo5HDParams)
{
    ASSERT(false && "TransDataTo5HD is not supported on current device!");
}

template <typename T>
__aicore__ inline void TransDataTo5HDVldVaRegImpl(
    __ubuf__ uint64_t* dst, __ubuf__ uint64_t* src, const TransDataTo5HDParams& transDataTo5HDParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported TransDataTo5HD on current device"); });
}

template <typename T>
__aicore__ inline void Transpose4DImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const TransposeParamsExt &transposeParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported transpose between NCHW and NHWC on current device!"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H