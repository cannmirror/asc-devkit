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
 * \file bitwise_xor.h
 * \brief
 */

#ifndef LIB_MATH_BITWISE_XOR_H
#define LIB_MATH_BITWISE_XOR_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "../../../impl/adv_api/detail/math/bitwise_xor/bitwise_xor_c310_impl.h"
#include "kernel_tensor.h"
namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup BitwiseXor
 * \param [out] dst, output LocalTensor
 * \param [in] src0, input LocalTensor
 * \param [in] src1, input LocalTensor
 * \param [in] count, amount of data to be calculated
 */
template <const BitwiseXorConfig& config = DEFAULT_BITWISE_XOR_CONFIG, typename T>
__aicore__ inline void BitwiseXor(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                                 const uint32_t count)
{
    BitwiseXorImpl<config, T>(dst, src0, src1, count);
}

#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_MATH_BITWISE_XOR_H