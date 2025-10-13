/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bitwise_and.h
 * \brief
 */

#ifndef LIB_MATH_BITWISE_NOT_H
#define LIB_MATH_BITWISE_NOT_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "../../../impl/adv_api/detail/math/bitwise_not/bitwise_not_c310_impl.h"
#include "kernel_tensor.h"
namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup BitwiseNot
 * \param [out] dst, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] count, amount of data to be calculated
 */
template <const BitwiseNotConfig& config = DEFAULT_BITWISE_NOT_CONFIG, typename T>
__aicore__ inline void BitwiseNot(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
{
    BitwiseNotImpl<config, T>(dst, src, count);
}

#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_MATH_BITWISE_NOT_H