/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file philox.h
 */
#ifndef AICORE_ADV_API_MATH_PHILOX_H
#define AICORE_ADV_API_MATH_PHILOX_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "kernel_tensor.h"
#include "detail/math/philox/philox_c310_impl.h"

namespace AscendC {

#pragma begin_pipe(V)

template <uint16_t Rounds = 7, typename T>
__aicore__ inline void PhiloxRandom(
    const LocalTensor<T>& dstLocal, const PhiloxKey& philoxKey, const PhiloxCounter& philoxCounter, uint16_t count)
{
    PhiloxRandomImpl<Rounds>(dstLocal, philoxKey, philoxCounter, count);
}

template <uint16_t Rounds = 7, typename T>
__aicore__ inline void PhiloxRandom(const LocalTensor<T>& dstLocal, const PhiloxKey& philoxKey,
    const PhiloxCounter& philoxCounter, const PhiloxRandomParams& params)
{
    PhiloxRandomImpl<Rounds>(dstLocal, philoxKey, philoxCounter, params);
}

#pragma end_pipe
} // namespace AscendC

#endif
#endif // AICORE_ADV_API_MATH_PHILOX_H
