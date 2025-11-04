/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arithprogression.h
 * \brief
 */
#ifndef LIB_ARITHPROGRESSION_H
#define LIB_ARITHPROGRESSION_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || defined(__DAV_L311__) || (__NPU_ARCH__ == 5102) || defined(__DAV_L300__)
#include "../../../impl/adv_api/detail/index/arithprogression/arithprogression_c310_impl.h"
#else
#include "../../../impl/adv_api/detail/index/arithprogression/arithprogression_common_impl.h"
#endif

namespace AscendC {
/* !
 * \brief This function realizes the arithmetic sequence function. The formula is as follows:
 * dst[i+1] = dst[i] + diffValue
 *
 * \note support data type: half, float, int16_t, int32_t
 * \note Ascend910 support data type: half, float, int16_t, int32_t, int64_t
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] firstValue, first value of arithmetic sequence
 * \param [in] diffValue, diff value of arithmetic sequence
 * \param [in] count, length of the sequence
 */
template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V, S) void Arange(const LocalTensor<T> &dstLocal,
    const T firstValue, const T diffValue, const int32_t count)
{
    ArithProgressionImpl(dstLocal, firstValue, diffValue, count);
}
} // namespace AscendC

#endif // LIB_ARITHPROGRESSION_H
