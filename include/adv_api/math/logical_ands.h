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
 * \file logical_ands.h
 * \brief
 */

#ifndef LIB_MATH_LOGICAL_ANDS_H
#define LIB_MATH_LOGICAL_ANDS_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"
#include "../../../impl/adv_api/detail/math/logical_ands/logical_ands_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup LogicalAnds
 * \param [out] dst, output LocalTensor
 * \param [in] src0, input LocalTensor
 * \param [in] src1, input LocalTensor
 * \param [in] count, amount of data to be calculated
 */
template <const LogicalAndsConfig& config = DEFAULT_LOGICAL_ANDS_CONFIG, typename T, typename U, typename S>
__aicore__ inline void LogicalAnds(const LocalTensor<T>& dst, const U& src0, const S& src1, const uint32_t count)
{
    LogicalAndsImpl<config, T, U, S>(dst, src0, src1, count);
}

#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_MATH_LOGICAL_ANDS_H
