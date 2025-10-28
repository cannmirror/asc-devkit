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
 * \file sincos.h
 * \brief
 */

#ifndef LIB_MATH_SINCOS_H
#define LIB_MATH_SINCOS_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"
#include "../../../impl/adv_api/detail/math/sincos/sincos_c310_impl.h"

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup SinCos
 * \param [out] dst0, output LocalTensor
 * \param [out] dst1, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] count, amount of data to be calculated
 */
template <const SinCosConfig& config = DEFAULT_SINCOS_CONFIG, typename T>
__aicore__ inline void SinCos(const LocalTensor<T>& dst0, const LocalTensor<T>& dst1, 
    const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
{
    SinCosRadianReductionImpl<config, T>(dst0, dst1, src, sharedTmpBuffer, count);
}

/* !
 * \ingroup SinCos
 * \note support data type: half and float
 * \param [out] dst0, output LocalTensor
 * \param [out] dst1, output LocalTensor
 * \param [in] src, input LocalTensor
 * \param [in] count, amount of data to be calculated
 */
template <const SinCosConfig& config = DEFAULT_SINCOS_CONFIG, typename T>
__aicore__ inline void SinCos(const LocalTensor<T>& dst0, const LocalTensor<T>& dst1, 
    const LocalTensor<T>& src, const uint32_t count)
{
    SinCosRadianReductionImpl<config, T>(dst0, dst1, src, count);
}

#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_MATH_SINCOS_H
