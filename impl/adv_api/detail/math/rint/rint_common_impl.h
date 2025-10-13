/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
 /* !
 * \file rint_common_impl.h
 * \brief
 */

#ifndef LIB_MATH_RINT_IMPL_H
#define LIB_MATH_RINT_IMPL_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"

namespace AscendC {
struct RintConfig {
    bool isReuseSource;
};
constexpr RintConfig DEFAULT_RINT_CONFIG = { false };

template <const RintConfig& config, typename T>
__aicore__ inline void RintImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer, 
    const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "Rint only support half/float data type on current device!");
    CHECK_FUNC_HIGHLEVEL_API(Rint, (T, config.isReuseSource), (dst, src, sharedTmpBuffer, count));
    Truncate<T, RoundMode::CAST_RINT>(dst, src, count);    
}

template <const RintConfig& config, typename T>
__aicore__ inline void RintImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    RintImpl<config, T>(dst, src, sharedTmpBuffer, count);
}

}
#endif
#endif  // IMPL_MATH_RINT_RINT_COMMON_IMPL_H
