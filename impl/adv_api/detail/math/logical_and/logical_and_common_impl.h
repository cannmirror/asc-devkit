/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
 /* !
 * \file logical_and_common_impl.h
 * \brief
 */

#ifndef LIB_MATH_LOGICAL_AND_IMPL_H
#define LIB_MATH_LOGICAL_AND_IMPL_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 5102)
#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "../logical_template/logical_template.h"
#ifdef ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_check/math/logical_and/logical_and_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
struct LogicalAndConfig {
    bool isReuseSource;
};
constexpr LogicalAndConfig DEFAULT_LOGICAL_AND_CONFIG = { false };

template <const LogicalAndConfig& config = DEFAULT_LOGICAL_AND_CONFIG, typename T, typename U>
__aicore__ inline void LogicalAndImpl(const LocalTensor<T>& dst, const LocalTensor<U>& src0,
    const LocalTensor<U>& src1, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    CHECK_FUNC_HIGHLEVEL_API(LogicalAnd, (T, U, config.isReuseSource), (dst, src0, src1, count));
    auto constexpr func = MicroAPI::MaskAnd;
    LogicalTemplateImpl<func, T, U>(dst, src0, src1, count);
}

}
#endif
#endif  // IMPL_MATH_LOGICAL_AND_LOGICAL_AND_COMMON_IMPL_H
