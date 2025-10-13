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
 * \file logical_ands_common_impl.h
 * \brief
 */

#ifndef LIB_MATH_LOGICAL_ANDS_IMPL_H
#define LIB_MATH_LOGICAL_ANDS_IMPL_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"
#include "../logical_template/logical_template.h"

namespace AscendC {
struct LogicalAndsConfig {
    bool isReuseSource;
    int8_t scalarTensorIndex;
};
constexpr LogicalAndsConfig DEFAULT_LOGICAL_ANDS_CONFIG = { false, 1 };

template <const LogicalAndsConfig& config = DEFAULT_LOGICAL_ANDS_CONFIG, typename T, typename U, typename S>
__aicore__ inline void LogicalAndsImpl(const LocalTensor<T>& dst, const U& src0, const S& src1, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    CHECK_FUNC_HIGHLEVEL_API(LogicalAnds, (T, U, S, config.isReuseSource), (dst, src0, src1, count));
    auto constexpr func = MicroAPI::MaskAnd;
    LogicalTemplateScalarImpl<func, T, U, S, config.scalarTensorIndex>(dst, src0, src1, count);
}

}
#endif
#endif  // IMPL_MATH_LOGICAL_ANDS_LOGICAL_ANDS_COMMON_IMPL_H
