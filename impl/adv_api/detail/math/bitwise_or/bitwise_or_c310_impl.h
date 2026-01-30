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
 * \file bitwise_or_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_BITWISE_OR_BITWISE_OR_C310_IMPL_H
#define IMPL_MATH_BITWISE_OR_BITWISE_OR_C310_IMPL_H
#include "../bitwise_template/bitwise_template.h"
#include "kernel_basic_intf.h"
#ifdef ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_check/math/bitwise_or/bitwise_or_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
struct BitwiseOrConfig {
    bool isReuseSource;
};
constexpr BitwiseOrConfig DEFAULT_BITWISE_OR_CONFIG = {false};
template <const BitwiseOrConfig& config, typename T>
__aicore__ inline void BitwiseOrImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                                     const uint32_t count)
{
    if ASCEND_IS_AIC {
        return;
    }

    CHECK_FUNC_HIGHLEVEL_API(BitwiseOr, (T, config.isReuseSource), (dst, src0, src1, count));

    if constexpr (sizeof(T) == 8) {
        BitwiseTemplateImpl<
            MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>, T>(
            dst, src0, src1, count);
    } else {
        BitwiseTemplateImpl<
            MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne>>, T>(
            dst, src0, src1, count);
    }
}
} // namespace AscendC

#endif // IMPL_MATH_BITWISE_OR_BITWISE_OR_C310_IMPL_H