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
 * \file bitwise_and_common_impl.h
 * \brief
 */

#ifndef LIB_MATH_BITWISE_AND_IMPL_H
#define LIB_MATH_BITWISE_AND_IMPL_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 5102)
#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "../bitwise_template/bitwise_template.h"
#ifdef ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_check/math/bitwise_and/bitwise_and_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_api_check.h"
namespace AscendC {
struct BitwiseAndConfig {
    bool isReuseSource;
};
constexpr BitwiseAndConfig DEFAULT_BITWISE_AND_CONFIG = {false};

template <const BitwiseAndConfig& config = DEFAULT_BITWISE_AND_CONFIG, typename T>
__aicore__ inline void BitwiseAndImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                                      const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    CHECK_FUNC_HIGHLEVEL_API(BitwiseAnd, (T, config.isReuseSource), (dst, src0, src1, count));

    if constexpr (sizeof(T) == 8) {
        BitwiseTemplateImpl<
            MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>, T>(
            dst, src0, src1, count);
    } else {
        BitwiseTemplateImpl<
            MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne>>, T>(
            dst, src0, src1, count);
    }
}

} // namespace AscendC
#endif
#endif  // IMPL_MATH_BITWISE_AND_BITWISE_AND_COMMON_IMPL_H
