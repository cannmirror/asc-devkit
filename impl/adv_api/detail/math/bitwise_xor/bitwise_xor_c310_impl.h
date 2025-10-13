/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file bitwise_xor_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_BITWISE_XOR_BITWISE_XOR_C310_IMPL_H
#define IMPL_MATH_BITWISE_XOR_BITWISE_XOR_C310_IMPL_H
#include "../bitwise_template/bitwise_template.h"
namespace AscendC {
struct BitwiseXorConfig {
    bool isReuseSource;
};
constexpr BitwiseXorConfig DEFAULT_BITWISE_XOR_CONFIG = {false};
template <const BitwiseXorConfig& config, typename T>
__aicore__ inline void BitwiseXorImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1,
                                      const uint32_t count)
{
    if ASCEND_IS_AIC {
        return;
    }

    CHECK_FUNC_HIGHLEVEL_API(BitwiseXor, (T, config.isReuseSource), (dst, src0, src1, count));

    if constexpr (sizeof(T) == 8) {
        BitwiseTemplateImpl<
            MicroAPI::Xor<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>, T>(
            dst, src0, src1, count);
    } else {
        BitwiseTemplateImpl<
            MicroAPI::Xor<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne>>, T>(
            dst, src0, src1, count);
    }
}
} // namespace AscendC

#endif // IMPL_MATH_BITWISE_XOR_BITWISE_XOR_C310_IMPL_H