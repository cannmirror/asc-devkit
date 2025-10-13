/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_processor_utils.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_UTILS_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_UTILS_H

namespace AscendC {
namespace Impl {
namespace Detail {
    
template <typename CType, typename AType>
__aicore__ inline constexpr static bool IsQuantSenario()
{
    using L0cT = typename GetMmDstType<AType>::Type;
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    if constexpr (IsTypeOneOfV<CType, half, bfloat16_t> &&
        !IsTypeOneOfV<AType, int8_t, hifloat8_t, fp8_e4m3fn_t, fp8_e5m2_t>) {
        return false;
    }
    if constexpr (IsTypeOneOfV<AType, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
        IsTypeOneOfV<CType, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t, half, bfloat16_t, float>) {
        return true;
    }
    if constexpr (IsSameTypeV<L0cT, int32_t> && IsSameTypeV<CType, bfloat16_t>) {
        return true;
    }
#endif

    if constexpr (IsSameTypeV<L0cT, int32_t> && IsTypeOneOfV<CType, half, int8_t, uint8_t>) {
        return true;
    } else if constexpr (IsSameTypeV<L0cT, float> && IsTypeOneOfV<CType, int8_t, uint8_t>) {
        return true;
    }
    return false;
}
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_QUANT_QUANT_PROCESSOR_UTILS_H