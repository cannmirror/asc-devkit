/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_vec_duplicate_impl.h
 * \brief AscendC l210 support vector duplicate memory base api.
 */

#ifndef ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckDuplicateSupportedType()
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, float>::value || std::is_same<T, uint8_t>::value ||
        std::is_same<T, int8_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
        std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
        "Duplicate instr only support half/float/uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t in this device");
}

template <typename T>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dst, const T scalarValue, const int32_t& count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

#define DUPLICATE_OP_IMPL(DATA_TYPE, REG_TYPE, BIT_WIDTH)                                        \
template <typename T = DATA_TYPE>                                                                \
__aicore__ inline void DuplicateImpl(__ubuf__ DATA_TYPE* dst, const DATA_TYPE scalarValue,  \
                                     const int32_t& count)                                    \
{                                                                                                \
    __VEC_SCOPE__ {                                                                              \
        for (uint16_t i = 0; i <= get_vloopn_bound_b##BIT_WIDTH(count); ++i) {                \
            vector_##REG_TYPE vreg;                                                              \
            vector_bool preg;                                                                    \
            vector_address offset;                                                               \
            preg = vpd_b##BIT_WIDTH();                                                           \
            offset = vag_b##BIT_WIDTH(ELE_CNT_B##BIT_WIDTH);                                     \
            vdup(vreg, scalarValue, preg, MODE_ZEROING);                                         \
            vst(vreg, dst, offset, NORM_B##BIT_WIDTH, preg);                                \
        }                                                                                        \
    }                                                                                            \
}

DUPLICATE_OP_IMPL(uint8_t, u8, 8)
DUPLICATE_OP_IMPL(int8_t, s8, 8)
DUPLICATE_OP_IMPL(uint16_t, u16, 16)
DUPLICATE_OP_IMPL(int16_t, s16, 16)
DUPLICATE_OP_IMPL(uint32_t, u32, 32)
DUPLICATE_OP_IMPL(int32_t, s32, 32)
DUPLICATE_OP_IMPL(half, f16, 16)
DUPLICATE_OP_IMPL(float, f32, 32)

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
