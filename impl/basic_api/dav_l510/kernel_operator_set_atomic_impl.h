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
 * \file kernel_operator_set_atomic_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H

namespace AscendC {
// set_atomic_none
__aicore__ inline void SetAtomicNoneImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicNone is not supported."); });
}
// set_atomic_add
template <typename T>
__aicore__ inline void SetAtomicAddImpl() {}

template <> __aicore__ inline void SetAtomicAddImpl<float>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicAdd is not supported."); });
}

template <>
__aicore__ inline void SetAtomicAddImpl<half>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicAdd is not supported."); });
}

template <>
__aicore__ inline void SetAtomicAddImpl<int16_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicAdd is not supported."); });
}

template <>
__aicore__ inline void SetAtomicAddImpl<int32_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicAdd is not supported."); });
}

template <>
__aicore__ inline void SetAtomicAddImpl<int8_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicAdd is not supported."); });
}

template <>
__aicore__ inline void SetAtomicAddImpl<bfloat16_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicAdd is not supported."); });
}
// set_atomic_max
template <typename T>
__aicore__ inline void SetAtomicMaxImpl() {}

template <>
__aicore__ inline void SetAtomicMaxImpl<float>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMax is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMaxImpl<half>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMax is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMaxImpl<int16_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMax is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMaxImpl<int32_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMax is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMaxImpl<int8_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMax is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMaxImpl<bfloat16_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMax is not supported."); });
}

// set_atomic_min
template <typename T>
__aicore__ inline void SetAtomicMinImpl() {}

template <>
__aicore__ inline void SetAtomicMinImpl<float>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMin is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMinImpl<half>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMin is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMinImpl<int16_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMin is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMinImpl<int32_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMin is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMinImpl<int8_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMin is not supported."); });
}

template <>
__aicore__ inline void SetAtomicMinImpl<bfloat16_t>()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicMin is not supported."); });
}

// set_atomic_type
template <typename T> __aicore__ inline void SetAtomicTypeImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetAtomicTypeImpl is not supported."); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H