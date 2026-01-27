/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H

namespace AscendC {
// bisheng::cce::set_atomic_none
__aicore__ inline void SetAtomicNoneImpl()
{
    bisheng::cce::set_atomic_none();
}
// bisheng::cce::set_atomic_add
template <typename T>
__aicore__ inline void SetAtomicAddImpl()
{
    static_assert(SupportType<T, float, half, int16_t, int32_t, int8_t, bfloat16_t>(),
        "Error type for SetAtomicAdd.");
}

template <> __aicore__ inline void SetAtomicAddImpl<float>()
{
    bisheng::cce::set_atomic_f32();
    bisheng::cce::set_atomic_add();
}

template <>
__aicore__ inline void SetAtomicAddImpl<half>()
{
    bisheng::cce::set_atomic_f16();
    bisheng::cce::set_atomic_add();
}

template <>
__aicore__ inline void SetAtomicAddImpl<int16_t>()
{
    bisheng::cce::set_atomic_s16();
    bisheng::cce::set_atomic_add();
}

template <>
__aicore__ inline void SetAtomicAddImpl<int32_t>()
{
    bisheng::cce::set_atomic_s32();
    bisheng::cce::set_atomic_add();
}

template <>
__aicore__ inline void SetAtomicAddImpl<int8_t>()
{
    bisheng::cce::set_atomic_s8();
    bisheng::cce::set_atomic_add();
}

template <>
__aicore__ inline void SetAtomicAddImpl<bfloat16_t>()
{
    bisheng::cce::set_atomic_bf16();
    bisheng::cce::set_atomic_add();
}
// bisheng::cce::set_atomic_max
template <typename T>
__aicore__ inline void SetAtomicMaxImpl()
{
    static_assert(SupportType<T, float, half, int16_t, int32_t, int8_t, bfloat16_t>(),
        "Error type for SetAtomicMax.");
}

template <>
__aicore__ inline void SetAtomicMaxImpl<float>()
{
    bisheng::cce::set_atomic_max();
    bisheng::cce::set_atomic_f32();
}

template <>
__aicore__ inline void SetAtomicMaxImpl<half>()
{
    bisheng::cce::set_atomic_max();
    bisheng::cce::set_atomic_f16();
}

template <>
__aicore__ inline void SetAtomicMaxImpl<int16_t>()
{
    bisheng::cce::set_atomic_max();
    bisheng::cce::set_atomic_s16();
}

template <>
__aicore__ inline void SetAtomicMaxImpl<int32_t>()
{
    bisheng::cce::set_atomic_max();
    bisheng::cce::set_atomic_s32();
}

template <>
__aicore__ inline void SetAtomicMaxImpl<int8_t>()
{
    bisheng::cce::set_atomic_max();
    bisheng::cce::set_atomic_s8();
}

template <>
__aicore__ inline void SetAtomicMaxImpl<bfloat16_t>()
{
    bisheng::cce::set_atomic_max();
    bisheng::cce::set_atomic_bf16();
}

// bisheng::cce::set_atomic_min
template <typename T>
__aicore__ inline void SetAtomicMinImpl()
{
    static_assert(SupportType<T, float, half, int16_t, int32_t, int8_t, bfloat16_t>(),
        "Error type for SetAtomicMin.");
}

template <>
__aicore__ inline void SetAtomicMinImpl<float>()
{
    bisheng::cce::set_atomic_min();
    bisheng::cce::set_atomic_f32();
}

template <>
__aicore__ inline void SetAtomicMinImpl<half>()
{
    bisheng::cce::set_atomic_min();
    bisheng::cce::set_atomic_f16();
}

template <>
__aicore__ inline void SetAtomicMinImpl<int16_t>()
{
    bisheng::cce::set_atomic_min();
    bisheng::cce::set_atomic_s16();
}

template <>
__aicore__ inline void SetAtomicMinImpl<int32_t>()
{
    bisheng::cce::set_atomic_min();
    bisheng::cce::set_atomic_s32();
}

template <>
__aicore__ inline void SetAtomicMinImpl<int8_t>()
{
    bisheng::cce::set_atomic_min();
    bisheng::cce::set_atomic_s8();
}

template <>
__aicore__ inline void SetAtomicMinImpl<bfloat16_t>()
{
    bisheng::cce::set_atomic_min();
    bisheng::cce::set_atomic_bf16();
}

template <typename T> __aicore__ inline void SetAtomicTypeImpl()
{
    static_assert(SupportType<T, float, half, int16_t, int32_t, int8_t, bfloat16_t>(),
        "Error type for SetAtomicType.");
}

template <> __aicore__ inline void SetAtomicTypeImpl<float>()
{
    bisheng::cce::set_atomic_f32();
}

template <> __aicore__ inline void SetAtomicTypeImpl<half>()
{
    bisheng::cce::set_atomic_f16();
}

template <> __aicore__ inline void SetAtomicTypeImpl<int16_t>()
{
    bisheng::cce::set_atomic_s16();
}

template <> __aicore__ inline void SetAtomicTypeImpl<int32_t>()
{
    bisheng::cce::set_atomic_s32();
}

template <> __aicore__ inline void SetAtomicTypeImpl<int8_t>()
{
    bisheng::cce::set_atomic_s8();
}

template <> __aicore__ inline void SetAtomicTypeImpl<bfloat16_t>()
{
    bisheng::cce::set_atomic_bf16();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H