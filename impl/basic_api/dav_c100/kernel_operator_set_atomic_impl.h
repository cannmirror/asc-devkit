/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_set_atomic_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SET_ATOMIC_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SET_ATOMIC_IMPL_H
namespace AscendC {
__aicore__ inline void SetAtomicNoneImpl()
{
    set_atomic_none();
}
template <typename T> __aicore__ inline void SetAtomicAddImpl() {}
template <> __aicore__ inline void SetAtomicAddImpl<float>()
{
    set_atomic_f32();
}
template <> __aicore__ inline void SetAtomicAddImpl<half>()
{
    ASSERT(false && "SetAtomicAdd<half> is not supported on current device");
}
template <> __aicore__ inline void SetAtomicAddImpl<int16_t>()
{
    ASSERT(false && "SetAtomicAdd<int16_t> is not supported on current device");
}
template <> __aicore__ inline void SetAtomicAddImpl<int32_t>()
{
    ASSERT(false && "SetAtomicAdd<int32_t> is not supported on current device");
}
template <> __aicore__ inline void SetAtomicAddImpl<int8_t>()
{
    ASSERT(false && "SetAtomicAdd<int8_t> is not supported on current device");
}

template <typename T> __aicore__ inline void SetAtomicTypeImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetAtomicType");
}

template <typename T>
__aicore__ inline void SetAtomicMaxImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetAtomicMax");
}

template <typename T>
__aicore__ inline void SetAtomicMinImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetAtomicMin");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SET_ATOMIC_IMPL_H