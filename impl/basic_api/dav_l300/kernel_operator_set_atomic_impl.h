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
#ifndef ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H

namespace AscendC {
// set_atomic_none
__aicore__ inline void SetAtomicNoneImpl()
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetAtomicNone");
    });
}

// set_atomic_add
template <typename T>
__aicore__ inline void SetAtomicAddImpl()
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported SetAtomicAdd"); });
}

// set_atomic_max
template <typename T>
__aicore__ inline void SetAtomicMaxImpl()
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported SetAtomicMax"); });
}

// set_atomic_min
template <typename T>
__aicore__ inline void SetAtomicMinImpl()
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported SetAtomicMin"); });
}

// set_atomic_type
template <typename T> __aicore__ inline void SetAtomicTypeImpl()
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported SetAtomicType"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H