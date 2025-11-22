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
 * \file kernel_operator_set_atomic_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SET_ATOMIC_INTERFACE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SET_ATOMIC_INTERFACE_IMPL_H
#include "kernel_tensor.h"

#if __NPU_ARCH__ == 1001
#include "dav_c100/kernel_operator_set_atomic_impl.h"
#elif __NPU_ARCH__ == 2002
#include "dav_m200/kernel_operator_set_atomic_impl.h"
#elif __NPU_ARCH__ == 2201
#include "dav_c220/kernel_operator_set_atomic_impl.h"
#elif __NPU_ARCH__ == 3002
#include "dav_m300/kernel_operator_set_atomic_impl.h"
#elif __NPU_ARCH__ == 3102
#include "dav_m310/kernel_operator_set_atomic_impl.h"
#elif __NPU_ARCH__ == 3101
#include "dav_c310/kernel_operator_set_atomic_impl.h"
#elif (__NPU_ARCH__ == 5102)
#include "dav_m510/kernel_operator_set_atomic_impl.h"
#endif

namespace AscendC {
template <typename T>
__aicore__ inline void SetAtomicType()
{
    SetAtomicTypeImpl<T>();
}

template <typename T>
__aicore__ inline void SetAtomicAdd()
{
    SetAtomicAddImpl<T>();
}

__aicore__ inline void SetAtomicNone()
{
    SetAtomicNoneImpl();
}

template <typename T>
__aicore__ inline void SetAtomicMax()
{
    SetAtomicMaxImpl<T>();
}

template <typename T>
__aicore__ inline void SetAtomicMin()
{
    SetAtomicMinImpl<T>();
}
}
#endif // ASCENDC_MODULE_OPERATOR_SET_ATOMIC_INTERFACE_IMPL_H
