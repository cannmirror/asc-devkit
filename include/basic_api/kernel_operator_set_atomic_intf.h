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
 * \file kernel_operator_set_atomic_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SET_ATOMIC_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_SET_ATOMIC_INTERFACE_H
#include "kernel_tensor.h"

namespace AscendC {
template <typename T>
__aicore__ inline void SetAtomicType();

template <typename T>
__aicore__ inline void SetAtomicAdd();

__aicore__ inline void SetAtomicNone();

template <typename T>
__aicore__ inline void SetAtomicMax();

template <typename T>
__aicore__ inline void SetAtomicMin();
} // namespace AscendC

#include "../../impl/basic_api/kernel_operator_set_atomic_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_SET_ATOMIC_INTERFACE_H