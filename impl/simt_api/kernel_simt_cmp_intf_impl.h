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

#ifndef ASCENDC_MODULE_SIMT_CMP_INTERFACE_IMPL_H
#define ASCENDC_MODULE_SIMT_CMP_INTERFACE_IMPL_H

#include "kernel_simt_common_intf_impl.h"

namespace AscendC {
namespace Simt {
template <typename T>
__aicore__ inline bool IsFinite(T x)
{
    static_assert(SupportType<T, half, float>(), "Input type T only supports half, float.");
    return IsFiniteImpl(x);
}

template <typename T>
__aicore__ inline bool IsNan(T x)
{
    static_assert(SupportType<T, half, float>(), "Input type T only supports half, float.");
    return IsNanImpl(x);
}

template <typename T>
__aicore__ inline bool IsInf(T x)
{
    static_assert(SupportType<T, half, float>(), "Input type T only supports half, float.");
    return IsInfImpl(x);
}

}  // namespace Simt
}  // namespace AscendC
#endif  // ASCENDC_MODULE_SIMT_CMP_INTERFACE_IMPL_H
