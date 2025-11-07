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

#ifndef ASCENDC_MODULE_SIMT_CAST_INTERFACE_H
#define ASCENDC_MODULE_SIMT_CAST_INTERFACE_H

namespace AscendC {
namespace Simt {

template <typename T, typename U, RoundMode roundMode>
__aicore__ inline T Cast(U x);

template <typename T>
__aicore__ inline T Round(T x);

template <typename T>
__aicore__ inline T Rint(T x);

template <typename T>
__aicore__ inline T Floor(T x);

template <typename T>
__aicore__ inline T Ceil(T x);

template <typename T>
__aicore__ inline T Trunc(T x);

}  // namespace Simt
}  // namespace AscendC

#include "impl/simt_api/kernel_simt_cast_intf_impl.h"
#endif  // ASCENDC_MODULE_SIMT_CAST_INTERFACE_H
