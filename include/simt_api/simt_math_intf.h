/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file simt_math_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_SIMT_C_MATH_INTERFACE_H
#define ASCENDC_MODULE_SIMT_C_MATH_INTERFACE_H

__simt_callee__ inline float fmaf(float x, float y, float z);

__simt_callee__ inline float fdimf(float x);

__simt_callee__ inline float remquof(float x, float y, int32_t *quo);

__simt_callee__ inline float fmodf(float x, float y);

__simt_callee__ inline float remainderf(float x, float y);

__simt_callee__ inline float copysignf(float x, float y);

__simt_callee__ inline float nearbyintf(float x);

__simt_callee__ inline float nextafterf(float x, float y);

__simt_callee__ inline float scalbnf(float x, int32_t n);

__simt_callee__ inline float scalblnf(float x, int64_t n);

#include "impl/simt_api/simt_math_intf_impl.h"
#endif  // ASCENDC_MODULE_SIMT_C_MATH_INTERFACE_H
