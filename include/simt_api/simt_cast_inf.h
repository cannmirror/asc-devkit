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
 * \file simt_cast_inf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_SIMT_C_CAST_INTERFACE_H
#define ASCENDC_MODULE_SIMT_C_CAST_INTERFACE_H

__simt_callee__ inline long int lroundf(float x);

__simt_callee__ inline long long int llroundf(float x);

__simt_callee__ inline long int lrintf(float x);

__simt_callee__ inline long long int llrintf(float x);

__simt_callee__ inline float truncf(float x);

#include "impl/simt_api/simt_cast_intf_impl.h"
#endif  // ASCENDC_MODULE_SIMT_C_CAST_INTERFACE_H