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
 * \file kernel_operator_mm_bitmode_struct.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_MM_BIT_MODE_STRUCT_H
#define ASCENDC_MODULE_OPERATOR_MM_BIT_MODE_STRUCT_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
#include "dav_c310/kernel_operator_mm_bitmode_impl.h"
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
#include "dav_m510/kernel_operator_mm_bitmode_impl.h"
#endif
#endif  // ASCENDC_MODULE_OPERATOR_MM_BIT_MODE_STRUCT_H