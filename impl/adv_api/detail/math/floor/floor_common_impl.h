/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file floor_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_FLOOR_FLOOR_COMMON_IMPL_H
#define IMPL_MATH_FLOOR_FLOOR_COMMON_IMPL_H

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
#include "floor_v220_impl.h"
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "floor_v200_impl.h"
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 300)
#include "floor_v300_impl.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__) || defined(__DAV_L311__) || (__NPU_ARCH__ == 5102)
#include "floor_c310_impl.h"
#endif

#endif // IMPL_MATH_FLOOR_FLOOR_COMMON_IMPL_H