/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_common.h
 * \brief
 */

#ifndef AICORE_ADV_API_DETAIL_CONV_COMMON_CONV_COMMON_H
#define AICORE_ADV_API_DETAIL_CONV_COMMON_CONV_COMMON_H

#include "matmul/matmul_tiling_base.h"

namespace ConvCommonApi {
using matmul_tiling::TPosition;

enum class ConvDtype : uint32_t {
    FLOAT16 = 0,
    FLOAT32,
    BF16,
    INT4,
    INT8,
    UINT8,
    INT32,
    INT64,
    UINT64,
    CONVDTYPEMAX
};

} // namespace ConvCommonApi
#endif
