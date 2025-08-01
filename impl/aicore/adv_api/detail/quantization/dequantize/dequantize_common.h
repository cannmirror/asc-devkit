/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dequantize_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_QUANTIZATION_DEQUANTIZE_DEQUANTIZE_COMMON_H
#define AICORE_ADV_API_DETAIL_QUANTIZATION_DEQUANTIZE_DEQUANTIZE_COMMON_H

namespace AscendC {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
enum class DequantizePolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP,
};

struct DequantizeConfig {
    DequantizePolicy policy;
    bool hasOffset = false;
    int32_t kDim = 1;
};

struct DequantizeParams {
    uint32_t m;
    uint32_t n;
    uint32_t groupSize = 0;
};
#endif
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_QUANTIZATION_DEQUANTIZE_DEQUANTIZE_COMMON_H
