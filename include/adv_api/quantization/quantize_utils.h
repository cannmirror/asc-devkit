/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quantize_utils.h
 * \brief
 */
#ifndef LIB_QUANTIZATION_QUANTIZE_UTILS_H
#define LIB_QUANTIZATION_QUANTIZE_UTILS_H

namespace AscendC {
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
enum class QuantizePolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP
};

struct QuantizeConfig {
    QuantizePolicy policy;
    bool hasOffset;
    RoundMode roundMode = RoundMode::CAST_RINT;
    int32_t kDim = 1;
};

struct QuantizeParams {
  uint32_t m;
  uint32_t n;
  uint32_t groupSize = 0;
};
#endif // defined(__DAV_C310__) || defined(__DAV_310R6__)
}; // namespace AscendC
#endif // LIB_QUANTIZATION_QUANTIZE_UTILS_H