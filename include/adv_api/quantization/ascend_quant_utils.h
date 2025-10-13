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
 * \file ascend_quant_utils.h
 * \brief
 */
#ifndef LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H
#define LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H

namespace AscendC {
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
struct AscendQuantConfig {
    bool hasOffset;
    int32_t kDim = 1;
    RoundMode roundMode = RoundMode::CAST_RINT;
};

enum class AscendQuantPolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP,
    PER_CHANNEL_PER_GROUP,
    PER_TOKEN_PER_GROUP
};

struct AscendQuantParam {
  uint32_t m;
  uint32_t n;
  uint32_t calCount;
  uint32_t groupSize = 0;
};
#else
struct AscendQuantConfig {
    __aicore__ constexpr AscendQuantConfig(const uint32_t calcCount, const uint32_t offsetCount,
        const uint32_t scaleCount, const uint32_t workLocalSize): calcCount(calcCount), offsetCount(offsetCount),
        scaleCount(scaleCount), workLocalSize(workLocalSize) {}
    uint32_t calcCount = 0;
    uint32_t offsetCount = 0;
    uint32_t scaleCount = 0;
    uint32_t workLocalSize = 0;
};

constexpr AscendQuantConfig ASCEND_QUANT_DEFAULT_CFG = {0, 0, 0, 0};
#endif
}; // namespace AscendC
#endif // LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H