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
 * \file ascend_quant_utils.h
 * \brief
 */
#ifndef LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H
#define LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H

namespace AscendC {
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
}; // namespace AscendC
#endif // LIB_QUANTIZATION_ASCEND_QUANT_UTILS_H