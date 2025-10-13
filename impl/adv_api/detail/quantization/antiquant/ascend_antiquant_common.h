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
 * \file ascend_antiquant_common.h
 * \brief
 */
#ifndef IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_COMMON_H
#define IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_COMMON_H

#include "include/adv_api/quantization/ascend_antiquant_utils.h"

namespace AscendC {
constexpr uint32_t ANTIQUANT_TWO = 2;
constexpr uint32_t ANTIQUANT_FOUR = 4;
constexpr uint32_t ANTIQUANT_BRCB_BASE = 8;    // BRCB need at least 8 number
constexpr uint32_t ANTIQUANT_MIN_METHOD2 = 80; // min tmpBuffersize must be 80 * N
constexpr uint32_t ANTIQUANT_SINGLE_N_SIZE = 64;
constexpr uint32_t ANTIQUANT_SINGLE_N_SIZE_BF16 = 64;
constexpr uint32_t ANTIQUANT_SINGLE_N_SIZE_FP16 = 128;
constexpr uint32_t ANTIQUANT_MAX_K = 255;
constexpr uint32_t MAX_K_FOR_FP16_BRCB = 4096;
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
constexpr uint32_t ANTIQUANT_FP4_PERGROUP_SIZE = 32;
constexpr uint32_t ANTIQUANT_BF16_MAN_LEN = 7;
#endif

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
struct AscendAntiQuantConfig {
    bool hasOffset;
    bool isTranspose;
    int32_t kDim = 1;
};

enum class AscendAntiQuantPolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP,
    PER_CHANNEL_PER_GROUP,
    PER_TOKEN_PER_GROUP
};

struct AscendAntiQuantParam {
    uint32_t m;
    uint32_t n;
    uint32_t calCount;
    uint32_t groupSize = 0;
};
#endif
} // namespace AscendC
#endif // IMPL_QUANTIZATION_ANTIQUANT_ASCEND_ANTIQUANT_COMMON_H
