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
 * \file ascend_dequant_common.h
 * \brief
 */
#ifndef LIB_ASCEND_DEQUANT_ASCEND_DEQUANT_COMMON_H
#define LIB_ASCEND_DEQUANT_ASCEND_DEQUANT_COMMON_H

#include "include/adv_api/quantization/ascend_dequant_utils.h"

namespace AscendC {
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
struct AscendDeQuantConfig {
    bool hasOffset;
    int32_t kDim = 1;
};

enum class AscendDeQuantPolicy : int32_t {
    PER_TOKEN,
    PER_GROUP,
    PER_CHANNEL_PER_GROUP,
    PER_TOEKN_PER_GROUP
};

struct AscendDeQuantParam {
    uint32_t m;
    uint32_t n;
    uint32_t calCount;
    uint32_t groupSize = 0;
};
#endif

} // namespace AscendC
#endif // LIB_ASCEND_ANTIQUANT_IMPL_ASCEND_ANTIQUANT_COMMON_H
