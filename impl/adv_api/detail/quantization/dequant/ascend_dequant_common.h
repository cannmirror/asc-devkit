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
 * \file ascend_dequant_common.h
 * \brief
 */
#ifndef LIB_ASCEND_DEQUANT_ASCEND_DEQUANT_COMMON_H
#define LIB_ASCEND_DEQUANT_ASCEND_DEQUANT_COMMON_H

#include "include/adv_api/quantization/ascend_dequant_utils.h"

namespace AscendC {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 5102 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
struct AscendDeQuantConfig {
    bool hasOffset;
    int32_t kDim = 1;
};

enum class AscendDeQuantPolicy : int32_t {
    PER_TOKEN,
    PER_GROUP,
    PER_CHANNEL_PER_GROUP,
    PER_TOKEN_PER_GROUP
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
