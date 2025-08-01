/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "index/arithprogression_tiling.h"

namespace AscendC {
// ArithProgression does not require temporary space. Therefore, the obtained maximum and minimum temporary space sizes
// are both 0.
void GetArithProgressionMaxMinTmpSize(uint32_t& maxValue, uint32_t& minValue)
{
    maxValue = 0;
    minValue = 0;
}
} // namespace AscendC
