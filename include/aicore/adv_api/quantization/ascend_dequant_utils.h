/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ascend_dequant_utils.h
 * \brief
 */
#ifndef AICORE_ADV_API_QUANTIZATION_ASCEND_DEQUANT_UTILS_H
#define AICORE_ADV_API_QUANTIZATION_ASCEND_DEQUANT_UTILS_H

namespace AscendC {
struct DequantParams {
    uint32_t m;        // outer axis length (do not need to be 32B aligned)              in unit of element num
    uint32_t n;        // inner axis length (must be 32B aligned)                        in unit of element num
    uint32_t calCount; // in one inner line, calCount elements do dequant calculation    in unit of element num
};

};     // namespace AscendC
#endif // AICORE_ADV_API_QUANTIZATION_ASCEND_DEQUANT_UTILS_H