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
 * \file ascend_antiquant_utils.h
 * \brief
 */
#ifndef AICORE_ADV_API_QUANTIZATION_ASCEND_ANTIQUANT_UTILS_H
#define AICORE_ADV_API_QUANTIZATION_ASCEND_ANTIQUANT_UTILS_H

namespace AscendC {
struct AntiQuantShapeInfo {
    uint32_t offsetHeight{0};
    uint32_t offsetWidth{0};
    uint32_t scaleHeight{0};
    uint32_t scaleWidth{0};
};

};     // namespace AscendC
#endif // AICORE_ADV_API_QUANTIZATION_ASCEND_ANTIQUANT_UTILS_H