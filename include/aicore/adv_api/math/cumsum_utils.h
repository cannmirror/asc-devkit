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
 * \file cumsum_utils.h
 * \brief
 */
#ifndef AICORE_ADV_API_MATH_CUMSUM_UTILS_H
#define AICORE_ADV_API_MATH_CUMSUM_UTILS_H

namespace AscendC {

struct CumSumConfig {
    bool isLastAxis{true};
    bool isReuseSource{false};
    bool outputLastRow{false};
};

struct CumSumInfo {
    uint32_t outter{0};
    uint32_t inner{0}; // 32-byte aligned
};

};     // namespace AscendC
#endif // AICORE_ADV_API_MATH_CUMSUM_UTILS_H