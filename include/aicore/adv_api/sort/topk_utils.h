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
 * \file topk_utils.h
 * \brief
 */
#ifndef AICORE_ADV_API_SORT_TOPK_UTILS_H
#define AICORE_ADV_API_SORT_TOPK_UTILS_H

namespace AscendC {
struct TopKInfo {
    int32_t outter = 1;
    int32_t inner; // inner = 32-byte alignment of n
    int32_t n;     // actual length of the tensor
};
#ifndef ASCC_ENUM_TOPKMODE
#define ASCC_ENUM_TOPKMODE
enum class TopKMode {
    TOPK_NORMAL,
    TOPK_NSMALL,
};
#endif

};     // namespace AscendC
#endif // AICORE_ADV_API_SORT_TOPK_UTILS_H