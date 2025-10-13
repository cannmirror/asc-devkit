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
 * \file mean_utils.h
 * \brief
 */
#ifndef LIB_REDUCE_MEAN_UTILS_H
#define LIB_REDUCE_MEAN_UTILS_H

namespace AscendC {
struct MeanParams {
    uint32_t outter = 1;
    uint32_t inner;  // inner = 32-byte alignment of n, inner = (n *sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T)
    uint32_t n;      // actual length of the tensor
};

}; // namespace AscendC
#endif // LIB_REDUCE_MEAN_UTILS_H