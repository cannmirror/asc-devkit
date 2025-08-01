/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_matmul_utils.h
 * \brief
 */

#ifndef AICORE_ADV_API_DETAIL_MATMUL_UTILS_BATCH_MATMUL_UTILS_H
#define AICORE_ADV_API_DETAIL_MATMUL_UTILS_BATCH_MATMUL_UTILS_H

#include "matmul_config_utils.h"
#include "matmul_type_def.h"

namespace AscendC {

template <typename A_TYPE, const auto& MM_CFG>
constexpr bool IsBmmEnableScheduler =
    DoMatmulNorm(MM_CFG)
    && ((A_TYPE::layout != LayoutMode::NONE && ToMatmulConfig(MM_CFG).batchMode == BatchMode::BATCH_LESS_THAN_L1)
        || (A_TYPE::layout == LayoutMode::NORMAL && ToMatmulConfig(MM_CFG).batchMode == BatchMode::BATCH_LARGE_THAN_L1)
        || (A_TYPE::layout == LayoutMode::NORMAL
            && ToMatmulConfig(MM_CFG).batchMode == BatchMode::SINGLE_LARGE_THAN_L1));

template <typename A_TYPE, const auto& MM_CFG>
constexpr bool IsBmmBatchScheduler = DoMatmulNorm(MM_CFG)
                                     && ((A_TYPE::layout != LayoutMode::NONE
                                          && ToMatmulConfig(MM_CFG).batchMode != BatchMode::SINGLE_LARGE_THAN_L1));

template <typename A_TYPE, const auto& MM_CFG>
constexpr bool IsBmmSingleScheduler = DoMatmulNorm(MM_CFG)
                                      && (A_TYPE::layout == LayoutMode::NORMAL
                                          && ToMatmulConfig(MM_CFG).batchMode == BatchMode::SINGLE_LARGE_THAN_L1);

} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATMUL_UTILS_BATCH_MATMUL_UTILS_H