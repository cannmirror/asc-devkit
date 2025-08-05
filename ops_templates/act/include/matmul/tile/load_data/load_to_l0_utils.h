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
 * \file load_to_l0_utils.h
 * \brief
 */

#ifndef ACT_INCLUDE_MATMUL_TILE_LOAD_DATA_LOAD_TO_L0_UTILS_H
#define ACT_INCLUDE_MATMUL_TILE_LOAD_DATA_LOAD_TO_L0_UTILS_H

namespace Act {
namespace Gemm {
namespace Tile {
constexpr uint16_t HW_N0 = 16;
constexpr uint16_t HW_M0 = 16;
constexpr uint16_t ALIGN_NUM = 16;
constexpr uint64_t M_POS_BIT = 48;
constexpr uint64_t K_POS_BIT = 32;
constexpr uint64_t M_STEP_BIT = 16;
constexpr uint8_t INDEX_SHIFT = 2;
constexpr uint8_t K_STEP_MIN_VAL_B32 = 2;
constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};

} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif