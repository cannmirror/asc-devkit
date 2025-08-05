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
 * \file tile_copy_policy.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_TILE_TILE_COPY_POLICY_H
#define ACT_INCLUDE_MATMUL_TILE_TILE_COPY_POLICY_H

#include "../../utils/arch.h"

namespace Act {
namespace Gemm {
namespace Tile {
//
// tile copy policies
//
struct CopyWithParams {};
struct CopyOutSplitMWithParams {};
struct CopyOutSplitNWithParams {};
struct CopyWithLayout {};
struct CopyEnUnitFlagWithLayout {};
struct CopySparseWithLayout {};
struct CopyNoGmIn {};
struct CopyBasedBaseK {};
struct CopyInAndCopyOutSplitMWithParams {};

//
// primary template class of Copy
//
template <class ArchTag, class DispatchPolicy, class DataType, class DstTrait, class SrcTrait, typename T = void,
          const auto& cfg = CFG_NORM>
struct Copy {};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
