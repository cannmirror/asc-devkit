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
 * \file block_mmad.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_H
#define ACT_INCLUDE_MATMUL_BLOCK_BLOCK_MMAD_H

#include <type_traits>
#include "../../utils/arch.h"
#include "../../utils/integral_constant.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
          class BiasType = CType, class TileCopy = void,
          typename = void // Supports specialization via DispatchPolicy type
          >
class BlockMmad {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
