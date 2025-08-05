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
 * \file block_scheduler_policy.h
 * \brief
 */

#ifndef ACT_BLOCK_SCHEDULER_SCHEDULER_POLICY_H
#define ACT_BLOCK_SCHEDULER_SCHEDULER_POLICY_H

namespace Act {
namespace Gemm {

// FULL_LOAD_MODE_表示全载模式 0: 非全载 2：B全载
template <uint64_t FULL_LOAD_MODE_ = 0>
struct BuiltInAswtScheduler {
    static constexpr uint64_t FULL_LOAD_MODE = FULL_LOAD_MODE_;
};
struct BuiltInIterBatchScheduler {};
struct IterateKScheduler {};
struct MisplaceCoreScheduler {};
struct L2NormMisplaceCoreScheduler {};
struct L2TailOptMisplaceCoreScheduler {};
struct AswtScheduler {};
struct QuantIterateKScheduler {};
struct GroupedMatmulAswtScheduler {};
struct GroupedMatmulPerTileAswtScheduler {};
struct GroupedMatmulAswtWithTailSplitScheduler {};
} // namespace Gemm
} // namespace Act
#endif