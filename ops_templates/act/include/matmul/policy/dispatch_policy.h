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
 * \file dispatch_policy.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_POLICY_DISPATCH_POLICY_H
#define ACT_INCLUDE_MATMUL_POLICY_DISPATCH_POLICY_H

#include "../../utils/integral_constant.h"

namespace Act {
namespace Gemm {
//
// block schedule policies
//
struct KernelNaivePipeline {};     // Basic pipelining without caching or optimization
struct KernelMultiBlock {};        // Multi-block pipelined data transfer
struct KernelMultiBlockOnKAxis {}; // Multi-tile pipelined transfer with K-axis caching
struct KernelMmadPerBaseK {};      // Perform matrix multiplication with baseK granularity
struct KernelL1Input {};           // L1 input pipeline
struct KernelIterBatch {};         // Multi-tile pipelined transfer with batch caching
struct KernelMmadWithScale {};     // Multi-block with scale

//
// block matmul policies
//
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct GMMPerTile {
    using ScheduleType = KernelMmadPerBaseK;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct QuantMatmulMultiBlock {
    using ScheduleType = KernelMmadWithScale;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// naive pipeline without caching or optimization, implemented based on Layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulNaivePipelineWithLayout {
    using ScheduleType = KernelNaivePipeline;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// enable multi block load, no bias, no quant, input expressed as Layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// enable multi block load, support bias, no quant, input expressed as Layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockBiasWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// enable multi block load, no bias, no quant, implemented based on highlevel api
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlock {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// enable multi block load, no quant, implemented based on Layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockWithOutQue {
    using ScheduleType = KernelMultiBlockOnKAxis;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// enable multi block load with multi of batch, no quant, no bias, implemented base on layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulIterBatch {
    using ScheduleType = KernelIterBatch;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// enable multi block load, support bias, no quant, implemented based on highlevel api
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockBias {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// enable multi block load, with K-axis caching, no quant, implemented based on Layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockOnKAxisWithLayout {
    using ScheduleType = KernelMultiBlockOnKAxis;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// sparse matmul, enable multi block load, with K-axis caching, no quant, no bias, implemented based on Layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct SparseMatmulMultiBlockOnKAxisWithLayout {
    using ScheduleType = KernelMultiBlockOnKAxis;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// L1 input, no bias, implemented based on highlevel api
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1Input {
    using ScheduleType = KernelL1Input;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// L1 input, implemented based on highlevel api
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputBias {
    using ScheduleType = KernelL1Input;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// L1 input, no bias, implemented based on layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputWithLayout {
    using ScheduleType = KernelL1Input;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// L1 input, implemented based on layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputBiasWithLayout {
    using ScheduleType = KernelL1Input;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// L0C output, implemented based on layout
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL0COutputWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};
} // namespace Gemm
} // namespace Act
#endif
