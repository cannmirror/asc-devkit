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
 * \file tile_copy.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_TILE_TILE_COPY_H
#define ACT_INCLUDE_MATMUL_TILE_TILE_COPY_H

#include "copy_in/copy_in_with_params.h"
#include "copy_in/copy_in_with_layout.h"

#include "copy_out/copy_out_with_params.h"
#include "copy_out/copy_out_split_m_with_params.h"
#include "copy_out/copy_out_split_n_with_params.h"
#include "copy_out/copy_out_with_layout.h"
#include "copy_out/copy_out_enunitflag_with_layout.h"

#include "load_data/copy_a1_to_a2_with_layout.h"
#include "load_data/copy_b1_to_b2_sparse_with_layout.h"
#include "load_data/copy_b1_to_b2_with_layout.h"
#include "tile_copy_policy.h"

namespace Act {
namespace Gemm {
namespace Tile {
template <class ArchTag, class DispatchPolicy>
struct TileCopy {};

template <class ArchTag>
struct TileCopy<ArchTag, CopyWithParams> {
    using CopyPolicy = CopyWithParams;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, void, void, InputType, void, COPY_CFG>;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, void, void, InputType, void, COPY_CFG>;

    template <class InputType, class OutputType>
    using CopyCo1ToOut = Copy<ArchTag, CopyPolicy, void, OutputType, InputType>;
};

template <class ArchTag>
struct TileCopy<ArchTag, CopyWithLayout> {
    using CopyPolicy = CopyWithLayout;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class OutputType, class DstTrait, class SrcTrait>
    using CopyCo1ToOut = Copy<ArchTag, CopyPolicy, OutputType, DstTrait, SrcTrait>;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopyPolicy, BType, DstTrait, SrcTrait>;
};

template <>
struct TileCopy<Arch::Ascend910B, CopyEnUnitFlagWithLayout> {
    using ArchTag = Arch::Ascend910B;
    using CopyPolicy = CopyWithLayout;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class OutputType, class DstTrait, class SrcTrait>
    using CopyCo1ToOut = Copy<ArchTag, CopyEnUnitFlagWithLayout, OutputType, DstTrait, SrcTrait>;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopyPolicy, BType, DstTrait, SrcTrait>;
};

template <>
struct TileCopy<Arch::Ascend910B, CopySparseWithLayout> {
    using ArchTag = Arch::Ascend910B;
    using CopyPolicy = CopyWithLayout;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class OutputType, class DstTrait, class SrcTrait>
    using CopyCo1ToOut = Copy<ArchTag, CopyPolicy, OutputType, DstTrait, SrcTrait>;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopySparseWithLayout, BType, DstTrait, SrcTrait>;
};

template <>
struct TileCopy<Arch::Ascend910_95, CopyInAndCopyOutSplitMWithParams> {
    using ArchTag = Arch::Ascend910_95;
    using CopyPolicy = CopyWithParams;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, void, void, InputType, void, COPY_CFG>;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, void, void, InputType, void, COPY_CFG>;

    template <class InputType, class OutputType>
    using CopyCo1ToOut = Copy<ArchTag, CopyOutSplitMWithParams, void, OutputType, InputType>;
};

template <>
struct TileCopy<Arch::Ascend910_95, CopyOutSplitMWithParams> {
    using ArchTag = Arch::Ascend910_95;

    template <class InputType, class OutputType>
    using CopyCo1ToOut = Copy<ArchTag, CopyOutSplitMWithParams, void, OutputType, InputType>;
};

template <>
struct TileCopy<Arch::Ascend910_95, CopyOutSplitNWithParams> {
    using ArchTag = Arch::Ascend910_95;

    template <class InputType, class OutputType>
    using CopyCo1ToOut = Copy<ArchTag, CopyOutSplitNWithParams, void, OutputType, InputType>;
};

template <>
struct TileCopy<Arch::Ascend910_95, CopyNoGmIn> {
    using ArchTag = Arch::Ascend910_95;
    using CopyPolicy = CopyWithParams;

    template <class InputType, class OutputType>
    using CopyCo1ToOut = Copy<ArchTag, CopyPolicy, void, OutputType, InputType>;
};

template <>
struct TileCopy<Arch::Ascend910_95, CopyBasedBaseK> {
    using CopyPolicy = CopyBasedBaseK;
    using ArchTag = Arch::Ascend910_95;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToA1 = void;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToB1 = void;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopyPolicy, BType, DstTrait, SrcTrait>;

    template <class InputType, class OutputType>
    using CopyCo1ToOut = Copy<ArchTag, CopyBasedBaseK, void, OutputType, InputType>;
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
