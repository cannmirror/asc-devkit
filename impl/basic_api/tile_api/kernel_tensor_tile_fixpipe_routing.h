/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_tensor_tile_fixpipe_routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_ROUTING_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_ROUTING_H

#include "kernel_tensor_tile_fixpipe_four_dim_3101_l0c_gm.h"
#include "kernel_tensor_tile_fixpipe_quant_four_dim_3101_l0c_gm.h"

namespace AscendC {
namespace TileInternal {

class FixpipeIgnore {
public:
    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {}
};

template <TPosition dstTPos, TPosition srcTpos, uint32_t Version, size_t dimension>
struct FixpipeTensor2Tensor {
    using type = FixpipeIgnore;
};

template <>
struct FixpipeTensor2Tensor<TPosition::GM, TPosition::CO1, ArchVersion::V3101, FOUR_DIM_DATA> {
    using type = FixpipeFourDim3101L0C2GM;
};

class FixpipeQuantIgnore {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant) {}
};

template <TPosition dstTPos, TPosition srcTpos, uint32_t Version, size_t dimension>
struct FixpipeQuantTensor2Tensor {
    using type = FixpipeQuantIgnore;
};

template <>
struct FixpipeQuantTensor2Tensor<TPosition::GM, TPosition::CO1, ArchVersion::V3101, FOUR_DIM_DATA> {
    using type = FixpipeQuantFourDim3101L0C2GM;
};

} // namespace TileInternal
} // namespace AscendC

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_ROUTING_H