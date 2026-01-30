/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file tensor_tile_fixpipe_routing.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_FIXPIPE_TENSOR_TILE_FIXPIPE_ROUTING_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_FIXPIPE_TENSOR_TILE_FIXPIPE_ROUTING_H

#include "impl/experimental/tensor_api/arch/tile_api/fixpipe/tensor_tile_fixpipe_four_dim_2201_l0c_gm.h"
#include "impl/experimental/tensor_api/arch/tile_api/fixpipe/tensor_tile_fixpipe_quant_four_dim_2201_l0c_gm.h"

namespace AscendC {
namespace TileInternal {

class FixpipeIgnore {
public:
    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {}

    template <typename T, typename U, const FixpipeTrait& trait, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {}
};

template <Hardware dstTPos, Hardware srcTpos, uint32_t Version, size_t dimension>
struct FixpipeTensor2Tensor {
    using type = FixpipeIgnore;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = FixpipeFourDim2201L0C2GM;
};

class FixpipeQuantIgnore {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant) {}

    template <typename T, typename U, typename V, const FixpipeTrait& trait, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Coord& coord) {}
};

template <Hardware dstTPos, Hardware srcTpos, uint32_t Version, size_t dimension>
struct FixpipeQuantTensor2Tensor {
    using type = FixpipeQuantIgnore;
};

template <>
struct FixpipeQuantTensor2Tensor<Hardware::GM, Hardware::L0C, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = FixpipeQuantFourDim2201L0C2GM;
};
} // namespace TileInternal
} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_FIXPIPE_TENSOR_TILE_FIXPIPE_ROUTING_H