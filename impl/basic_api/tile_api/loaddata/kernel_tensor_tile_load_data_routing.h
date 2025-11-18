/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
 * \file kernel_tensor_tile_load_data_routing.h
 * \brief
 */

#ifndef IMPL_TILE_API_KERNEL_TENSOR_TILE_LOAD_DATA_ROUTING_H
#define IMPL_TILE_API_KERNEL_TENSOR_TILE_LOAD_DATA_ROUTING_H

#include "kernel_tensor_tile_load_data_four_dim_3101_l1_l0a.h"
#include "kernel_tensor_tile_load_data_four_dim_3101_l1_l0b.h"

namespace AscendC {
namespace TileInternal {

class LoadDataIgnore {
public:
    template<typename T, typename U, const LoadDataTrait& config>
    __aicore__ inline void Run(const T& dst, const U& src) {}
};

template <TPosition dstTPos, TPosition srcTPos, uint32_t Version, size_t dimension>
struct LoadDataTensor2Tensor {
    using type = LoadDataIgnore;
};

template <>
struct LoadDataTensor2Tensor<TPosition::A2, TPosition::A1, ArchVersion::V3101, FOUR_DIM_DATA>
{
    using type = LoadDataFourDim3101L12L0A;
};

template <>
struct LoadDataTensor2Tensor<TPosition::B2, TPosition::B1, ArchVersion::V3101, FOUR_DIM_DATA>
{
    using type = LoadDataFourDim3101L12L0B;
};

} // namespace TileInternal
} // namespace AscendC
#endif // IMPL_TILE_API_KERNEL_TENSOR_TILE_LOAD_DATA_ROUTING_H