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
 * \file kernel_tensor_tile_data_copy_routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_ROUTING_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_ROUTING_H

#include "kernel_tensor_tile_data_copy_four_dim_3101_gm_l1.h"
#include "kernel_tensor_tile_data_copy_four_dim_3101_l1_bt.h"
#include "kernel_tensor_tile_data_copy_four_dim_3101_l1_fb.h"

namespace AscendC {
namespace TileInternal {

class DataCopyIgnore {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {}
};

template <TPosition dstTPos, TPosition srcTpos, uint32_t Version, size_t dimension>
struct DataCopyTensor2Tensor {
    using type = DataCopyIgnore;
};

template <>
struct DataCopyTensor2Tensor<TPosition::A1, TPosition::GM, ArchVersion::V3101, FOUR_DIM_DATA> {
    using type = DataCopyFourDim3101GM2L1;
};

template <>
struct DataCopyTensor2Tensor<TPosition::B1, TPosition::GM, ArchVersion::V3101, FOUR_DIM_DATA> {
    using type = DataCopyFourDim3101GM2L1;
};

template <>
struct DataCopyTensor2Tensor<TPosition::C1, TPosition::GM, ArchVersion::V3101, FOUR_DIM_DATA> {
    using type = DataCopyFourDim3101GM2L1;
};

template <>
struct DataCopyTensor2Tensor<TPosition::C2, TPosition::C1, ArchVersion::V3101, FOUR_DIM_DATA> {
    using type = CopyCbufToBT;
};

template <>
struct DataCopyTensor2Tensor<TPosition::C2PIPE2GM, TPosition::C1, ArchVersion::V3101, FOUR_DIM_DATA> {
    using type = CopyCbufToFB;
};

} // namespace TileInternal
} // namespace AscendC

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_ROUTING_H