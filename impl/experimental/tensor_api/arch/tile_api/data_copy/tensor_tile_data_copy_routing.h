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
 * \file tensor_tile_data_copy_routing.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_DATA_COPY_TENSOR_TILE_DATA_COPY_ROUTING_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_DATA_COPY_TENSOR_TILE_DATA_COPY_ROUTING_H

#include "tensor_tile_data_copy_four_dim_2201_gm_l1.h"
#include "tensor_tile_data_copy_four_dim_2201_l1_bt.h"
#include "tensor_tile_data_copy_four_dim_2201_l1_fb.h"

namespace AscendC {
namespace TileInternal {

class DataCopyIgnore {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {}
};

template <Hardware dstTPos, Hardware srcTpos, uint32_t Version, size_t dimension>
struct DataCopyTensor2Tensor {
    using type = DataCopyIgnore;
};

template <>
struct DataCopyTensor2Tensor<Hardware::L1, Hardware::GM, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = DataCopyFourDim2201GM2L1;
};

template <>
struct DataCopyTensor2Tensor<Hardware::BIAS, Hardware::L1, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = CopyCbufToBT2201;
};

template <>
struct DataCopyTensor2Tensor<Hardware::FIXBUF, Hardware::L1, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = CopyCbufToFB2201;
};
} // namespace TileInternal
} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_DATA_COPY_TENSOR_TILE_DATA_COPY_ROUTING_H