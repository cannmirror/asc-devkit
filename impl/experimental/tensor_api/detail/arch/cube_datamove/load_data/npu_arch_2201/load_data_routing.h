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
 * \file load_data_routing.h
 * \brief
 */

#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_ROUTING_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_ROUTING_H

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/load_data/npu_arch_2201/load_data_four_dim_2201_l1_l0a.h"
#include "impl/experimental/tensor_api/detail/arch/cube_datamove/load_data/npu_arch_2201/load_data_four_dim_2201_l1_l0b.h"

namespace AscendC {
namespace TensorInternal {

class LoadDataIgnore {
public:
    template<typename T, typename U, const LoadDataTrait& config, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {}
};

template <Hardware dstPos, Hardware srcPos, uint32_t Version, size_t dimension>
struct LoadDataTensor2Tensor {
    using type = LoadDataIgnore;
};

template <>
struct LoadDataTensor2Tensor<Hardware::L0A, Hardware::L1, ArchVersion::V2201, FOUR_DIM_DATA>
{
    using type = LoadDataFourDim2201L12L0A;
};

template <>
struct LoadDataTensor2Tensor<Hardware::L0B, Hardware::L1, ArchVersion::V2201, FOUR_DIM_DATA>
{
    using type = LoadDataFourDim2201L12L0B;
};
} // namespace TensorInternal
} // namespace AscendC
#endif // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_ROUTING_H