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
 * \file fixpipe_routing.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_ROUTING_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_ROUTING_H

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_four_dim_2201_l0c_gm.h"
#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_quant_four_dim_2201_l0c_gm.h"

namespace AscendC {
namespace TensorInternal {

class FixpipeIgnore {
public:
    template <const FixpipeTrait& trait, typename ...Args>
    __aicore__ inline void Run(const Args&... args) {}
};

template <Hardware dstPos, Hardware srcpos, Hardware quantpos, uint32_t Version, size_t dimension>
struct FixpipeTensor2Tensor {
    using type = FixpipeIgnore;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, Hardware::MAX, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = FixpipeFourDim2201L0C2GM;
};

template <>
struct FixpipeTensor2Tensor<Hardware::GM, Hardware::L0C, Hardware::L1, ArchVersion::V2201, FOUR_DIM_DATA> {
    using type = FixpipeQuantFourDim2201L0C2GM;
};
} // namespace TensorInternal
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_ROUTING_H