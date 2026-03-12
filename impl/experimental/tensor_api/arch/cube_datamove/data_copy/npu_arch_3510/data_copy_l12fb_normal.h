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
 * \file data_copy_l12fb_normal.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12FB_NORMAL_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12FB_NORMAL_H

#include "impl/experimental/tensor_api/arch/utils/arch_utils.h"
#include "impl/experimental/tensor_api/arch/utils/check_format.h"
#include "impl/experimental/tensor_api/arch/utils/check_data_type_3510.h"
#include "impl/experimental/tensor_api/tensor/pointer_impl.h"
#include "impl/experimental/tensor_api/tensor/layout_impl.h"
#include "impl/experimental/tensor_api/tensor/local_tensor_impl.h"
#include "impl/experimental/tensor_api/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_l12fb/base.h"

namespace AscendC {
namespace Te {

class DataCopyFourDim3510L12FB {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        Execute<trait>(dst, src, ZeroCoord2DType{});
    }
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        Execute<trait>(dst, src, coord);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Execute(const T& dst, const U& src, const Coord& coord) {
        CopyL12FBBase fpStrategy;
        fpStrategy.Run<trait, T, U, Coord>(dst, src, coord);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H