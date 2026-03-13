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
 * \file mmad.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_H
#define IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_H

#include "impl/experimental/tensor_api/arch/cube_compute/mmad/npu_arch_3510/mmad/normal.h"
#include "impl/experimental/tensor_api/arch/cube_compute/mmad/npu_arch_3510/mmad/mx.h"

namespace AscendC {
namespace Te {

class Mmad3510{
public:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename Params>
    __aicore__ inline void Run(const T& dst, const U& fm, const S& filter, const Params& params) 
    {   
        Execute<trait, T, U, S>(dst, fm, filter, params);
    }

private:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename Params>
    __aicore__ inline void Execute(const T& dst, const U& fm, const S& filter, const Params& params) {
  
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            if constexpr (trait.mmadType == MmadTrait::MmadType::NORMAL) {
                MmadNormal normalStrategy;
                normalStrategy.Run<trait, T, U, S, Params>(dst, fm, filter, params);
            } else if constexpr (trait.mmadType == MmadTrait::MmadType::MX) {
                MmadMx mxStrategy;
                mxStrategy.Run<trait, T, U, S, Params>(dst, fm, filter, params);
            }
        }

    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_H