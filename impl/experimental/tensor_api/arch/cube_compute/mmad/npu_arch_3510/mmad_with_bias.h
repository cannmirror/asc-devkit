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
 * \file mmad_with_bias.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_WITH_BIAS_H
#define IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_WITH_BIAS_H

#include "impl/experimental/tensor_api/arch/cube_compute/mmad/npu_arch_3510/mmad/bias.h"
#include "impl/experimental/tensor_api/arch/cube_compute/mmad/npu_arch_3510/mmad/mx_bias.h"

namespace AscendC {
namespace Te {

class MmadWithBias3510{
public:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename V, typename Params>
    __aicore__ inline void Run(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params) 
    {   
        Execute<trait, T, U, S, V, Params>(dst, fm, filter, bias, params);
    }

private:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename V, typename Params>
    __aicore__ inline void Execute(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params) {
        using dstType = typename T::elementType;
        using fmType = typename U::elementType;
        using filterType = typename S::elementType;

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            constexpr bool isMx = Std::is_one_of_v<Std::tuple<dstType, fmType, filterType>,
                Std::tuple<__cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e2m1_t>,
                Std::tuple<__cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e1m2_t>,
                Std::tuple<__cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e2m1_t>,
                Std::tuple<__cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e1m2_t>,
                Std::tuple<__cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
                Std::tuple<__cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
                Std::tuple<__cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
                Std::tuple<__cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>>;
            if constexpr (isMx) {
                MmadMxBias mxBiasStrategy;
                mxBiasStrategy.Run<trait, T, U, S, V>(dst, fm, filter, bias, params);
            } else {
                MmadBias biasStrategy;
                biasStrategy.Run<trait, T, U, S, V>(dst, fm, filter, bias, params);
            }
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_WITH_BIAS_H