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
 * \file mx.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_MX_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_MX_H

#include "impl/experimental/tensor_api/arch/cube_compute/mmad/npu_arch_3510/instruction.h"
#include "impl/experimental/tensor_api/arch/utils/check_format.h"
#include "impl/experimental/tensor_api/arch/utils/check_data_type_3510.h"

namespace AscendC {
namespace Te {

class MmadMx {
public:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename Params>    
    __aicore__ inline void Run(const T& dst, const U& fm, const S& filter, const Params& params) 
    {   
        MmadImpl<trait, T, U, S>(dst, fm, filter, params);
    }

private:
    template <const MmadTrait& trait, typename T, typename U, typename S>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using dstDataType = typename T::elementType;
        using fmDataType = typename U::elementType;
        using filterDataType = typename S::elementType;
        CheckFormat::CheckL0CNZTemplate<T>();
        CheckFormat::CheckNZTemplate<U>();
        CheckFormat::CheckZNTemplate<S>();
        CheckDataTypeFor3510::CheckMxMmadDataType<Hardware::L0C, dstDataType, fmDataType, filterDataType, dstDataType>();
    }

    template <const MmadTrait& trait, typename T, typename U, typename S, typename Params>
    __aicore__ inline void MmadImpl(const T& dst, const U& fm, const S& filter, const Params& params)
    {
        CheckTemplate<trait, T, U, S>();
        using fmType = typename U::elementType;
        auto fmLayout = fm.Layout();
        auto dstLayout = dst.Layout();

        uint16_t m = GetEleFromLayout<decltype(fmLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(fmLayout) * FRACTAL_FIXED;
        uint16_t k = GetEleFromLayout<decltype(fmLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(fmLayout) * C0_SIZE / sizeof(fmType);
        uint16_t n = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * FRACTAL_FIXED;

        MmadMxInstr mmadMxInstr;
        mmadMxInstr.Mmad(dst, fm, filter, m, k, n, params.unitFlag, trait.disableGemv, trait.cmatrixSource, 
            params.cmatrixInitVal);
    }
};

} // namespace Te
} // namespace AscendC

#endif