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
 * \file nz2nd.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2UB_NZ2ND_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2UB_NZ2ND_H

#include "impl/experimental/tensor_api/arch/cube_datamove/fixpipe/fixpipe_utils.h"
#include "impl/experimental/tensor_api/arch/cube_datamove/fixpipe/npu_arch_3510/instruction.h"
#include "impl/experimental/tensor_api/arch/utils/utils.h"

namespace AscendC {
namespace Te {

class Fixpipe2UbNz2NdBase3510 {
public:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        SetRegisterImpl<trait, T, U>(dst, src);
        DataCopyImpl<trait, quantPre, T, U, Coord>(dst, src, coord);
    }

private:
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        CheckFormat::CheckNDTemplate<T>();
        CheckFormat::CheckL0CNZTemplate<U>();
        CheckDataTypeFor3510::CheckL0C2UbDataType<quantPre, T, U>();

    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline void SetRegisterImpl(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNdStride = 0;
        uint32_t dstNdStride = 0;
        SetRegisterBase3510 setRegisterInst;
        setRegisterInst.SetRegister(ndNum, dstNdStride, srcNdStride);
    }
    
    template <const FixpipeTrait& trait, QuantMode_t quantPre, typename T, typename U, typename Coord>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, quantPre, T, U>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = Std::min(GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) - Std::get<1>(coord));
        uint32_t mSize = Std::min(GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout),
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) *
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) - Std::get<0>(coord));
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        uint8_t dualDstCtl = trait.dualDstCtl;

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = trait.unitFlag;
        bool subBlockId = false;
        bool nz2ndEn = true;
        bool nz2dnEn = false;
        auto dstDNTensor = dst(coord, dst.Layout().Shape());
        CopyMatrixCcToUbBase3510 copyInst;
        copyInst.DataCopy<trait, quantPre, T, U>(dstDNTensor, src, nSize, mSize, srcStride, dstStride, dualDstCtl,
            reluEn, unitFlag, subBlockId, nz2ndEn, nz2dnEn);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2UB_NZ2ND_H