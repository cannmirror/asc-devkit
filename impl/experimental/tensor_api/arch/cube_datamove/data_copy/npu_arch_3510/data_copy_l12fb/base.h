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
 * \file base.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12FB_BASE_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12FB_BASE_H

#include "impl/experimental/tensor_api/arch/utils/check_format.h"
#include "impl/experimental/tensor_api/arch/utils/check_data_type_3510.h"
#include "impl/experimental/tensor_api/arch/cube_datamove/data_copy/npu_arch_3510/data_copy_l12fb/instruction.h"

namespace AscendC {
namespace Te {

class CopyL12FBBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        DataCopyImpl<trait, T, U>(dst, src);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        CheckDataTypeFor3510::CheckL12FbDataType<dstType, srcType>();

        constexpr bool isFp8 = Std::is_same_v<dstType, __fbuf__ fp8_e8m0_t> && Std::is_same_v<srcType, __cbuf__ fp8_e8m0_t>;
        if constexpr (isFp8) {
            CheckFormat::CheckNDFp8Template<T>();
            CheckFormat::CheckNDFp8Template<U>();
        } else {
            CheckFormat::CheckNDTemplate<T>();
            CheckFormat::CheckNDTemplate<U>();
        }
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto DataCopyImpl(const T& dst, const U& src)
    {
        constexpr uint32_t C2PIPE2GM_UNIT = C0_SIZE * 2;
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint16_t srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint16_t srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
        uint16_t dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        uint16_t blockCount = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint16_t blockLen = CeilDivision(srcCol * sizeof(srcType), C2PIPE2GM_UNIT);
        uint16_t srcStride = CeilDivision(srcRow * sizeof(srcType), C0_SIZE);
        uint16_t dstStride = CeilDivision(dstRow * sizeof(dstType), C2PIPE2GM_UNIT);

        CopyL12FBInstr copyInstr;
        copyInstr.DataCopy(dst, src, blockCount, blockLen, srcStride, dstStride);
    }
};

} // namespace Te
} // namespace AscendC

#endif