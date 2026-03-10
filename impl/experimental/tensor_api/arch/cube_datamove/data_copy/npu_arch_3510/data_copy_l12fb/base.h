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
    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "CopyCbufToFB Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "CopyCbufToFB Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "CopyCbufToFB Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "CopyCbufToFB Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "CopyCbufToFB Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }

    template <typename T>
    __aicore__ inline constexpr void CheckNDFp8Template()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<2>>, "CopyCbufToFB Layout->Shape->Row->ZeroDim, is not Std::Int<2> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "CopyCbufToFB Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>, "CopyCbufToFB Layout->Stride->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "CopyCbufToFB Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<MX_SCALE_K0>>, "CopyCbufToFB Layout->Stride->Column->OneDim, is not Std::Int<MX_SCALE_K0> type!");
    }
    
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__fbuf__ bool, __cbuf__ bool>, 
            Std::tuple<__fbuf__ int8_t, __cbuf__ int8_t>, Std::tuple<__fbuf__ uint8_t, __cbuf__ uint8_t>, 
            Std::tuple<__fbuf__ hifloat8_t, __cbuf__ hifloat8_t>, Std::tuple<__fbuf__ fp8_e5m2_t, __cbuf__ fp8_e5m2_t>, 
            Std::tuple<__fbuf__ fp8_e4m3fn_t, __cbuf__ fp8_e4m3fn_t>, Std::tuple<__fbuf__ fp8_e8m0_t, __cbuf__ fp8_e8m0_t>, 
            Std::tuple<__fbuf__ int16_t, __cbuf__ int16_t>, Std::tuple<__fbuf__ uint16_t, __cbuf__ uint16_t>, 
            Std::tuple<__fbuf__ half, __cbuf__ half>, Std::tuple<__fbuf__ float, __cbuf__ float>, 
            Std::tuple<__fbuf__ bfloat16_t, __cbuf__ bfloat16_t>, Std::tuple<__fbuf__ int32_t, __cbuf__ int32_t>, 
            Std::tuple<__fbuf__ uint32_t, __cbuf__ uint32_t>, Std::tuple<__fbuf__ int64_t, __cbuf__ int64_t>, 
            Std::tuple<__fbuf__ uint64_t, __cbuf__ uint64_t>, Std::tuple<__fbuf__ double, __cbuf__ double>>, 
            "The data type is not supported.");
#endif

        constexpr bool isFp8 = Std::is_same_v<dstType, __fbuf__ fp8_e8m0_t> && Std::is_same_v<srcType, __cbuf__ fp8_e8m0_t>;
        if constexpr (isFp8) {
            CheckNDFp8Template<T>();
            CheckNDFp8Template<U>();
        } else {
            CheckNDTemplate<T>();
            CheckNDTemplate<U>();
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