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
 * \file nz2zn.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_NZ2ZN_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_NZ2ZN_H

#include "impl/experimental/tensor_api/arch/cube_datamove/load_data/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {
class LoadDataFourDim3510L12L0BNZ2ZN {

public:
    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src) {
        LoadDataImpl<TraitHolder<trait, true>::traitTransposed, T, U>(dst, src);
    }

private:
    template<const LoadDataTrait& trait, bool transpose>
    struct TraitHolder {
        static constexpr LoadDataTrait traitTransposed = LoadDataTrait(trait, transpose);
    };

    template<typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "LoadDataFourDim3510L12L0B Layout->Shape->Row->ZeroDim is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim3510L12L0B Layout->Shape->Column->ZeroDim is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim3510L12L0B Layout->Stride->Row-ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "LoadDataFourDim3510L12L0B Layout->Stride->Column->ZeroDim is not Std::Int<1> type!");
    }
    
    template<typename T>
    __aicore__ inline constexpr void CheckZNTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
            "LoadDataFourDim3510L12L0B Layout->Shape->Column->ZeroDim is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim3510L12L0B Layout->Shape->Row->ZeroDim is not Std::Int<C0Size/Type> type!"); 

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim3510L12L0B Layout->Stride->Column-ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>,
            "LoadDataFourDim3510L12L0B Layout->Stride->Row->ZeroDim is not Std::Int<1> type!");
    }

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        CheckZNTemplate<T>();
        CheckNZTemplate<U>();
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__cb__ half, __cbuf__ half>, 
            Std::tuple<__cb__ int16_t, __cbuf__ int16_t>, Std::tuple<__cb__ uint16_t, __cbuf__ uint16_t>,
            Std::tuple<__cb__ bfloat16_t, __cbuf__ bfloat16_t>, Std::tuple<__cb__ uint32_t, __cbuf__ uint32_t>, 
            Std::tuple<__cb__ int32_t, __cbuf__ int32_t>, Std::tuple<__cb__ float, __cbuf__ float>, 
            Std::tuple<__cb__ uint8_t, __cbuf__ uint8_t>, Std::tuple<__cb__ int8_t, __cbuf__ int8_t>, 
            Std::tuple<__cb__ fp8_e4m3fn_t, __cbuf__ fp8_e4m3fn_t>, Std::tuple<__cb__ fp8_e5m2_t, __cbuf__ fp8_e5m2_t>, 
            Std::tuple<__cb__ fp4x2_e2m1_t, __cbuf__ fp4x2_e2m1_t>, Std::tuple<__cb__ fp4x2_e1m2_t, __cbuf__ fp4x2_e1m2_t>,
            Std::tuple<__cb__ hifloat8_t, __cbuf__ hifloat8_t>>, "The data type is not supported.");
#endif
    }

    template <const LoadDataTrait& trait, typename T, typename U>
    __aicore__ inline void LoadDataImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto mStartPosition = 0;
        auto kStartPosition = 0;
        auto mStep = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) / FRACTAL_FIXED;
        auto kStep = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) *
                GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) * sizeof(DstType) / C0_SIZE;
        // Nz -> Zn
        uint32_t STRIDE_UNIT = FRACTAL_FIXED * (C0_SIZE / sizeof(DstType));
        auto srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / STRIDE_UNIT;
        auto dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout) / STRIDE_UNIT;
        LoadCbufToCbBase loadCbufToCb;
        loadCbufToCb.template LoadData<trait>(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_3510_LOAD_DATA_L12L0B_NZ2ZN_H