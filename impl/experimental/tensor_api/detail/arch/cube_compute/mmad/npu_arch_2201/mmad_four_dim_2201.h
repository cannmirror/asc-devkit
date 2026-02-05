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
 * \file mmad_four_dim_2201.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_2201_MMAD_FOUR_DIM_2201_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_2201_MMAD_FOUR_DIM_2201_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "include/experimental/tensor_api/tensor/make.h"

namespace AscendC {
namespace TensorInternal {

class MmadGenParams
{
public:
    template <typename T, typename U, typename S, const MmadTrait& trait>
    __aicore__ inline auto GenParams(const T& dst, const U& fm, const S& filter)
    {
        return GenParamsImpl<T, U, S, trait>(dst, fm, filter);
    }
private:
    template<typename T>
    __aicore__ inline constexpr void CheckZZTemplate()
    {
        using dataType = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(dataType)>>,
            "Fm Layout->Shape->Column->ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "Fm Layout->Shape->Row->ZeroDim is not Std::Int<16> type!"); 

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "Fm Layout->Stride->Column-ZeroDim is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(dataType)>>,
            "Fm Layout->Stride->Row->ZeroDim is not Std::Int<C0Size/Type> type!");
    }
    template<typename T>
    __aicore__ inline constexpr void CheckZNTemplate()
    {
        using dataType = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
            "Filter Layout->Shape->Column->ZeroDim is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<C0_SIZE / sizeof(dataType)>>,
            "Filter Layout->Shape->Row->ZeroDim is not Std::Int<C0Size/Type> type!"); 

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<C0_SIZE / sizeof(dataType)>>,
            "Filter Layout->Stride->Column-ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>,
            "Filter Layout->Stride->Row->ZeroDim is not Std::Int<1> type!");
    }
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Dst Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>, "Dst Layout->Shape->Column->ZeroDim, is not Std::Int<16> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<FRACTAL_FIXED>>, "Dst Layout->Stride->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Dst Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, typename S, const MmadTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using dstDataType = typename T::elementType;
        using fmDataType = typename U::elementType;
        using filterDataType = typename S::elementType;
        CheckL0CNZTemplate<T>();
        CheckZZTemplate<U>();
        CheckZNTemplate<S>();
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstDataType, fmDataType, filterDataType>,
            Std::tuple<__cc__ int32_t, __ca__ int8_t, __cb__ int8_t>, Std::tuple<__cc__ float, __ca__ half, __cb__ half>, 
            Std::tuple<__cc__ float, __ca__ bfloat16_t, __cb__ bfloat16_t>, Std::tuple<__cc__ float, __ca__ float, __cb__ float>>, 
            "The data type is not supported.");
#endif
    }

    template <typename T, typename U, typename S, const MmadTrait& trait>
    __aicore__ inline auto GenParamsImpl(const T& dst, const U& fm, const S& filter)
    {
        CheckTemplate<T, U, S, trait>();
        using fmType = typename U::elementType;
        auto fmLayout = fm.Layout();
        auto dstLayout = dst.Layout();

        uint16_t m = GetEleFromLayout<decltype(fmLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(fmLayout) * FRACTAL_FIXED;
        uint16_t k = GetEleFromLayout<decltype(fmLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(fmLayout) * C0_SIZE / sizeof(fmType);
        uint16_t n = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * FRACTAL_FIXED;
        auto params = Std::make_tuple(m, k, n, trait.unitFlag, trait.kDirectionAlign, trait.cmatrixSource, 
            trait.cmatrixInitVal);
        return params;
    }
};

class MmadCore
{
public:
    template <typename T, typename U, typename S, typename V, const MmadTrait& trait, size_t... Is>
    __aicore__ inline void Mmad(const T& dst, const U& fm, const S& filter, const V& tupleParams, Std::index_sequence<Is...>)
    {
        // MTE2
        MmadImpl(dst.Engine().Begin().Get(), fm.Engine().Begin().Get(), filter.Engine().Begin().Get(), Std::get<Is>(tupleParams)...);
    }
private:
    template <typename T, typename U, typename S>
    __aicore__ inline void MmadImpl(__cc__ T* dst, __ca__ U* fm, __cb__ S* filter, uint16_t m, uint16_t k, uint16_t n,
        int8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal) {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            mad(dst, fm, filter, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal);
        }
    }
};

class MmadFourDim2201 : public MmadCore, public MmadGenParams
{
public:
    template <typename T, typename U, typename S, const MmadTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& fm, const S& filter)
    {
        auto params = GenParams<T, U, S, trait>(dst, fm, filter);
        Mmad<T, U, S, decltype(params), trait>(dst, fm, filter, params, tuple_sequence<decltype(params)>{});
    }
};
} // namespace TensorInternal
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_2201_MMAD_FOUR_DIM_2201_H