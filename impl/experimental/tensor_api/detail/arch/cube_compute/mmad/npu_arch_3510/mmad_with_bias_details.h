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
 * \file mmad_with_bias_details.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_WITH_BIAS_DETAILS_H
#define IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_WITH_BIAS_DETAILS_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "include/experimental/tensor_api/tensor/make.h"

namespace AscendC {
namespace Te {

class MmadWithBiasGenParams3510
{
public:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename V, typename Params>
    __aicore__ inline auto GenParams(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params)
    {
        return GenParamsImpl<trait, T, U, S, V, Params>(dst, fm, filter, bias, params);
    }
private:
    template <typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Dst Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>, "Dst Layout->Shape->Column->ZeroDim, is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>, "Dst Layout->Stride->Row->ZeroDim, is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Dst Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED>>,
            "Dst Layout->Stride->Row->OneDim, is not Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED> type!");
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

    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Bias Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Bias Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Bias Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Bias Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "Bias Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }

    template <const MmadTrait& trait, typename T, typename U, typename S, typename V>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using dstDataType = typename T::elementType;
        using fmDataType = typename U::elementType;
        using filterDataType = typename S::elementType;
        using biasDataType = typename V::elementType;
        constexpr auto biasPos = GetHardPos<V>();

        CheckL0CNZTemplate<T>();
        CheckNZTemplate<U>();
        CheckZNTemplate<S>();
        CheckNDTemplate<V>();
        
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510     
        if constexpr (biasPos == Hardware::BIAS) {
            static_assert(Std::is_one_of_v<Std::tuple<biasDataType, dstDataType, fmDataType, filterDataType>,
                Std::tuple<__biasbuf__ int32_t, __cc__ int32_t, __ca__ int8_t, __cb__ int8_t>, 
                Std::tuple<__biasbuf__ float, __cc__ float, __ca__ half, __cb__ half>, 
                Std::tuple<__biasbuf__ float, __cc__ float, __ca__ bfloat16_t, __cb__ bfloat16_t>, 
                Std::tuple<__biasbuf__ fp8_e4m3fn_t, __cc__ fp8_e4m3fn_t, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>, 
                Std::tuple<__biasbuf__ fp8_e5m2_t, __cc__ fp8_e5m2_t, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>, 
                Std::tuple<__biasbuf__ fp8_e5m2_t, __cc__ fp8_e5m2_t, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>, 
                Std::tuple<__biasbuf__ fp8_e4m3fn_t, __cc__ fp8_e4m3fn_t, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>, 
                Std::tuple<__biasbuf__ float, __cc__ float, __ca__ float, __cb__ float>>, "The data type is not supported.");
        } else if constexpr (biasPos == Hardware::L0C) {
            static_assert(Std::is_one_of_v<Std::tuple<biasDataType, dstDataType, fmDataType, filterDataType>,
                Std::tuple<__cc__ int32_t, __cc__ int32_t, __ca__ int8_t, __cb__ int8_t>, 
                Std::tuple<__cc__ float, __cc__ float, __ca__ half, __cb__ half>, 
                Std::tuple<__cc__ float, __cc__ float, __ca__ bfloat16_t, __cb__ bfloat16_t>, 
                Std::tuple<__cc__ fp8_e4m3fn_t, __cc__ fp8_e4m3fn_t, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>, 
                Std::tuple<__cc__ fp8_e5m2_t, __cc__ fp8_e5m2_t, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>, 
                Std::tuple<__cc__ fp8_e5m2_t, __cc__ fp8_e5m2_t, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>, 
                Std::tuple<__cc__ fp8_e4m3fn_t, __cc__ fp8_e4m3fn_t, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>,
                Std::tuple<__cc__ float, __cc__ float, __ca__ float, __cb__ float>>, "The data type is not supported.");
        }
#endif
    }

    template <const MmadTrait& trait, typename T, typename U, typename S, typename V, typename Params>
    __aicore__ inline auto GenParamsImpl(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params)
    {
        CheckTemplate<trait, T, U, S, V>();
        using fmType = typename U::elementType;
        constexpr auto biasPos = GetHardPos<V>();
        auto fmLayout = fm.Layout();
        auto dstLayout = dst.Layout();

        uint16_t m = GetEleFromLayout<decltype(fmLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(fmLayout) * FRACTAL_FIXED;
        uint16_t k = GetEleFromLayout<decltype(fmLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(fmLayout) * C0_SIZE / sizeof(fmType);
        uint16_t n = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * FRACTAL_FIXED;
        bool cmatrixSource = false;
        if (biasPos == Hardware::BIAS) {
            cmatrixSource = true;
        }
        auto genParams = Std::make_tuple(m, k, n, params.unitFlag, trait.disableGemv, cmatrixSource, 
            params.cmatrixInitVal);
        return genParams;
    }
};

class MmadWithBiasCore3510
{
public:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename V, typename Params, typename K, size_t... Is>
    __aicore__ inline void Mmad(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params, const K& tupleParams, Std::index_sequence<Is...>)
    {
        // MTE2
        MmadImpl(dst.Data().Get(), fm.Data().Get(), filter.Data().Get(), 
            reinterpret_cast<uint64_t>(bias.Data().Get()), Std::get<Is>(tupleParams)...);
    }
private:
    template <typename T, typename U, typename S>
    __aicore__ inline void MmadImpl(__cc__ T* dst, __ca__ U* fm, __cb__ S* filter, uint64_t bias, uint16_t m, uint16_t k, uint16_t n,
        int8_t unitFlag, bool disableGemv, bool cmatrixSource, bool cmatrixInitVal) {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            mad(dst, fm, filter, bias, m, k, n, unitFlag, disableGemv, cmatrixSource, cmatrixInitVal);
        }
    }
};

class MmadWithBiasFourDim3510 : public MmadWithBiasCore3510, public MmadWithBiasGenParams3510
{
public:
    template <const MmadTrait& trait, typename ...Args>
    __aicore__ inline void Run(const Args&... args) 
    {
        auto params = GenParams<trait>(args...);
        Mmad<trait>(args..., params, tuple_sequence<decltype(params)>{});
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_WITH_BIAS_DETAILS_H