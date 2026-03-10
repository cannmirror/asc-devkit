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
 * \file mx_bias.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_MX_BIAS_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_NPU_ARCH_3510_MMAD_MX_BIAS_H

#include "impl/experimental/tensor_api/arch/cube_compute/mmad/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class MmadMxBias {
public:
    template <const MmadTrait& trait, typename T, typename U, typename S, typename V, typename Params>    
    __aicore__ inline void Run(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params) 
    {
        MmadImpl<trait, T, U, S, V>(dst, fm, filter, bias, params);
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

    template<Hardware biasPos, typename biasDataType, typename dstDataType, typename fmDataType, typename filterDataType>
    __aicore__ inline constexpr void CheckMmadTypes()
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510     
    if constexpr (biasPos == Hardware::BIAS) {
        static_assert(Std::is_one_of_v<Std::tuple<biasDataType, dstDataType, fmDataType, filterDataType>,
            Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e2m1_t>,
            Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e1m2_t>,
            Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e2m1_t>,
            Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e1m2_t>,
            Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
            Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
            Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
            Std::tuple<__biasbuf__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>>, 
            "The data type is not supported for BIAS position.");
    } else if constexpr (biasPos == Hardware::L0C) {
        static_assert(Std::is_one_of_v<Std::tuple<biasDataType, dstDataType, fmDataType, filterDataType>,
            Std::tuple<__cc__ float, __cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e2m1_t>,
            Std::tuple<__cc__ float, __cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e1m2_t>,
            Std::tuple<__cc__ float, __cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e2m1_t>,
            Std::tuple<__cc__ float, __cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e1m2_t>,
            Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
            Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
            Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
            Std::tuple<__cc__ float, __cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>>, 
            "The data type is not supported for L0C position.");
    }
#endif
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
        CheckMmadTypes<biasPos, biasDataType, dstDataType, fmDataType, filterDataType>();

    }

    template <const MmadTrait& trait, typename T, typename U, typename S, typename V, typename Params>
    __aicore__ inline void MmadImpl(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params)
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
        MmadMxBiasInstr mmadMxBiasInstr;
        mmadMxBiasInstr.Mmad(dst, fm, filter, bias, m, k, n, params.unitFlag, trait.disableGemv, cmatrixSource, 
            params.cmatrixInitVal);
    }
};

} // namespace Te
} // namespace AscendC

#endif