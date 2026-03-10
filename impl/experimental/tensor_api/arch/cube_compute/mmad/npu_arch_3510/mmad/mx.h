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
    template <typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        constexpr auto c0Size = is_b4_type<type> ? C0_SIZE * 2 : C0_SIZE / sizeof(type);
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Dst Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<c0Size>>, "Dst Layout->Shape->Column->ZeroDim, is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<c0Size>>, "Dst Layout->Stride->Row->ZeroDim, is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Dst Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<c0Size * FRACTAL_FIXED>>,
            "Dst Layout->Stride->Row->OneDim, is not Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED> type!");
    }
    template<typename T>
    __aicore__ inline constexpr void CheckZNTemplate()
    {
        using dataType = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        constexpr auto c0Size = is_b4_type<dataType> ? C0_SIZE * 2 : C0_SIZE / sizeof(dataType);
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
            "Filter Layout->Shape->Column->ZeroDim is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<c0Size>>,
            "Filter Layout->Shape->Row->ZeroDim is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<c0Size>>,
            "Filter Layout->Stride->Column->ZeroDim is not Std::Int<C0Size/Type> type!");
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

    template <const MmadTrait& trait, typename T, typename U, typename S>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using dstDataType = typename T::elementType;
        using fmDataType = typename U::elementType;
        using filterDataType = typename S::elementType;
        CheckL0CNZTemplate<T>();
        CheckNZTemplate<U>();
        CheckZNTemplate<S>();    
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstDataType, fmDataType, filterDataType>,
            Std::tuple<__cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e2m1_t>,
            Std::tuple<__cc__ float, __ca__ fp4x2_e2m1_t, __cb__ fp4x2_e1m2_t>,
            Std::tuple<__cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e2m1_t>,
            Std::tuple<__cc__ float, __ca__ fp4x2_e1m2_t, __cb__ fp4x2_e1m2_t>,
            Std::tuple<__cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e4m3fn_t>,
            Std::tuple<__cc__ float, __ca__ fp8_e4m3fn_t, __cb__ fp8_e5m2_t>,
            Std::tuple<__cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e4m3fn_t>,
            Std::tuple<__cc__ float, __ca__ fp8_e5m2_t, __cb__ fp8_e5m2_t>>, 
            "The data type is not supported.");
#endif
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