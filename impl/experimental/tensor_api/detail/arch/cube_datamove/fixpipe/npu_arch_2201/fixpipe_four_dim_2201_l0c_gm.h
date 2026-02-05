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
 * \file fixpipe_four_dim_2201_l0c_gm.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_FOUR_DIM_2201_L0C_GM_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_FOUR_DIM_2201_L0C_GM_H

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_2201_base.h"

namespace AscendC {
namespace TensorInternal {

class FixpipetNz2Nz2201Base : public Copy2201MatrixCcToGmBase {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        CheckCoord<T, U, Coord>(dst, src, coord);
        auto params = GenFixpipeParams<trait, T, U>(dst, src); 
        auto dstNzTensor = MakeTensorWithCoord<T, Coord>(dst, coord, 0);

        DataCopy<trait, T, U, decltype(params)>(dstNzTensor, src, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Shape->Column->ZeroDim, is not Std::Int<16> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Stride->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Fixpipe Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__gm__ float, __cc__ float>,
            Std::tuple<__gm__ int32_t, __cc__ int32_t>>, "The data type is not supported.");
#endif
        CheckL0CNZTemplate<T>();
        CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();
        using dstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout) * sizeof(dstType) / C0_SIZE;

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, reluEn, unitFlag, isChannelSplit, nz2ndEn);
        return params;
    }
};

class FixpipetNz2Nd2201Base : public Copy2201MatrixCcToGmBase, public SetRegister2201Base {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        CheckCoord<T, U, Coord>(dst, src, coord);
        auto ndParams = GenRegisterParams<trait, T, U>(dst, src);
        SetRegister<decltype(ndParams)>(ndParams);
        auto params = GenFixpipeParams<trait, T, U>(dst, src);
        auto dstNDTensor = MakeTensorWithCoord<T, Coord>(dst, coord, 0);

        DataCopy<trait, T, U, decltype(params)>(dstNDTensor, src, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Shape->Column->ZeroDim, is not Std::Int<16> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Stride->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Fixpipe Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Fixpipe Src Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Fixpipe Src Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Fixpipe Src Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Fixpipe Src Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "Fixpipe Src Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__gm__ float, __cc__ float>,
            Std::tuple<__gm__ int32_t, __cc__ int32_t>>, "The data type is not supported.");
#endif
        CheckNDTemplate<T>();
        CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint64_t ndNum = 1;
        uint64_t srcNdStride = 0;
        uint64_t dstNdStride = 0;
        auto params = Std::make_tuple(ndNum, dstNdStride, srcNdStride);
        return params;
    }
    
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, reluEn, unitFlag, isChannelSplit, nz2ndEn);
        return params;
    }
};

class FixpipeFourDim2201L0C2GM : public FixpipetNz2Nz2201Base, public FixpipetNz2Nd2201Base {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        if constexpr (IsL0cNZFormat<U>::value && IsL0cNZFormat<T>::value) {
            FixpipetNz2Nz2201Base::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsL0cNZFormat<U>::value && IsNDFormat<T>::value) {
            FixpipetNz2Nd2201Base::Run<trait, T, U, Coord>(dst, src, coord);
        }
    }
};
} // namespace TensorInternal
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_FOUR_DIM_2201_L0C_GM_H