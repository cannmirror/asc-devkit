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
 * \file fixpipe_l0c2gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2GM_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2GM_H

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_3510/fixpipe_base.h"

namespace AscendC {
namespace Te {

class FixpipetNz2NzBase3510 : public CopyMatrixCcToGmBase3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenFixpipeParams<trait, T, U, Coord>(dst, src, coord);
        auto dstNzTensor = dst(coord, dst.Layout().Shape());
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

        CheckL0CNZTemplate<T>();
        CheckL0CNZTemplate<U>();
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__gm__ float, __cc__ float>, 
            Std::tuple<__gm__ int32_t, __cc__ int32_t>>, "The data type is not supported.");
#endif
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
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
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst.Data().Get());

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = trait.unitFlag;
        bool isChannelSplit = trait.enableChannleSplit;
        bool nz2ndEn = false;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit,
            nz2ndEn, nz2dnEn);
        return params;
    }
};

class FixpipetNz2NdBase3510 : public CopyMatrixCcToGmBase3510, public SetRegisterBase3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto loop3Params = GenRegisterParams<trait, T, U>(dst, src);
        SetRegister<decltype(loop3Params)>(loop3Params);
        auto params = GenFixpipeParams<trait, T, U, Coord>(dst, src, coord);
        auto dstNDTensor = dst(coord, dst.Layout().Shape());
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
        CheckNDTemplate<T>();
        CheckL0CNZTemplate<U>();
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__gm__ float, __cc__ float>, 
            Std::tuple<__gm__ int32_t, __cc__ int32_t>>, "The data type is not supported.");
#endif
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNdStride = 0;
        uint32_t dstNdStride = 0;
        auto params = Std::make_tuple(ndNum, dstNdStride, srcNdStride);
        return params;
    }
    
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
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
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst.Data().Get());

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = trait.unitFlag;
        bool isChannelSplit = trait.enableChannleSplit;
        bool nz2ndEn = true;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit,
            nz2ndEn, nz2dnEn);

        return params;
    }
};

class FixpipetNz2DnBase3510 : public CopyMatrixCcToGmBase3510, public SetRegisterBase3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto loop3Params = GenRegisterParams<trait, T, U>(dst, src);
        SetRegister<decltype(loop3Params)>(loop3Params);
        auto params = GenFixpipeParams<trait, T, U, Coord>(dst, src, coord);
        auto dstDNTensor = dst(coord, dst.Layout().Shape());
        DataCopy<trait, T, U, decltype(params)>(dstDNTensor, src, params);
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
    __aicore__ inline constexpr void CheckDNTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Fixpipe Src->Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Fixpipe Src->Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Fixpipe Src->Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>, "Fixpipe Src->Layout->Stride->Row->OneDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Fixpipe Src->Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        CheckDNTemplate<T>();
        CheckL0CNZTemplate<U>();
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__gm__ float, __cc__ float>, 
            Std::tuple<__gm__ int32_t, __cc__ int32_t>>, "The data type is not supported.");
#endif
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t dnNum = 1;
        uint32_t srcNzMatrixStride = 0;
        uint32_t dstDnMatrixStride = 0;
        uint32_t srcNzC0Stride = 1;
        auto params = Std::make_tuple(dnNum, dstDnMatrixStride, srcNzMatrixStride, srcNzC0Stride);
        return params;
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<trait, T, U>();
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
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst.Data().Get());

        bool reluEn = trait.enableRelu;
        uint8_t unitFlag = trait.unitFlag;
        bool isChannelSplit = trait.enableChannleSplit;
        bool nz2ndEn = false;
        bool nz2dnEn = true;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit,
            nz2ndEn, nz2dnEn);

        return params;
    }
};

class FixpipeFourDimL0C2GM3510 : public FixpipetNz2NzBase3510, public FixpipetNz2NdBase3510, public FixpipetNz2DnBase3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto fourDimDst = PreProcess(dst);
        Execute<trait>(fourDimDst, src, coord);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Execute(const T& dst, const U& src, const Coord& coord) {
        if constexpr (IsL0cNZFormat<U>::value && IsL0cNZFormat<T>::value) {
            FixpipetNz2NzBase3510::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsL0cNZFormat<U>::value && IsNDFormat<T>::value) {
            FixpipetNz2NdBase3510::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsL0cNZFormat<U>::value && IsDNFormat<T>::value) {
            FixpipetNz2DnBase3510::Run<trait, T, U, Coord>(dst, src, coord);
        }
    }
};
} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_L0C2GM_H