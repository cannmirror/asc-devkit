/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_tensor_tile_fixpipe_four_dim_3101_l0c_gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_FOUR_DIM_3101_L0C_GM_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_FOUR_DIM_3101_L0C_GM_H

#include "kernel_tensor_tile_fixpipe_common.h"

namespace AscendC {
namespace TileInternal {

class FixpipetNz2NzBase : public CopyMatrixCcToGmBase {
public:
    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenFixpipeParams<T, U, trait>(dst, src);
        DataCopy<T, U, decltype(params), trait>(dst, src, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::LiteType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Shape->Column->ZeroDim, is not Std::Int<16> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<FRACTAL_FIXED>>, "Fixpipe Layout->Stride->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Fixpipe Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        static_assert(Std::is_same_v<srcType, dstType>, "The source data and target data have inconsistent data types.");

        CheckL0CNZTemplate<T>();
        CheckL0CNZTemplate<U>();
        static_assert((Std::is_one_of_v<srcType, int32_t, float>), "The source data type is not supported.");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();
        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit,
            nz2ndEn, nz2dnEn);
        return params;
    }
};

class FixpipetNz2NdBase : public CopyMatrixCcToGmBase, SetRegisterBase {
public:
    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto loop3Params = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<decltype(loop3Params)>(loop3Params);
        auto params = GenFixpipeParams<T, U, trait>(dst, src);
        DataCopy<T, U, decltype(params), trait>(dst, src, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::LiteType;
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

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        static_assert(Std::is_same_v<srcType, dstType>, "The source data and target data have inconsistent data types.");
        CheckNDTemplate<T>();
        CheckL0CNZTemplate<U>();
        static_assert((Std::is_one_of_v<srcType, int32_t, float>), "The source data type is not supported.");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNdStride = 0;
        uint32_t dstNdStride = 0;
        auto params = Std::make_tuple(ndNum, dstNdStride, srcNdStride);
        return params;
    }
    
    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();
        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit,
            nz2ndEn, nz2dnEn);

        return params;
    }
};

class FixpipetNz2DnBase : public CopyMatrixCcToGmBase, SetRegisterBase {
public:
    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto loop3Params = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<decltype(loop3Params)>(loop3Params);
        auto params = GenFixpipeParams<T, U, trait>(dst, src);
        DataCopy<T, U, decltype(params), trait>(dst, src, params);
    }

private:
   template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::LiteType;
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

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        static_assert(Std::is_same_v<srcType, dstType>, "The source data and target data have inconsistent data types.");
        CheckDNTemplate<T>();
        CheckL0CNZTemplate<U>();
        static_assert((Std::is_one_of_v<srcType, int32_t, float>), "The source data type is not supported.");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t dnNum = 1;
        uint32_t srcNzMatrixStride = 0;
        uint32_t dstDnMatrixStride = 0;
        uint32_t srcNzC0Stride = 1;
        auto params = Std::make_tuple(dnNum, dstDnMatrixStride, srcNzMatrixStride, srcNzC0Stride);
        return params;
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenFixpipeParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();
        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        bool nz2dnEn = true;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit,
            nz2ndEn, nz2dnEn);

        return params;
    }
};

class FixpipeFourDim3101L0C2GM : public FixpipetNz2NzBase, public FixpipetNz2NdBase, public FixpipetNz2DnBase {
public:
    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        using srcTraitType = GetTensorTraitType<U>;
        using dstTraitType = GetTensorTraitType<T>;
        if constexpr (IsL0cNZFormat<srcTraitType>::value && IsL0cNZFormat<dstTraitType>::value) {
            FixpipetNz2NzBase::Run<T, U, trait>(dst, src);
        } else if constexpr (IsL0cNZFormat<srcTraitType>::value && IsNDFormat<dstTraitType>::value) {
            FixpipetNz2NdBase::Run<T, U, trait>(dst, src);
        } else if constexpr (IsL0cNZFormat<srcTraitType>::value && IsDNFormat<dstTraitType>::value) {
            FixpipetNz2DnBase::Run<T, U, trait>(dst, src);
        }
    }
};
} // namespace TileInternal
} // namespace AscendC

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_FOUR_DIM_3101_L0C_GM_H