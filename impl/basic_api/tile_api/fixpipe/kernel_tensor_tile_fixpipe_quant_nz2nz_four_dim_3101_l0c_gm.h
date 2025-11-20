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
 * \file kernel_tensor_tile_fixpipe_quant_nz2nz_four_dim_3101_l0c_gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_NZ2NZ_FOUR_DIM_3101_L0C_GM_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_NZ2NZ_FOUR_DIM_3101_L0C_GM_H

#include "kernel_tensor_tile_fixpipe_common.h"

namespace AscendC {
namespace TileInternal {

class FixpipeNZ2NZSimpleQuant : public CopyMatrixCcToGmBase, public SetRegisterBase {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<V, decltype(registerParams)>(quant, registerParams);
        auto copyParams = GenFixpipeQuantParams<T, U, trait>(dst, src);
        DataCopy<T, U, decltype(copyParams), trait>(dst, src, copyParams);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::LiteType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Shape->Column->ZeroDim, is not Std::Int<16> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Stride->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "Fixpipe Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        CheckL0CNZTemplate<T>();
        CheckL0CNZTemplate<U>();
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        auto params = Std::make_tuple();
        return params;
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenFixpipeQuantParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();
        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst);
        uint64_t quantPre = static_cast<uint64_t>(QuantMode_t::NoQuant);
        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(
            nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
        return params;
    }
};


class FixpipeNZ2NZVectorBase : public CopyMatrixCcToGmBase, public CopyDeqTensorToFbuf {
public:
    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait>
    __aicore__ inline void FixpipeNZ2NZVectorEntrance(const T& dst, const U& src, const V& quant, const S& params)
    {
        FixpipeNZ2NZVectorImpl<T, U, V, S, trait>(dst, src, quant, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait, size_t... Is>
    __aicore__ inline void FixpipeNZ2NZVectorImpl(
        const T& dst, const U& src, const V& quant, const S& tupleParams, Std::index_sequence<Is...>)
    {
        FixpipeNZ2NZVectorCompute<T, U, V, trait>(dst, src, quant, Std::get<Is>(tupleParams)...);
    }

    template <typename T, typename U, const FixpipeTrait& trait, bool isTail>
    __aicore__ inline auto GenParams(const T& dst, const U& src)
    {
        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        if constexpr (isTail) {
            nSize = nSize % MAIN_LOOP_N_SIZE;
        } else {
            if (nSize > MAIN_LOOP_N_SIZE) {
                nSize = MAIN_LOOP_N_SIZE;
            }
        }
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(
            nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
        return params;
    }

    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void FixpipeNZ2NZVectorCompute(const T& dst, const U& src, const V& quant, uint32_t nIterNum,
        uint32_t calNSize, uint32_t tailNSize, uint32_t dstOffset, uint32_t srcOffset)
    {
        auto mainLoopParam = GenParams<T, U, trait, false>(dst, src);
        for (uint16_t i = 0; i < nIterNum; ++i) {
            CopyDeqTensorToFbufImpl(quant, calNSize, i);
            InsertPipeFix();
            DataCopy<T, U, decltype(mainLoopParam), trait>(dst[dstOffset * i], src[srcOffset * i], mainLoopParam);
        }
        auto tailParam = GenParams<T, U, trait, true>(dst, src);
        if (tailNSize) {
            CopyDeqTensorToFbufImpl(quant, tailNSize, nIterNum);
            InsertPipeFix();
            DataCopy<T, U, decltype(tailParam), trait>(dst[dstOffset * nIterNum], src[srcOffset * nIterNum], tailParam);
        }
    }
};

class FixpipeNZ2NZVectorQuant : public FixpipeNZ2NZVectorBase {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        auto params = GenParams<T, U, trait>(dst, src);
        FixpipeNZ2NZVectorEntrance<T, U, V, decltype(params), trait>(dst, src, quant, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::LiteType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Shape->Column->ZeroDim, is not Std::Int<16> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Stride->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "Fixpipe Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        CheckL0CNZTemplate<T>();
        CheckL0CNZTemplate<U>();
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();
        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);

        uint16_t nIterNum = 1;
        uint32_t calNSize = nSize;
        uint32_t tailNSize = 0;
        uint32_t dstOffset = CBURST_NUM * dstStride;
        uint32_t srcOffset = CBURST_NUM * srcStride * BLOCK_CUBE;
        if (calNSize > MAIN_LOOP_N_SIZE) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE;
            tailNSize = nSize % MAIN_LOOP_N_SIZE;
            calNSize = MAIN_LOOP_N_SIZE;
        }
        auto params = Std::make_tuple(nIterNum, calNSize, tailNSize, dstOffset, srcOffset);
        return params;
    }
};

}  // namespace TileInternal
}  // namespace AscendC

#endif  // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_NZ2NZ_FOUR_DIM_3101_L0C_GM_H
