/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_tensor_tile_fixpipe_quant_four_dim_3101_l0c_gm.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_FOUR_DIM_3101_L0C_GM_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_FOUR_DIM_3101_L0C_GM_H

#include "kernel_tensor_tile_fixpipe_common.h"

namespace AscendC {
namespace TileInternal {
constexpr uint32_t MAIN_LOOP_N_SIZE = 512;
constexpr uint32_t CBURST_NUM = MAIN_LOOP_N_SIZE / BLOCK_CUBE;

enum class Format : uint8_t { None, NZ, ND, DN };
enum class QuantMode : uint8_t { None, Scalar, Vector, Direct };

template <typename T>
__aicore__ inline constexpr Format GetDataFormat()
{
    using traitType = GetTensorTraitType<T>;
    if constexpr (IsL0cNZFormat<traitType>::value) {
        return Format::NZ;
    } else if constexpr (IsNDFormat<traitType>::value) {
        return Format::ND;
    } else if constexpr (IsDNFormat<traitType>::value) {
        return Format::DN;
    }
    return Format::None;
}

template <const FixpipeTrait& trait>
__aicore__ inline constexpr QuantMode GetQuantMode()
{
    if constexpr (IsVectorQuantMode<trait.quantPre>()) {
        return QuantMode::Vector;
    } else if constexpr (IsScalarQuantMode<trait.quantPre>()) {
        return QuantMode::Scalar;
    } else if constexpr (IsDirectQuantMode<trait.quantPre>()) {
        return QuantMode::Direct;
    }
    return QuantMode::None;
}

class CopyDeqTensorToFbuf {
public:
    template <typename T>
    __aicore__ inline void CopyDeqTensorToFbufImpl(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        auto params = CopyDeqTensorToFbufGenParams(src, calNSize, nIterIndex);
        uint64_t dstAddr = AllocTempBuf(calNSize);
        DataCopyImpl<T, decltype(params)>(dstAddr, src, params, tuple_sequence<decltype(params)>{});
        SetFpc(dstAddr);
        FreeTempBuf(dstAddr);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>,
            "CopyCbufToFB Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
            "CopyCbufToFB Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>,
            "CopyCbufToFB Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
            "CopyCbufToFB Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>,
            "CopyCbufToFB Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }
    template <typename T>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename T::LiteType;
        CheckNDTemplate<T>();
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3101
        static_assert(Std::is_one_of_v<srcType, uint64_t>, "The source data type is not supported.");
#endif
    }

    template <typename T>
    __aicore__ inline auto CopyDeqTensorToFbufGenParams(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        CheckTemplate<GetTensorTraitType<T>>();
        constexpr uint16_t fbufBurstLenUnit = 64;
        using srcType = typename GetTensorTraitType<T>::LiteType;
        auto layout = src.GetTensorTrait().GetLayout();
        uint16_t colLength = GetEleFromLayout<decltype(layout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layout);
        uint16_t rowStride = GetEleFromLayout<decltype(layout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(layout);
        uint16_t blockCount = CeilDivision(calNSize, colLength);
        uint16_t blockLen = CeilDivision(colLength * sizeof(srcType), fbufBurstLenUnit);
        uint16_t srcStride = CeilDivision(rowStride * sizeof(srcType), C0_SIZE);
        uint16_t dstStride = blockLen;
        uint32_t deqValueOffset = MAIN_LOOP_N_SIZE / colLength * rowStride * nIterIndex;

        auto params = Std::make_tuple(blockCount, blockLen, srcStride, dstStride, deqValueOffset);
        return params;
    }

    template <typename T, typename U, size_t... Is>
    __aicore__ inline void DataCopyImpl(
        const uint64_t& dstAddr, const T& src, const U& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename GetTensorTraitType<T>::LiteType;
        CopyCbufToFbuf<srcType, decltype(tupleParams)>(
            dstAddr, (__cbuf__ uint64_t *)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
    }

    template <typename T, typename U>
    __aicore__ inline void CopyCbufToFbuf(uint64_t dst, __cbuf__ T *src, uint16_t blockCount,
        uint16_t blockLen, uint16_t srcStride, uint16_t dstStride, uint32_t deqValueOffset)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            copy_cbuf_to_fbuf((__fbuf__ uint64_t *)dst, src + deqValueOffset, blockCount, blockLen, srcStride, dstStride);
        }
    }
};


// ####################### Quant Scalar/Direct Below ##############################

class FixpipeNZ2NZSimpleQuant : CopyMatrixCcToGmBase, SetRegisterBase {
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
class FixpipeNZ2NDSimpleQuant : public CopyMatrixCcToGmBase, SetRegisterBase {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<V, decltype(registerParams)>(quant, registerParams);
        auto params = GenFixpipeQuantParams<T, U, trait>(dst, src);
        DataCopy<T, U, decltype(params), trait>(dst, src, params);
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

    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>,
            "Fixpipe Src Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
            "Fixpipe Src Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>,
            "Fixpipe Src Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
            "Fixpipe Src Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>,
            "Fixpipe Src Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        CheckNDTemplate<T>();
        CheckL0CNZTemplate<U>();
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNDStride = 0;
        uint32_t dstNDStride = 0;
        auto params = Std::make_tuple(ndNum, dstNDStride, srcNDStride);
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
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst);
        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(
            nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
        return params;
    }
};

class FixpipeNZ2DNSimpleQuant : public CopyMatrixCcToGmBase, SetRegisterBase {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<V, decltype(registerParams)>(quant, registerParams);
        auto params = GenFixpipeQuantParams<T, U, trait>(dst, src);
        DataCopy<T, U, decltype(params), trait>(dst, src, params);
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

    template <typename T>
    __aicore__ inline constexpr void CheckDNTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>,
            "Fixpipe Src->Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
            "Fixpipe Src->Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>,
            "Fixpipe Src->Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>,
            "Fixpipe Src->Layout->Stride->Row->OneDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
            "Fixpipe Src->Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        CheckDNTemplate<T>();
        CheckL0CNZTemplate<U>();
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t dnNum = 1;
        uint32_t srcNZMatrixStride = 0;
        uint32_t dstDNMatrixStride = 0;
        uint32_t srcNZC0Stride = 1;
        auto params = Std::make_tuple(dnNum, dstDNMatrixStride, srcNZMatrixStride, srcNZC0Stride);
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
        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        bool nz2dnEn = true;
        auto params = Std::make_tuple(
            nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
        return params;
    }
};

// ############################### Vecotr Quant Below #####################################################

class FixpipeNZ2NZVectorBase : public CopyMatrixCcToGmBase, CopyDeqTensorToFbuf {
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
    template <typename T>
    __aicore__ inline constexpr void CheckDNTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>,
            "Fixpipe Src->Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
            "Fixpipe Src->Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>,
            "Fixpipe Src->Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>,
            "Fixpipe Src->Layout->Stride->Row->OneDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
            "Fixpipe Src->Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
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

class FixpipeNZ2NDVectorBase : public CopyMatrixCcToGmBase, CopyDeqTensorToFbuf {
public:
    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait>
    __aicore__ inline void FixpipeNZ2NDVectorEntrance(const T& dst, const U& src, const V& quant, const S& params)
    {
        FixpipeNZ2NDVectorImpl<T, U, V, S, trait>(dst, src, quant, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait, size_t... Is>
    __aicore__ inline void FixpipeNZ2NDVectorImpl(
        const T& dst, const U& src, const V& quant, const S& tupleParams, Std::index_sequence<Is...>)
    {
        FixpipeNZ2NDVectorCompute<T, U, V, trait>(dst, src, quant, Std::get<Is>(tupleParams)...);
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
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(
            nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
        return params;
    }

    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void FixpipeNZ2NDVectorCompute(const T& dst, const U& src, const V& quant, uint32_t nIterNum,
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

class FixpipeNZ2NDVectorQuant : public FixpipeNZ2NDVectorBase, SetRegisterBase {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<decltype(registerParams)>(registerParams);
        auto params = GenParams<T, U, trait>(dst, src);
        FixpipeNZ2NDVectorEntrance<T, U, V, decltype(params), trait>(dst, src, quant, params);
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
    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>,
            "Fixpipe Src Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
            "Fixpipe Src Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>,
            "Fixpipe Src Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
            "Fixpipe Src Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>,
            "Fixpipe Src Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        CheckNDTemplate<T>();
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
        uint32_t dstOffset = MAIN_LOOP_N_SIZE;
        uint32_t srcOffset = CBURST_NUM * srcStride * BLOCK_CUBE;
        if (calNSize > MAIN_LOOP_N_SIZE) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE;
            tailNSize = nSize % MAIN_LOOP_N_SIZE;
            calNSize = MAIN_LOOP_N_SIZE;
        }
        auto params = Std::make_tuple(nIterNum, calNSize, tailNSize, dstOffset, srcOffset);
        return params;
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t ndNum = 1;
        uint32_t srcNDStride = 0;
        uint32_t dstNDStride = 0;
        auto params = Std::make_tuple(ndNum, dstNDStride, srcNDStride);
        return params;
    }
};

class FixpipeNZ2DNVectorBase : public CopyMatrixCcToGmBase, CopyDeqTensorToFbuf {
public:
    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait>
    __aicore__ inline void FixpipeNZ2DNVectorEntrance(const T& dst, const U& src, const V& quant, const S& params)
    {
        FixpipeNZ2DNVectorImpl<T, U, V, S, trait>(dst, src, quant, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait, size_t... Is>
    __aicore__ inline void FixpipeNZ2DNVectorImpl(
        const T& dst, const U& src, const V& quant, const S& tupleParams, Std::index_sequence<Is...>)
    {
        FixpipeNZ2DNVectorCompute<T, U, V, trait>(dst, src, quant, Std::get<Is>(tupleParams)...);
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
        bool nz2dnEn = true;
        auto params = Std::make_tuple(
            nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
        return params;
    }

    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void FixpipeNZ2DNVectorCompute(const T& dst, const U& src, const V& quant, uint32_t nIterNum,
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

class FixpipeNZ2DNVectorQuant : public FixpipeNZ2DNVectorBase, SetRegisterBase {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<decltype(registerParams)>(registerParams);
        auto params = GenParams<T, U, trait>(dst, src);
        FixpipeNZ2DNVectorEntrance<T, U, V, decltype(params), trait>(dst, src, quant, params);
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
    template <typename T>
    __aicore__ inline constexpr void CheckDNTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>,
            "Fixpipe Src->Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
            "Fixpipe Src->Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>,
            "Fixpipe Src->Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>,
            "Fixpipe Src->Layout->Stride->Row->OneDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
            "Fixpipe Src->Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        CheckDNTemplate<T>();
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
        uint32_t dstOffset = MAIN_LOOP_N_SIZE * dstStride;
        uint32_t srcOffset = CBURST_NUM * srcStride * BLOCK_CUBE;
        if (calNSize > MAIN_LOOP_N_SIZE) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE;
            tailNSize = nSize % MAIN_LOOP_N_SIZE;
            calNSize = MAIN_LOOP_N_SIZE;
        }
        auto params = Std::make_tuple(nIterNum, calNSize, tailNSize, dstOffset, srcOffset);
        return params;
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenRegisterParams(const T& dst, const U& src)
    {
        uint32_t dnNum = 1;
        uint32_t srcNZMatrixStride = 0;
        uint32_t dstDNMatrixStride = 0;
        uint32_t srcNZC0Stride = 1;
        auto params = Std::make_tuple(dnNum, dstDNMatrixStride, srcNZMatrixStride, srcNZC0Stride);
        return params;
    }
};

class FormatRegistorIgnore {
public:
    template <typename T, typename U, typename V, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant) {}
};

template <Format dstFormat, Format srcFormat, QuantMode quantMode>
struct FormatRegistor {
    using type = FormatRegistorIgnore;
};

template <>
struct FormatRegistor<Format::NZ, Format::NZ, QuantMode::Direct> {
    using type = FixpipeNZ2NZSimpleQuant;
};

template <>
struct FormatRegistor<Format::ND, Format::NZ, QuantMode::Direct> {
    using type = FixpipeNZ2NDSimpleQuant;
};

template <>
struct FormatRegistor<Format::DN, Format::NZ, QuantMode::Direct> {
    using type = FixpipeNZ2DNSimpleQuant;
};

template <>
struct FormatRegistor<Format::NZ, Format::NZ, QuantMode::Scalar> {
    using type = FixpipeNZ2NZSimpleQuant;
};

template <>
struct FormatRegistor<Format::ND, Format::NZ, QuantMode::Scalar> {
    using type = FixpipeNZ2NDSimpleQuant;
};

template <>
struct FormatRegistor<Format::DN, Format::NZ, QuantMode::Scalar> {
    using type = FixpipeNZ2DNSimpleQuant;
};

template <>
struct FormatRegistor<Format::NZ, Format::NZ, QuantMode::Vector> {
    using type = FixpipeNZ2NZVectorQuant;
};

template <>
struct FormatRegistor<Format::ND, Format::NZ, QuantMode::Vector> {
    using type = FixpipeNZ2NDVectorQuant;
};

template <>
struct FormatRegistor<Format::DN, Format::NZ, QuantMode::Vector> {
    using type = FixpipeNZ2DNVectorQuant;
};

class FixpipeQuantFourDim3101L0C2GM {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        using FixpipeQuantL0C2GM =
            typename FormatRegistor<GetDataFormat<T>(), GetDataFormat<U>(), GetQuantMode<trait>()>::type;
        FixpipeQuantL0C2GM{}.template Run<T, U, V, trait>(dst, src, quant);
    }
};
}  // namespace TileInternal
}  // namespace AscendC

#endif  // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_QUANT_FOUR_DIM_3101_L0C_GM_H
