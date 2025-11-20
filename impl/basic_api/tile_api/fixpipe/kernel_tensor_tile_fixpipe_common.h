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
 * \file kernel_tensor_tile_fixpipe_common.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_COMMON_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_COMMON_H

#include "../kernel_tensor_tile_utils.h"

namespace AscendC {
namespace TileInternal {
constexpr uint32_t MAIN_LOOP_N_SIZE = 512;
constexpr uint32_t CBURST_NUM = MAIN_LOOP_N_SIZE / BLOCK_CUBE;

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
        constexpr TPosition srcTPos = T::tPos;
        static_assert(srcTPos == TPosition::C1, "The logical position of quant must be C1");
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

class CopyMatrixCcToGmBase {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const V& params)
    {
        DataCopyImpl<T, U, V, trait>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, typename U, typename V, const FixpipeTrait& trait, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename GetTensorTraitType<U>::LiteType;
        using dstType = typename GetTensorTraitType<T>::LiteType;
        CopyMatrixCcToGm<trait.quantPre, dstType, srcType>(
            (__gm__ dstType *)dst.GetPhyAddr(), (__cc__ srcType *)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
    }

    template <QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline void CopyMatrixCcToGm(__gm__ T *dst, __cc__ U *src, uint32_t nSize, uint32_t mSize,
        uint32_t srcStride, uint32_t dstStride, uint8_t cacheMode, bool reluEn, uint8_t unitFlag, bool isChannelSplit,
        bool nz2ndEn, bool nz2dnEn)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            copy_matrix_cc_to_gm(dst, src, 0, nSize, mSize, dstStride, srcStride, cacheMode, 0, unitFlag, static_cast<uint64_t>(quantPre),
                reluEn, isChannelSplit, nz2ndEn, static_cast<uint64_t>(QuantMode_post::NoConv), 0, false, false, 0, false, false, false, false, false, 
                nz2dnEn); 
        }
    }
};

class SetRegisterBase {
public:
    template <typename T, typename U>
    __aicore__ inline void SetRegister(const T& quant, const U& params)
    {
        SetQuantPre(quant);
        SetRegisterImpl<U>(params, tuple_sequence<decltype(params)>{});
    }
    template <typename T>
    __aicore__ inline void SetRegister(const T& params)
    {
        SetRegisterImpl<T>(params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, size_t... Is>
    __aicore__ inline void SetRegisterImpl(const T& tupleParams, Std::index_sequence<Is...>)
    {
        if constexpr (sizeof...(Is) == 0) {
            return;
        } else {
            SetParamsToRegister<uint64_t>(Std::get<Is>(tupleParams)...);
        }
    }

    template <typename T>
    __aicore__ inline void SetQuantPre(const T& quant)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            set_quant_pre(quant);
        }
    }

    template <typename T>
    __aicore__ inline void SetParamsToRegister(uint32_t ndNum, uint32_t dstNDStride, uint32_t srcNDStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            T loop3Para = static_cast<T>(dstNDStride) << 32;
            loop3Para |= static_cast<T>(srcNDStride) << 16;
            loop3Para |= static_cast<T>(ndNum);
            set_loop3_para(loop3Para);
        }
    }

    template <typename T>
    __aicore__ inline void SetParamsToRegister(uint32_t dnNum, uint32_t dstDNStride, uint32_t srcNZMatrixStride, uint32_t srcNZC0Stride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            T loop3Para = static_cast<T>(dstDNStride) << 32;
            loop3Para |= static_cast<T>(srcNZMatrixStride) << 16;
            loop3Para |= static_cast<T>(dnNum);
            set_loop3_para(loop3Para);
            T channelPara = static_cast<T>(srcNZC0Stride) << 48;
            set_channel_para(channelPara);
        }
    }
};

}
}

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_FIXPIPE_COMMON_H