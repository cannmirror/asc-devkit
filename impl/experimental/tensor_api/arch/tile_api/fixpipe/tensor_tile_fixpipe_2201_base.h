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
 * \file tensor_tile_fixpipe_2201_base.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_FIXPIPE_TENSOR_TILE_FIXPIPE_2201_BASE_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_FIXPIPE_TENSOR_TILE_FIXPIPE_2201_BASE_H

#include "impl/experimental/tensor_api/utils/tensor_tile_utils.h"
#include "impl/experimental/tensor_api/struct/tensor_tile_struct.h"

namespace AscendC {
namespace TileInternal {
constexpr uint32_t MAIN_LOOP_N_SIZE_2201 = 512;
constexpr uint32_t CBURST_NUM_2201 = MAIN_LOOP_N_SIZE_2201 / BLOCK_CUBE;

template <typename T>
__aicore__ inline auto AllocTempBuf(const T& calNSize)
{
    uint64_t deqTensorTempBuf = 0;
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        deqTensorTempBuf = reinterpret_cast<uint64_t>(get_imm(0));
    }
    return deqTensorTempBuf;
}

template <typename T>
__aicore__ inline void FreeTempBuf(const T& deqTensorTempBuf)
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        (void)(deqTensorTempBuf);
    }
}

template <typename T>
__aicore__ inline void SetFpc(const T& deqTensorTempBuf)
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        uint64_t deqTensorAddr = (deqTensorTempBuf >> static_cast<uint64_t>(7)) << 8;
        set_fpc(deqTensorAddr);
    }
}

 __aicore__ inline void InsertSync()
{
 	if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101 || 
 	              CURRENT_ARCH_VERSION == ArchVersion::V2201) {
 	    pipe_barrier(PIPE_FIX);
 	}
}

class Copy2201DeqTensorToFbuf {
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
        using srcType = typename T::elementType;
        CheckNDTemplate<T>();
        constexpr Hardware srcTPos = TileInternal::GetHardPos<T>();
        static_assert(srcTPos == Hardware::L1, "The hardware of quant must be L1");
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<srcType, __cbuf__ uint64_t>, "The source data type is not supported.");
#endif
    }

    template <typename T>
    __aicore__ inline auto CopyDeqTensorToFbufGenParams(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        CheckTemplate<T>();
        constexpr uint16_t fbufBurstLenUnit = 128;
        using srcType = typename T::elementType;
        auto layout = src.Layout();
        uint16_t colLength = GetEleFromLayout<decltype(layout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(layout);
        uint16_t rowStride = GetEleFromLayout<decltype(layout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(layout);
        uint16_t blockCount = CeilDivision(calNSize, colLength);
        uint16_t blockLen = CeilDivision(colLength * sizeof(srcType), fbufBurstLenUnit);
        uint16_t srcStride = CeilDivision(rowStride * sizeof(srcType), C0_SIZE);
        uint16_t dstStride = blockLen;
        uint32_t deqValueOffset = MAIN_LOOP_N_SIZE_2201 / colLength * rowStride * nIterIndex;

        auto params = Std::make_tuple(blockCount, blockLen, srcStride, dstStride, deqValueOffset);
        return params;
    }

    template <typename T, typename U, size_t... Is>
    __aicore__ inline void DataCopyImpl(
        const uint64_t& dstAddr, const T& src, const U& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename T::elementType;
        CopyCbufToFbuf<srcType, decltype(tupleParams)>(
            dstAddr, (__cbuf__ uint64_t *)src.Engine().Begin().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T, typename U>
    __aicore__ inline void CopyCbufToFbuf(uint64_t dst, __cbuf__ T *src, uint16_t blockCount,
        uint16_t blockLen, uint16_t srcStride, uint16_t dstStride, uint32_t deqValueOffset)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            copy_cbuf_to_fbuf((__fbuf__ uint64_t *)dst, src + deqValueOffset, blockCount, blockLen, srcStride, dstStride);
        }
    }
};

class Copy2201MatrixCcToGmBase {
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
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        CopyMatrixCcToGm<trait.quantPre, dstType, srcType>(
            (__gm__ dstType *)dst.Engine().Begin().Get(), (__cc__ srcType *)src.Engine().Begin().Get(), Std::get<Is>(tupleParams)...);
    }

    template <QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline void CopyMatrixCcToGm(__gm__ T *dst, __cc__ U *src, uint32_t nSize, uint32_t mSize,
        uint32_t srcStride, uint32_t dstStride, bool reluEn, uint8_t unitFlag, bool isChannelSplit, bool nz2ndEn)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
#if defined(ASCENDC_CPU_DEBUG)
            copy_matrix_cc_to_gm(dst, src, 0, nSize, mSize, dstStride, srcStride, unitFlag,
                quantPre, reluEn, isChannelSplit, nz2ndEn);
#else
            copy_matrix_cc_to_gm(dst, src, 0, nSize, mSize, dstStride, srcStride, unitFlag,
                static_cast<uint64_t>(quantPre), reluEn, isChannelSplit, nz2ndEn);
#endif
        }
    }
};

class SetRegister2201Base {
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
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            set_quant_pre(quant);
        }
    }

    template <typename T>
    __aicore__ inline void SetParamsToRegister(uint64_t ndNum, uint64_t dstNDStride, uint64_t srcNDStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            T ndPara = 0;
            ndPara = ndPara | (static_cast<T>(ndNum));
            ndPara = ndPara | (static_cast<T>(srcNDStride) << 16);
            ndPara = ndPara | (static_cast<T>(dstNDStride) << 32);
            set_nd_para(ndPara);
        }
    }
};

}
}

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ARCH_TILE_API_FIXPIPE_TENSOR_TILE_FIXPIPE_2201_BASE_H