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
 * \file kernel_tensor_tile_data_copy_four_dim_3101_l1_bt.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_FOUR_DIM_3101_L1_BT_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_FOUR_DIM_3101_L1_BT_H

#include "kernel_tensor_tile_utils.h"

namespace AscendC {
namespace TileInternal {

class CopyCbufToBT {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenDataCopyParams<T, U, trait>(dst, src);
        DataCopyImpl<T, U, decltype(params), trait>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "CopyCbufToBT Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "CopyCbufToBT Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "CopyCbufToBT Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "CopyCbufToBT Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "CopyCbufToBT Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        static_assert(Std::is_same_v<srcType, dstType>, "The source data and target data have inconsistent data types.");

        CheckNDTemplate<T>();
        CheckNDTemplate<U>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3101
        static_assert(Std::is_one_of_v<srcType, half, bfloat16_t, float, int32_t>, "The source data type is not supported.");
#endif
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();

        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();

        uint16_t srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint16_t srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
        uint16_t dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        using srcType = typename GetTensorTraitType<U>::LiteType;
        using dstType = typename GetTensorTraitType<T>::LiteType;

        bool convControl = false;
        uint16_t blockCount = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint16_t blockLen = srcCol * sizeof(dstType) / ONE_BLK_SIZE;
        uint16_t srcStride = (srcRow - srcCol) * sizeof(srcType) / ONE_BLK_SIZE;
        uint16_t dstStride = (dstRow - srcCol) * sizeof(dstType) / ONE_BLK_SIZE;

        return Std::make_tuple(convControl, blockCount, blockLen, srcStride, dstStride);
    }

    template <typename T, typename U, typename V, const DataCopyTrait& trait, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename GetTensorTraitType<U>::LiteType;
        CopyCbufToBt((uint64_t)dst.GetPhyAddr(), (__cbuf__ srcType*)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
    }

    template <typename T>
    __aicore__ inline void CopyCbufToBt(uint64_t dst, __cbuf__ T* src, bool convControl, uint16_t blockCount, uint16_t blockLen,
        uint16_t srcStride, uint16_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            copy_cbuf_to_bt(dst, src, convControl, blockCount, blockLen, srcStride, dstStride);
        }
    }
};

} // namespace TileInternal
} // namespace AscendC

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_FOUR_DIM_3101_L1_BT_H