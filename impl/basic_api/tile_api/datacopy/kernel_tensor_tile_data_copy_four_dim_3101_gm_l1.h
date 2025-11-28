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
 * \file kernel_tensor_tile_data_copy_four_dim_3101_gm_l1.h
 * \brief
 */
#ifndef IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_FOUR_DIM_3101_GM_L1_H
#define IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_FOUR_DIM_3101_GM_L1_H

#include "../kernel_tensor_tile_utils.h"

namespace AscendC {
namespace TileInternal {

template <typename T>
__aicore__ inline void SetMTE2NzPara(const T& para) {
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
        set_mte2_nz_para(para);
    }
}

class CopyGmToCbufAlignV2Base {
public:
    template <typename T, typename U, typename V, const DataCopyTrait& trait>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const V& tupleParams) {
        DataCopyImpl<T, U, V, trait>(dst, src, tupleParams, tuple_sequence<V>{});
    }

private:
    template <typename T, typename U, typename V, const DataCopyTrait& trait, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename GetTensorTraitType<U>::LiteType;
        if constexpr (sizeof(srcType) == sizeof(uint32_t)) {
            CopyGmToCbufAlignV2((__cbuf__ uint32_t*)dst.GetPhyAddr(), (__gm__ uint32_t*)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(uint16_t)) {
            CopyGmToCbufAlignV2((__cbuf__ uint16_t*)dst.GetPhyAddr(), (__gm__ uint16_t*)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(uint8_t)) {
            CopyGmToCbufAlignV2((__cbuf__ uint8_t*)dst.GetPhyAddr(), (__gm__ uint8_t*)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        }
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufAlignV2(__cbuf__ T* dst, __gm__ T* src, uint32_t blockCount, uint32_t blockLen,
        uint8_t leftPaddingCnt, uint8_t rigntPaddingCnt, uint8_t cacheMode, int64_t srcStride, int64_t dstStride)
    {
        if (ASCEND_IS_AIV) {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            copy_gm_to_cbuf_align_v2(dst, src, 0, blockCount, blockLen, leftPaddingCnt, rigntPaddingCnt, true,
                cacheMode, srcStride, dstStride);
        }
    }
};

class CopyGmToCbufAlignV2NZBase : public CopyGmToCbufAlignV2Base {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {

        auto params = GenDataCopyParams<T, U, trait>(dst, src);
        CopyGmToCbufAlignV2Base::DataCopy<T, U, decltype(params), trait>(dst, src, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::LiteType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>, "Layout->Shape->Column->ZeroDim, is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>, "Layout->Stride->Row->ZeroDim, is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        static_assert(Std::is_same_v<srcType, dstType>, "The source data and target data have inconsistent data types.");

        CheckNZTemplate<T>();
        CheckNZTemplate<U>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3101
        static_assert(Std::is_one_of_v<srcType, bfloat16_t, half, float, int16_t, int32_t, int8_t, uint16_t, uint32_t, uint8_t>,
            "The source data type is not supported.");
#endif
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();

        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();

        auto smallFractalSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
        auto bigFractalSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        auto srcStrideSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
        auto dstStrideSize = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        uint8_t leftPaddingCnt = 0;
        uint8_t rigntPaddingCnt = 0;
        uint8_t cacheMode = GetCacheModeFromTensor(src);

        using type = typename GetTensorTraitType<U>::LiteType;

        auto blockCount = bigFractalSize;
        auto blockLen = smallFractalSize * sizeof(type);
        auto srcStride = srcStrideSize * sizeof(type);
        auto dstStride = dstStrideSize * sizeof(type);

        return Std::make_tuple(blockCount, blockLen, leftPaddingCnt, rigntPaddingCnt, cacheMode, srcStride, dstStride);
    }
};

class CopyGmToCbufAlignV2NDBase : public CopyGmToCbufAlignV2Base {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenDataCopyParams<T, U, trait>(dst, src);
        CopyGmToCbufAlignV2Base::DataCopy<T, U, decltype(params), trait>(dst, src, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
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

        auto dstShapeRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto dstShapeColumns = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        auto dstStrideRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        auto srcStrideRows = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);

        uint8_t leftPaddingCnt = 0;
        uint8_t rigntPaddingCnt = 0;
        uint8_t cacheMode = GetCacheModeFromTensor(src);

        using type = typename GetTensorTraitType<U>::LiteType;

        auto blockCount = dstShapeRows;
        auto blockLen = dstShapeColumns * sizeof(type);
        auto srcStride = srcStrideRows * sizeof(type);
        auto dstStride = dstStrideRows * sizeof(type);

        return Std::make_tuple(blockCount, blockLen, leftPaddingCnt, rigntPaddingCnt, cacheMode, srcStride, dstStride);
    }
};

class CopyGmToCbufMultiDn2nzBase {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenDataCopyParams<T, U, trait>(dst, src);
        DataCopyImpl<T, U, decltype(params), trait>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckDNTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Src->Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Src->Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Src->Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>, "Src->Layout->Stride->Row->OneDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Src->Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
    }

    template <typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::LiteType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Dst->Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>, "Dst->Layout->Shape->Column->ZeroDim, is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>, "Dst->Layout->Stride->Row->ZeroDim, is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED>>,
            "Dst->Layout->Stride->Row->OneDim, is not Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Dst->Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        static_assert(Std::is_same_v<srcType, dstType>, "The source data and target data have inconsistent data types.");

        CheckDNTemplate<U>();
        CheckNZTemplate<T>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3101
        static_assert(Std::is_one_of_v<srcType, bfloat16_t, half, float, int16_t, int32_t, int8_t, uint16_t, uint32_t, uint8_t>,
            "The source data type is not supported.");
#endif
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();

        using type = typename GetTensorTraitType<U>::LiteType;
        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();

        uint16_t dnNum = 1;
        uint16_t nValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint64_t srcDnMatrixStride = 0;
        uint64_t srcDValue = nValue;
        uint16_t dstNzC0Stride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout)
            * sizeof(type) / C0_SIZE;
        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = 0;

        uint64_t loop1SrcStride = srcDValue * sizeof(type);
        uint64_t loop4SrcStride = srcDnMatrixStride * sizeof(type);

        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_size
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(type) / ONE_BLK_SIZE);

        uint8_t cacheMode = GetCacheModeFromTensor(src);
        return Std::make_tuple(dnNum, loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode,
            nValue, dValue, loop4SrcStride, false);
    }

    template <typename T, typename U, typename V, const DataCopyTrait& trait, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename GetTensorTraitType<U>::LiteType;
        if constexpr(sizeof(srcType) == sizeof(int8_t)) {
            CopyGmToCbufMultiDn2nz((__cbuf__ int8_t *)dst.GetPhyAddr(), (__gm__ int8_t *)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(half)) {
            CopyGmToCbufMultiDn2nz((__cbuf__ half *)dst.GetPhyAddr(), (__gm__ half *)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(float)) {
            CopyGmToCbufMultiDn2nz((__cbuf__ float *)dst.GetPhyAddr(), (__gm__ float *)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        }
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufMultiDn2nz(__cbuf__ T* dst, __gm__ T* src, uint16_t dnNum, uint16_t loop2DstStride,
        uint16_t loop3DstStride, uint16_t loop4DstStride, uint64_t loop1SrcStride, uint8_t cacheMode, uint16_t nValue,
        uint32_t dValue, uint64_t loop4SrcStride, bool enableSmallC0)
    {
        if (ASCEND_IS_AIV) {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
            mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
            mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
            mte2NzPara |= static_cast<uint64_t>(dnNum);            // MTE2_NZ_PARA[15:0]
            SetMTE2NzPara(mte2NzPara);   // CCE: store parameters for DN2NZ DMA instructions
            copy_gm_to_cbuf_multi_dn2nz(dst, src, 0, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, enableSmallC0);
        }
    }
};

class CopyGmToCbufMultiNd2nzBase {
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
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>, "Src Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>, "Src Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>, "Src Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>, "Src Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>, "Src Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }

    template <typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::LiteType;
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
            "Dst Layout->Stride->Column->ZeroDim, is not Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED> type!");
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        static_assert(Std::is_same_v<srcType, dstType>, "The source data and target data have inconsistent data types.");

        CheckNDTemplate<U>();
        CheckNZTemplate<T>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3101
        static_assert(Std::is_one_of_v<srcType, bfloat16_t, half, float, int16_t, int32_t, int8_t, uint16_t, uint32_t, uint8_t>,
            "The source data type is not supported.");
#endif
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();

        using type = typename GetTensorTraitType<U>::LiteType;
        auto dstLayout = dst.GetTensorTrait().GetLayout();
        auto srcLayout = src.GetTensorTrait().GetLayout();

        uint16_t ndNum = 1;
        uint16_t nValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint64_t srcNdMatrixStride = 0;
        uint64_t srcDValue = dValue;
        uint16_t dstNzC0Stride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout)
            * sizeof(type) / C0_SIZE;
        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = 0;

        uint64_t loop1SrcStride = srcDValue * sizeof(type);
        uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(type);

        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_size
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(type) / ONE_BLK_SIZE);

        uint8_t cacheMode = GetCacheModeFromTensor(src);
        return Std::make_tuple(ndNum, loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode,
            nValue, dValue, loop4SrcStride, false);
    }

    template <typename T, typename U, typename V, const DataCopyTrait& trait, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename GetTensorTraitType<U>::LiteType;
        if constexpr(sizeof(srcType) == sizeof(int8_t)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ int8_t *)dst.GetPhyAddr(), (__gm__ int8_t *)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(half)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ half *)dst.GetPhyAddr(), (__gm__ half *)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(float)) {
            CopyGmToCbufMultiNd2nz((__cbuf__ float *)dst.GetPhyAddr(), (__gm__ float *)src.GetPhyAddr(), Std::get<Is>(tupleParams)...);
        }
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src, uint16_t ndNum, uint16_t loop2DstStride,
        uint16_t loop3DstStride, uint16_t loop4DstStride, uint64_t loop1SrcStride, uint8_t cacheMode, uint16_t nValue,
        uint32_t dValue, uint64_t loop4SrcStride, bool enableSmallC0)
    {
        if (ASCEND_IS_AIV) {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
            mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
            mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
            mte2NzPara |= static_cast<uint64_t>(ndNum);            // MTE2_NZ_PARA[15:0]
            SetMTE2NzPara(mte2NzPara);   // CCE: store parameters for ND2NZ DMA instructions
            copy_gm_to_cbuf_multi_nd2nz(dst, src, 0, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, enableSmallC0);
        }
    }
};

class DataCopyFourDim3101GM2L1 : public CopyGmToCbufMultiNd2nzBase, public CopyGmToCbufMultiDn2nzBase,
    public CopyGmToCbufAlignV2NZBase, public CopyGmToCbufAlignV2NDBase {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        using srcTraitType = GetTensorTraitType<U>;
        using dstTraitType = GetTensorTraitType<T>;
        if constexpr (IsNZFormat<srcTraitType>::value && IsNZFormat<dstTraitType>::value) {
            CopyGmToCbufAlignV2NZBase::Run<T, U, trait>(dst, src);
        } else if constexpr (IsNDFormat<srcTraitType>::value && IsNZFormat<dstTraitType>::value) {
            CopyGmToCbufMultiNd2nzBase::Run<T, U, trait>(dst, src);
        } else if constexpr (IsDNFormat<srcTraitType>::value && IsNZFormat<dstTraitType>::value) {
            CopyGmToCbufMultiDn2nzBase::Run<T, U, trait>(dst, src);
        } else if constexpr (IsNDFormat<srcTraitType>::value && IsNDFormat<dstTraitType>::value) {
            CopyGmToCbufAlignV2NDBase::Run<T, U, trait>(dst, src);
        }
    }
};

} // namespace TileInternal
} // namespace AscendC

#endif // IMPL_TENSOR_TILE_API_KERNEL_TENSOR_TILE_DATA_COPY_FOUR_DIM_3101_GM_L1_H