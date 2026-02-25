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
 * \file data_copy_gm2l1.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "impl/experimental/tensor_api/tensor/pointer_impl.h"
#include "impl/experimental/tensor_api/tensor/local_tensor_impl.h"

namespace AscendC {
namespace Te {

template <typename T>
__aicore__ inline void SetMTE2NzPara(const T& para) {
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
        set_mte2_nz_para(para);
    }
}

class CopyGmToCbufAlignV2Base {
public:
    template <const DataCopyTrait& trait, typename T, typename U, typename V>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const V& tupleParams) {
        DataCopyImpl<trait, T, U, V>(dst, src, tupleParams, tuple_sequence<V>{});
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename U::elementType;
        if constexpr (sizeof(srcType) == sizeof(uint32_t)) {
            CopyGmToCbufAlignV2(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(uint16_t)) {
            CopyGmToCbufAlignV2(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(uint8_t)) {
            CopyGmToCbufAlignV2(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        }
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufAlignV2(__cbuf__ T* dst, __gm__ T* src, uint32_t blockCount, uint32_t blockLen, 
        uint8_t leftPaddingCnt, uint8_t rigntPaddingCnt, uint8_t cacheMode, int64_t srcStride, int64_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            copy_gm_to_cbuf_align_v2(dst, src, 0, blockCount, blockLen, leftPaddingCnt, rigntPaddingCnt, true,
                cacheMode, srcStride, dstStride);
        }
    }
};

class CopyGmToCbufAlignV2NZBase : public CopyGmToCbufAlignV2Base {
public:
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {

        auto params = GenDataCopyParams<trait, T, U>(dst, src);
        CopyGmToCbufAlignV2Base::DataCopy<trait, T, U, decltype(params)>(dst, src, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>, "Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>, "Layout->Shape->Column->ZeroDim, is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>, "Layout->Stride->Row->ZeroDim, is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>, "Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        CheckNZTemplate<T>();
        CheckNZTemplate<U>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, 
            Std::tuple<__cbuf__ half, __gm__ half>, Std::tuple<__cbuf__ float, __gm__ float>, 
            Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, 
            Std::tuple<__cbuf__ int8_t, __gm__ int8_t>, Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>, 
            Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>, Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>>,
            "The data type is not supported.");
#endif
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto smallFractalSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
        auto bigFractalSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout)
            * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        auto srcStrideSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
        auto dstStrideSize = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        uint8_t leftPaddingCnt = 0;
        uint8_t rigntPaddingCnt = 0;
        uint8_t cacheMode = GetCacheModeFromTensor(src.Data().Get());

        using type = typename U::elementType;

        auto blockCount = bigFractalSize;
        auto blockLen = smallFractalSize * sizeof(type);
        auto srcStride = srcStrideSize * sizeof(type);
        auto dstStride = dstStrideSize * sizeof(type);

        return Std::make_tuple(blockCount, blockLen, leftPaddingCnt, rigntPaddingCnt, cacheMode, srcStride, dstStride);
    }
};

class CopyGmToCbufAlignV2NDBase : public CopyGmToCbufAlignV2Base {
public:
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenDataCopyParams<trait, T, U>(dst, src);
        CopyGmToCbufAlignV2Base::DataCopy<trait, T, U, decltype(params)>(dst, src, params);
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

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        CheckNDTemplate<T>();
        CheckNDTemplate<U>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, 
            Std::tuple<__cbuf__ half, __gm__ half>, Std::tuple<__cbuf__ float, __gm__ float>, 
            Std::tuple<__cbuf__ int32_t, __gm__ int32_t>>, "The data type is not supported.");
#endif
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
 
        auto dstShapeRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto dstShapeColumns = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        auto dstStrideRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        auto srcStrideRows = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
 
        uint8_t leftPaddingCnt = 0;
        uint8_t rigntPaddingCnt = 0;
        uint8_t cacheMode = GetCacheModeFromTensor(src.Data().Get());
 
        using type = typename U::elementType;
 
        auto blockCount = dstShapeRows;
        auto blockLen = dstShapeColumns * sizeof(type);
        auto srcStride = srcStrideRows * sizeof(type);
        auto dstStride = dstStrideRows * sizeof(type);
 
        return Std::make_tuple(blockCount, blockLen, leftPaddingCnt, rigntPaddingCnt, cacheMode, srcStride, dstStride);
    }
};

class CopyGmToCbufMultiDn2nzBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenDataCopyParams<trait, T, U>(dst, src);
        DataCopyImpl<trait, T, U, decltype(params)>(dst, src, params, tuple_sequence<decltype(params)>{});
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
        using type = typename T::elementType;
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

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        
        CheckDNTemplate<U>();
        CheckNZTemplate<T>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, 
            Std::tuple<__cbuf__ half, __gm__ half>, Std::tuple<__cbuf__ float, __gm__ float>, 
            Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, 
            Std::tuple<__cbuf__ int8_t, __gm__ int8_t>, Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>, 
            Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>, Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>>,
            "The data type is not supported.");
#endif
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

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
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(type) / C0_SIZE);

        uint8_t cacheMode = GetCacheModeFromTensor(src.Data().Get());
        return Std::make_tuple(dnNum, loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode,
            nValue, dValue, loop4SrcStride, false);
    }

    template <const DataCopyTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename U::elementType;
        if constexpr(sizeof(srcType) == sizeof(int8_t)) {
            CopyGmToCbufMultiDn2nz(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(half)) {
            CopyGmToCbufMultiDn2nz(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(float)) {
            CopyGmToCbufMultiDn2nz(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        }
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufMultiDn2nz(__cbuf__ T* dst, __gm__ T* src, uint16_t dnNum, uint16_t loop2DstStride, 
        uint16_t loop3DstStride, uint16_t loop4DstStride, uint64_t loop1SrcStride, uint8_t cacheMode, uint16_t nValue, 
        uint32_t dValue, uint64_t loop4SrcStride, bool enableSmallC0)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
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
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto shape = MakeShape(
            GetEleFromLayout<decltype(dst.Layout()), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dst.Layout()) *
            GetEleFromLayout<decltype(dst.Layout()), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dst.Layout()),
            GetEleFromLayout<decltype(dst.Layout()), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dst.Layout()) *
            GetEleFromLayout<decltype(dst.Layout()), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dst.Layout())
            );
        auto sliceTensor = src(coord, shape);
        auto params = GenDataCopyParams<trait>(dst, sliceTensor);
        DataCopyImpl<trait>(dst, sliceTensor, params, tuple_sequence<decltype(params)>{});
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
        using type = typename T::elementType;
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

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        CheckNDTemplate<U>();
        CheckNZTemplate<T>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3510
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, 
            Std::tuple<__cbuf__ half, __gm__ half>, Std::tuple<__cbuf__ float, __gm__ float>, 
            Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, 
            Std::tuple<__cbuf__ int8_t, __gm__ int8_t>, Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>, 
            Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>, Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>>,
            "The data type is not supported.");
#endif
    }

    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        uint16_t ndNum = 1;
        uint16_t nValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t dValue = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint64_t srcNdMatrixStride = 0;

        auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
        uint64_t srcDValue = srcRowStride;
        uint16_t dstNzC0Stride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout)
            * sizeof(type) / C0_SIZE;
        uint16_t dstNzNStride = 1;
        uint32_t dstNzMatrixStride = 0;

        uint64_t loop1SrcStride = srcDValue * sizeof(type);
        uint64_t loop4SrcStride = srcNdMatrixStride * sizeof(type);

        uint16_t loop2DstStride = dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
        uint16_t loop3DstStride = dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
        // loop4_dst_stride: dst_nz_matrix_stride * size_of_dst_type / C0_size
        uint16_t loop4DstStride = static_cast<uint16_t>(dstNzMatrixStride * sizeof(type) / C0_SIZE);

        uint8_t cacheMode = GetCacheModeFromTensor(src.Data().Get());
        return Std::make_tuple(ndNum, loop2DstStride, loop3DstStride, loop4DstStride, loop1SrcStride, cacheMode,
            nValue, dValue, loop4SrcStride, false);
    }

    template <const DataCopyTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename U::elementType;
        if constexpr(sizeof(srcType) == sizeof(int8_t)) {
            CopyGmToCbufMultiNd2nz(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(half)) {
            CopyGmToCbufMultiNd2nz(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        } else if constexpr (sizeof(srcType) == sizeof(float)) {
            CopyGmToCbufMultiNd2nz(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
        }
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src, uint16_t ndNum, uint16_t loop2DstStride, 
        uint16_t loop3DstStride, uint16_t loop4DstStride, uint64_t loop1SrcStride, uint8_t cacheMode, uint16_t nValue, 
        uint32_t dValue, uint64_t loop4SrcStride, bool enableSmallC0)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
            mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
            mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
            mte2NzPara |= static_cast<uint64_t>(ndNum);            // MTE2_NZ_PARA[15:0]
            SetMTE2NzPara(mte2NzPara);   // CCE: store parameters for ND2NZ DMA instructions
            copy_gm_to_cbuf_multi_nd2nz(dst, src, 0, loop1SrcStride, cacheMode, nValue, dValue, loop4SrcStride, enableSmallC0);
        }
    }
};

class DataCopyFourDim3510GM2L1 : public CopyGmToCbufMultiNd2nzBase, public CopyGmToCbufMultiDn2nzBase,
    public CopyGmToCbufAlignV2NZBase, public CopyGmToCbufAlignV2NDBase {
public:
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto fourDimSrc = PreProcess(src);
        Execute<trait>(dst, fourDimSrc, coord);
    }

private:
    template <const DataCopyTrait& trait, typename T, typename U, typename Coord>
    __aicore__ inline void Execute(const T& dst, const U& src, const Coord& coord) {
        if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufAlignV2NZBase::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsNDFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufMultiNd2nzBase::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsDNFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufMultiDn2nzBase::Run<trait, T, U, Coord>(dst, src, coord);
        } else if constexpr (IsNDFormat<U>::value && IsNDFormat<T>::value) {
            CopyGmToCbufAlignV2NDBase::Run<trait, T, U, Coord>(dst, src, coord);
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2L1_H