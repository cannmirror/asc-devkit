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
 * \file data_copy_four_dim_2201_gm_l1.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_FOUR_DIM_2201_GM_L1_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_FOUR_DIM_2201_GM_L1_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "include/experimental/tensor_api/tensor/make.h"

namespace AscendC {
namespace TensorInternal {
class CopyGmToCbufBase {
public:
    template <typename T, typename U, typename V, const DataCopyTrait& trait>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const V& tupleParams) {
        DataCopyImpl<T, U, V, trait>(dst, src, tupleParams, tuple_sequence<V>{});
    }

private:
    template <typename T, typename U, typename V, const DataCopyTrait& trait, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        CopyGmToCbuf(dst.Engine().Begin().Get(), src.Engine().Begin().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbuf(__cbuf__ T* dst, __gm__ T* src, uint32_t blockCount, uint32_t blockLen,
                                        int64_t srcStride, int64_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            copy_gm_to_cbuf(dst, src, 0, blockCount, blockLen, srcStride, dstStride, static_cast<pad_t>(0));
        }
    }
};

class CopyGmToCbufNZBase : public CopyGmToCbufBase {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {

        auto params = GenDataCopyParams<T, U, trait>(dst, src);
        CopyGmToCbufBase::DataCopy<T, U, decltype(params), trait>(dst, src, params);
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

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        CheckNZTemplate<T>();
        CheckNZTemplate<U>();

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, 
            Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, Std::tuple<__cbuf__ half, __gm__ half>, 
            Std::tuple<__cbuf__ float, __gm__ float>, Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, 
            Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, Std::tuple<__cbuf__ int8_t, __gm__ int8_t>,
            Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>, Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>, 
            Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>>, "The data type is not supported.");
#endif
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<T, U, trait>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto dstRow0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
        auto dstRow1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto dstCol0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
        auto dstCol1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

        auto srcStrideSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
        using type = typename U::elementType;

        auto blockCount = dstCol1;
        auto blockLen = dstRow1 * dstRow0 * dstCol0 * sizeof(type);
        auto srcStride = srcStrideSize * sizeof(type);
        auto dstStride = blockLen;
        return Std::make_tuple(blockCount, blockLen, srcStride, dstStride);
    }
};

class CopyGmToCbufNDBase : public CopyGmToCbufBase {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenDataCopyParams<T, U, trait>(dst, src);
        CopyGmToCbufBase::DataCopy<T, U, decltype(params), trait>(dst, src, params);
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
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        CheckNDTemplate<T>();
        CheckNDTemplate<U>();

#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, 
            Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, Std::tuple<__cbuf__ half, __gm__ half>, 
            Std::tuple<__cbuf__ float, __gm__ float>, Std::tuple<__cbuf__ int32_t, __gm__ int32_t>>, "The data type is not supported.");
#endif
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<T, U, trait>();

        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
 
        auto dstShapeRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto dstShapeColumns = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        auto dstStrideRows = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);
        auto srcStrideRows = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
 
        using type = typename U::elementType;
 
        // considering block len is encoded num.
        auto blockCount = dstShapeRows;
        auto blockLen = dstShapeColumns * sizeof(type);
        auto srcStride = srcStrideRows * sizeof(type);
        auto dstStride = dstStrideRows * sizeof(type);
 
        return Std::make_tuple(blockCount, blockLen, srcStride, dstStride);
    }
};

class CopyGmToCbufMultiND2NZBase {
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
            "Dst Layout->Stride->Row->OneDim, is not Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED> type!");
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        
        CheckNDTemplate<U>();
        CheckNZTemplate<T>();
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, 
            Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, Std::tuple<__cbuf__ half, __gm__ half>, 
            Std::tuple<__cbuf__ float, __gm__ float>, Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, 
            Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, Std::tuple<__cbuf__ int8_t, __gm__ int8_t>,
            Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>, Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>, 
            Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>>, "The data type is not supported.");
#endif
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<T, U, trait>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * 16;
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * 32 / sizeof(type);
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

        uint16_t ndNum = 1;
        uint64_t srcNdMatrixStride = 0;
        uint32_t dstNzMatrixStride = 0;
        uint16_t nValue = dstRow;
        uint16_t dValue = dstCol;
        uint16_t srcDValue = srcCol;
        uint16_t dstNzC0Stride = dstRow;
        uint16_t dstNzNStride = 1;
        return Std::make_tuple(ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
    }

    template <typename T, typename U, typename V, const DataCopyTrait& trait, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        CopyGmToCbufMultiNd2nz(dst.Engine().Begin().Get(), src.Engine().Begin().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src,
            uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue,
            uint16_t dstNzC0Stride, uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (sizeof(T) == 1) {
            copy_gm_to_cbuf_multi_nd2nz_b8(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
        } else if constexpr (sizeof(T) == 2) {
            copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
        } else if constexpr (sizeof(T) == 4) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
        }
    }
};

class CopyGmToCbufMultiDN2ZNBase {
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
    __aicore__ inline constexpr void CheckZNTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>, "Dst Layout->Shape->Column->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<C0_SIZE / sizeof(type)>>, "Dst Layout->Shape->Row->ZeroDim, is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<C0_SIZE / sizeof(type)>>, "Dst Layout->Stride->Column->ZeroDim, is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>, "Dst Layout->Stride->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED>>,
            "Dst Layout->Stride->Column->OneDim, is not Std::Int<C0_SIZE / sizeof(type) * FRACTAL_FIXED> type!");
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        
        CheckDNTemplate<U>();
        CheckZNTemplate<T>();
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, 
            Std::tuple<__cbuf__ bfloat16_t, __gm__ bfloat16_t>, Std::tuple<__cbuf__ half, __gm__ half>, 
            Std::tuple<__cbuf__ float, __gm__ float>, Std::tuple<__cbuf__ int16_t, __gm__ int16_t>, 
            Std::tuple<__cbuf__ int32_t, __gm__ int32_t>, Std::tuple<__cbuf__ int8_t, __gm__ int8_t>,
            Std::tuple<__cbuf__ uint16_t, __gm__ uint16_t>, Std::tuple<__cbuf__ uint32_t, __gm__ uint32_t>, 
            Std::tuple<__cbuf__ uint8_t, __gm__ uint8_t>>, "The data type is not supported.");
#endif
    }

    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline auto GenDataCopyParams(const T& dst, const U& src)
    {
        CheckTemplate<T, U, trait>();

        using type = typename U::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();

        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * 32 / sizeof(type);
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * 16;
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

        uint16_t ndNum = 1;
        uint64_t srcNdMatrixStride = 0;
        uint32_t dstNzMatrixStride = 0;
        uint16_t nValue = dstCol;
        uint16_t dValue = dstRow;
        uint16_t srcDValue = srcRow;
        uint16_t dstNzC0Stride = dstCol;
        uint16_t dstNzNStride = 1;

        return Std::make_tuple(ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
    }

    template <typename T, typename U, typename V, const DataCopyTrait& trait, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        CopyGmToCbufMultiNd2nz(dst.Engine().Begin().Get(), src.Engine().Begin().Get(), Std::get<Is>(tupleParams)...);
    }

    template <typename T>
    __aicore__ inline void CopyGmToCbufMultiNd2nz(__cbuf__ T* dst, __gm__ T* src,
            uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue,
            uint16_t dstNzC0Stride, uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (sizeof(T) == 1) {
            copy_gm_to_cbuf_multi_nd2nz_b8(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
        } else if constexpr (sizeof(T) == 2) {
            copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride,
                dstNzNStride, dstNzMatrixStride);
        } else if constexpr (sizeof(T) == 4) {
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
        }
    }
};

class DataCopyFourDim2201GM2L1 : public CopyGmToCbufMultiND2NZBase, public CopyGmToCbufMultiDN2ZNBase,
    public CopyGmToCbufNZBase, public CopyGmToCbufNDBase {
public:
    template <typename T, typename U, const DataCopyTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufNZBase::Run<T, U, trait>(dst, src);
        } else if constexpr (IsNDFormat<U>::value && IsNZFormat<T>::value) {
            CopyGmToCbufMultiND2NZBase::Run<T, U, trait>(dst, src);
        } else if constexpr (IsDNFormat<U>::value && IsZNFormat<T>::value) {
            CopyGmToCbufMultiDN2ZNBase::Run<T, U, trait>(dst, src);
        } else if constexpr (IsNDFormat<U>::value && IsNDFormat<T>::value) {
            CopyGmToCbufNDBase::Run<T, U, trait>(dst, src);
        }
    }
};

} // namespace TensorInternal
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_2201_DATA_COPY_FOUR_DIM_2201_GM_L1_H