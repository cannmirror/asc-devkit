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
 * \file load_data_four_dim_2201_l1_l0a.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_FOUR_DIM_2201_L1_L0A_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_FOUR_DIM_2201_L1_L0A_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "include/experimental/tensor_api/tensor/make.h"

namespace AscendC {
namespace TensorInternal {

class LoadDataFourDim2201L12L0ABase{
public:
    template <typename T, typename U, const LoadDataTrait& trait, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        return;
    }

    template<typename T>
    __aicore__ inline constexpr void CheckZNTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Shape->Row->ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
            "LoadDataFourDim2201L12L0A Layout->Shape->Column->ZeroDim is not Std::Int<16> type!"); 

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Row-ZeroDim is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Column->ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<FRACTAL_FIXED * C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Column->OneDim is not Std::Int<16*C0Size/Type> type!");
    }

    template<typename T>
    __aicore__ inline constexpr void CheckZZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "LoadDataFourDim2201L12L0A Layout->Shape->Row->ZeroDim is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Shape->Column->ZeroDim is not Std::Int<C0Size/Type> type!"); 

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Row-ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Column->ZeroDim is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<FRACTAL_FIXED * C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Column->OneDim is not Std::Int<16*C0Size/Type> type!");
    }

    template<typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "LoadDataFourDim2201L12L0A Layout->Shape->Row->ZeroDim is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Shape->Column->ZeroDim is not Std::Int<C0Size/Type> type!"); 

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Row-ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<FRACTAL_FIXED * C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Row-OneDim is not Std::Int<16*C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "LoadDataFourDim2201L12L0A Layout->Stride->Column->ZeroDim is not Std::Int<1> type!");
    }

    template <typename T, typename U, const LoadDataTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;

        if constexpr (!trait.transposed) {
            CheckNZTemplate<U>();
        } else {
            CheckZNTemplate<U>();
        }
        CheckZZTemplate<T>();
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 2201
        static_assert(Std::is_one_of_v<Std::tuple<dstType, srcType>, 
            Std::tuple<__ca__ bfloat16_t, __cbuf__ bfloat16_t>, Std::tuple<__ca__ half, __cbuf__ half>, 
            Std::tuple<__ca__ float, __cbuf__ float>, Std::tuple<__ca__ int16_t, __cbuf__ int16_t>, 
            Std::tuple<__ca__ int32_t, __cbuf__ int32_t>, Std::tuple<__ca__ int8_t, __cbuf__ int8_t>,
            Std::tuple<__ca__ uint16_t, __cbuf__ uint16_t>, Std::tuple<__ca__ uint32_t, __cbuf__ uint32_t>, 
            Std::tuple<__ca__ uint8_t, __cbuf__ uint8_t>>, "The data type is not supported.");
#endif
    }

    __aicore__ inline void SetMatrixL0AImpl(uint64_t config)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            set_fmatrix(config);
        }
    }

    template <typename T>
    __aicore__ inline void LoadData3DV2L12L0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t kExtension, uint16_t mExtension,
        uint16_t kStartPt, uint16_t mStartPt, bool enTranspose, uint16_t channelSize)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            img2colv2_cbuf_to_ca(dst, src, kExtension, mExtension, kStartPt, mStartPt, 1, 1, 1, 1, 1, 1, false, false, enTranspose, false, channelSize);
        }
    }

    template<typename T>
    __aicore__ inline void LoadDataTransposeL12L0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t startIdx, uint16_t repeatTime, uint16_t srcStride,
        uint16_t dstStride, uint16_t dstFracStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V2201) {
            load_cbuf_to_ca_transpose(dst, src, startIdx, repeatTime, srcStride, dstStride, false, dstFracStride);
        }
    }
};

class LoadDataFourDim2201L12L0ANZ2ZZ : public LoadDataFourDim2201L12L0ABase {
public:
    template <typename T, typename U, const LoadDataTrait& trait, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<T, U, trait, Coord>(dst, src, coord);
        LoadDataL0AImpl<T, U, decltype(params), trait>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, typename U, const LoadDataTrait& trait, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<T, U, trait>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * 16;
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * 32 / sizeof(DstType);
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * 16;
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * 32 / sizeof(DstType);
        auto indexRow = Std::get<1>(Std::get<0>(coord)) * 16;
        auto indexCol = Std::get<1>(Std::get<1>(coord)) * 32 / sizeof(DstType);
        // NZ
        auto config = srcRow | (1u << 16);
        auto params = Std::make_tuple(dstRow, dstCol, srcRow, srcCol, indexRow, indexCol, config);
        return params;
    }

    template <typename T, typename U, typename V, const LoadDataTrait& trait, size_t... Is>
    __aicore__ inline void LoadDataL0AImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0AImpl<trait.transposed>(dst.Engine().Begin().Get(), src.Engine().Begin().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template<bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t dstRow, uint16_t dstCol, uint16_t srcRow,
        uint16_t srcCol, uint16_t indexRow, uint16_t indexCol, uint64_t config)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr(needTranspose) {
            SetMatrixL0AImpl(config);
            LoadData3DV2L12L0AImpl(dst, src, dstRow, dstCol, indexRow, indexCol, true, srcRow);
        } else {
            SetMatrixL0AImpl(config);
            LoadData3DV2L12L0AImpl(dst, src, dstCol, dstRow, indexCol, indexRow, false, srcCol);
        }
    }
};

class LoadDataFourDim2201L12L0AZN2ZZB8 : public LoadDataFourDim2201L12L0ABase {
public:
    template <typename T, typename U, const LoadDataTrait& trait, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<T, U, trait, Coord>(dst, src, coord);
        LoadDataL0AImpl<T, U, decltype(params), trait>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, typename U, const LoadDataTrait& trait, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<T, U, trait>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * 16;
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * 32 / sizeof(DstType);
        // ZN
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * 32 / sizeof(DstType);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * 16;
        auto indexRow = Std::get<1>(Std::get<0>(coord)) * 32 / sizeof(DstType);
        auto indexCol = Std::get<1>(Std::get<1>(coord)) * 16;
        constexpr const uint32_t SHIFT_BLOCK_LEN = 4;
        constexpr const uint32_t SHIFT_BLOCK_BYTE = 5;
        constexpr uint16_t CUBE_BLOCK_SIZE = 512;
        constexpr uint16_t fracNum = 2;
        uint16_t srcColNum = srcCol >> (SHIFT_BLOCK_LEN + fracNum - 1);
        uint16_t dstColNum = dstCol * sizeof(DstType) >> SHIFT_BLOCK_BYTE;
        uint16_t dstRowNum = dstRow >> (SHIFT_BLOCK_LEN + fracNum - 1);
        uint16_t startIdx0 = (indexCol >> (SHIFT_BLOCK_LEN + fracNum - 1)) +
            (indexRow * srcColNum * sizeof(DstType) >> SHIFT_BLOCK_BYTE);
        auto params = Std::make_tuple(dstRowNum, dstColNum, fracNum, startIdx0, srcColNum);
        return params;
    }

    template <typename T, typename U, typename V, const LoadDataTrait& trait, size_t... Is>
    __aicore__ inline void LoadDataL0AImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0AImpl<trait.transposed>(dst.Engine().Begin().Get(), src.Engine().Begin().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template<bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t dstRowNum, uint16_t dstColNum,
        uint16_t fracNum, uint16_t startIdx0, uint16_t srcColNum)
    {
        if ASCEND_IS_AIV {
            return;
        }
        uint16_t dstGap = 0;
        uint16_t dstFracGap = 0;
        constexpr uint16_t CUBE_BLOCK_SIZE = 512;
        if (dstRowNum >= dstColNum) {
            dstGap = fracNum * dstColNum - 1;
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i++) {
                LoadDataTransposeL12L0AImpl(dst, src, startIdx0 + i, dstRowNum, srcColNum, dstGap, dstFracGap);
                dst += CUBE_BLOCK_SIZE;
            }
        } else {
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstRowNum; i++) {
                LoadDataTransposeL12L0AImpl(dst, src, startIdx0 + i * srcColNum, dstColNum, 1, 0, dstFracGap);
                dst += dstColNum * CUBE_BLOCK_SIZE * fracNum;
            }
        }
    }
};

class LoadDataFourDim2201L12L0AZN2ZZ : public LoadDataFourDim2201L12L0ABase {
public:
    template <typename T, typename U, const LoadDataTrait& trait, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        auto params = GenLoadDataParams<T, U, trait, Coord>(dst, src, coord);
        LoadDataL0AImpl<T, U, decltype(params), trait>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, typename U, const LoadDataTrait& trait, class Coord>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src, const Coord& coord)
    {
        CheckTemplate<T, U, trait>();
        using DstType = typename T::elementType;
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        auto dstRow = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout) * 16;
        auto dstCol = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout) * 32 / sizeof(DstType);
        auto srcRow = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * 32 / sizeof(DstType);
        auto srcCol = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout) * 16;
        auto indexRow = Std::get<1>(Std::get<0>(coord)) * 32 / sizeof(DstType);
        auto indexCol = Std::get<1>(Std::get<1>(coord)) * 16;
        auto config = srcCol | (1u << 16);
        auto params = Std::make_tuple(dstRow, dstCol, srcRow, srcCol, indexRow, indexCol, config);
        return params;
    }

    template <typename T, typename U, typename V, const LoadDataTrait& trait, size_t... Is>
    __aicore__ inline void LoadDataL0AImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        LoadL1ToL0AImpl<trait.transposed>(dst.Engine().Begin().Get(), src.Engine().Begin().Get(),
            Std::get<Is>(tupleParams)...);
    }

    template<bool needTranspose, typename T>
    __aicore__ inline void LoadL1ToL0AImpl(__ca__ T* dst, __cbuf__ T* src, uint16_t dstRow, uint16_t dstCol, uint16_t srcRow,
        uint16_t srcCol, uint16_t indexRow, uint16_t indexCol, uint64_t config)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr(needTranspose) {
            SetMatrixL0AImpl(config);
            LoadData3DV2L12L0AImpl(dst, src, dstRow, dstCol, indexRow, indexCol, true, srcRow);
        } else {
            SetMatrixL0AImpl(config);
            LoadData3DV2L12L0AImpl(dst, src, dstCol, dstRow, indexCol, indexRow, false, srcCol);
        }
    }
};

class LoadDataFourDim2201L12L0A : public LoadDataFourDim2201L12L0ANZ2ZZ, public LoadDataFourDim2201L12L0AZN2ZZB8,
    public LoadDataFourDim2201L12L0AZN2ZZ {
public:
    template <typename T, typename U, const LoadDataTrait& trait, class Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const Coord& coord) {
        if constexpr (IsNZFormat<U>::value && IsZZFormat<T>::value) {
            LoadDataFourDim2201L12L0ANZ2ZZ::Run<T, U, trait, Coord>(dst, src, coord);
        } else if constexpr (IsZNFormat<U>::value && IsZZFormat<T>::value && (sizeof(typename U::elementType) == 1)) {
            LoadDataFourDim2201L12L0AZN2ZZB8::Run<T, U, trait, Coord>(dst, src, coord);
        } else if constexpr (IsZNFormat<U>::value && IsZZFormat<T>::value) {
            LoadDataFourDim2201L12L0AZN2ZZ::Run<T, U, trait, Coord>(dst, src, coord);
        }
    }
};

} // namespace TensorInternal
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_LOAD_DATA_NPU_ARCH_2201_LOAD_DATA_FOUR_DIM_2201_L1_L0A_H