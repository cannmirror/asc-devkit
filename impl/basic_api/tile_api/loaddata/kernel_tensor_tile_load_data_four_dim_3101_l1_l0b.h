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
 * \file kernel_tensor_tile_load_data_four_dim_3101_l1_l0b.h
 * \brief
 */
#ifndef IMPL_TILE_API_KERNEL_TENSOR_TILE_LOAD_DATA_FOUR_DIM_3101_L1_L0B_H
#define IMPL_TILE_API_KERNEL_TENSOR_TILE_LOAD_DATA_FOUR_DIM_3101_L1_L0B_H

#include "../kernel_tensor_tile_utils.h"

namespace AscendC {
namespace TileInternal {

class LoadDataFourDim3101L12L0B {
public:
    template <typename T, typename U, const LoadDataTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src) {
        auto params = GenLoadDataParams<T, U, trait>(dst, src);
        LoadDataAlignV2Impl<T, U, decltype(params), trait>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template<typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::LiteType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "LoadDataFourDim3101L12L0B Layout->Shape->Row->ZeroDim is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim3101L12L0B Layout->Shape->Column->ZeroDim is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim3101L12L0B Layout->Stride->Row-ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "LoadDataFourDim3101L12L0B Layout->Stride->Column->ZeroDim is not Std::Int<1> type!");
    }

    template<typename T>
    __aicore__ inline constexpr void CheckZNTemplate()
    {
        using type = typename T::LiteType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
            "LoadDataFourDim3101L12L0B Layout->Shape->Column->ZeroDim is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim3101L12L0B Layout->Shape->Row->ZeroDim is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<C0_SIZE / sizeof(type)>>,
            "LoadDataFourDim3101L12L0B Layout->Stride->Column-ZeroDim is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideRow0, Std::Int<1>>,
            "LoadDataFourDim3101L12L0B Layout->Stride->Row->ZeroDim is not Std::Int<1> type!");
    }

    template <typename T, typename U, const LoadDataTrait& trait>
    __aicore__ inline constexpr void CheckTemplate()
    {
        using srcType = typename U::LiteType;
        using dstType = typename T::LiteType;
        static_assert(Std::is_same_v<srcType, dstType>, "The source data and target data have inconsistent data types.");
        CheckZNTemplate<T>();
        CheckNZTemplate<U>();
#if defined(__NPU_ARCH__ ) && __NPU_ARCH__ == 3101
        static_assert(Std::is_one_of_v<srcType, half, bfloat16_t, uint32_t, int32_t, float, uint8_t, int8_t,
            fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t>, "The source data type is not supported.");
#endif
    }

    template <typename T, typename U, const LoadDataTrait& trait>
    __aicore__ inline auto GenLoadDataParams(const T& dst, const U& src)
    {
        CheckTemplate<GetTensorTraitType<T>, GetTensorTraitType<U>, trait>();

        using DstType = typename GetTensorTraitType<T>::LiteType;
        auto dstLayout = dst.GetLayout();
        auto srcLayout = src.GetLayout();

        // offset
        uint16_t mStartPosition = 0;
        uint16_t kStartPosition = 0;
        auto mStep = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto kStep = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        // Nz -> Nz
        uint32_t STRIDE_UNIT = FRACTAL_FIXED * (C0_SIZE / sizeof(DstType));
        auto srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / STRIDE_UNIT;
        auto dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout) / STRIDE_UNIT;
        auto params = Std::make_tuple(mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
        return params;
    }

    template <typename T, typename U, typename V, const LoadDataTrait& trait, size_t... Is>
    __aicore__ inline void LoadDataAlignV2Impl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        using srcType = typename GetTensorTraitType<U>::LiteType;
        // MTE2
        LoadL1ToL0BAlignV2<srcType, !trait.transposed>((__cb__ srcType*)dst.GetPhyAddr(), (__cbuf__ srcType*)src.GetPhyAddr(),
            Std::get<Is>(tupleParams)...);
    }

    template <typename T, bool transpose>
    __aicore__ inline void LoadL1ToL0BAlignV2(__cb__ T* dst, __cbuf__ T* src, uint16_t mStartPosition,
        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride, uint16_t dstStride)
    {
        if (ASCEND_IS_AIV) {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3101) {
            load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, transpose);
        }
    }
};

} // namespace ACTE
} // namespace AscendC

#endif // IMPL_TILE_API_KERNEL_TENSOR_TILE_LOAD_DATA_FOUR_DIM_3101_L1_L0B_H