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
 * \file fixpipe_quant_nz2nd_four_dim_2201_l0c_gm.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_NZ2ND_FOUR_DIM_2201_L0C_GM_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_NZ2ND_FOUR_DIM_2201_L0C_GM_H

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/fixpipe/npu_arch_2201/fixpipe_2201_base.h"

namespace AscendC {
namespace TensorInternal {

class FixpipeNZ2ND2201SimpleQuant : public Copy2201MatrixCcToGmBase, public SetRegister2201Base {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<V, decltype(registerParams)>(quant, registerParams);
        auto params = GenFixpipeQuantParams<T, U, trait>(dst, src);
        DataCopy<T, U, decltype(params), trait>(dst, src, params);
    }

    template <typename T, typename U, typename V, const FixpipeTrait& trait, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Coord& coord)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<V, decltype(registerParams)>(quant, registerParams);
        auto params = GenFixpipeQuantParams<T, U, trait>(dst, src);
        auto dstNDLayout = dst.Layout();
        auto index = Crd2Idx(coord, dstNDLayout);
        using dstType = typename T::elementType;
        auto dstNew = reinterpret_cast<dstType *>(dst.Engine().Begin().Get() + index);
        uint32_t dstN = GetEleFromLayout<decltype(dstNDLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstNDLayout) *
                        GetEleFromLayout<decltype(dstNDLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstNDLayout);
        uint32_t dstM = GetEleFromLayout<decltype(dstNDLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstNDLayout) *
                        GetEleFromLayout<decltype(dstNDLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstNDLayout);
        auto dstNDIterator = MakeGMmemPtr(dstNew);
        auto dstNDMatrixLayout = MakeRowMajorLayout<dstType>(dstM, dstN);
        auto dstNDTensor = MakeTensor(dstNDIterator, dstNDMatrixLayout); 

        DataCopy<T, U, decltype(params), trait>(dstNDTensor, src, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::elementType;
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
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
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
        CheckTemplate<T, U, trait>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, reluEn, unitFlag, isChannelSplit, nz2ndEn);
        return params;
    }
};

class FixpipeNZ2ND2201VectorBase : public Copy2201MatrixCcToGmBase, public Copy2201DeqTensorToFbuf {
public:
    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait>
    __aicore__ inline void FixpipeNZ2NDVectorEntrance(const T& dst, const U& src, const V& quant, const S& params)
    {
        FixpipeNZ2NDVectorImpl<T, U, V, S, trait>(dst, src, quant, params, tuple_sequence<decltype(params)>{});
    }

    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait, typename Coord>
    __aicore__ inline void FixpipeNZ2NDVectorEntrance(const T& dst, const U& src, const V& quant, const Coord& coord, const S& params)
    {
        FixpipeNZ2NDVectorImpl<T, U, V, S, trait, Coord>(dst, src, quant, coord, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait, size_t... Is>
    __aicore__ inline void FixpipeNZ2NDVectorImpl(
        const T& dst, const U& src, const V& quant, const S& tupleParams, Std::index_sequence<Is...>)
    {
        FixpipeNZ2NDVectorCompute<T, U, V, trait>(dst, src, quant, Std::get<Is>(tupleParams)...);
    }

    template <typename T, typename U, typename V, typename S, const FixpipeTrait& trait, typename Coord, size_t... Is>
    __aicore__ inline void FixpipeNZ2NDVectorImpl(
        const T& dst, const U& src, const V& quant, const Coord& coord, const S& tupleParams, Std::index_sequence<Is...>)
    {
        FixpipeNZ2NDVectorCompute<T, U, V, trait, Coord>(dst, src, quant, coord, Std::get<Is>(tupleParams)...);
    }

    template <typename T, typename U, const FixpipeTrait& trait, bool isTail>
    __aicore__ inline auto GenParams(const T& dst, const U& src)
    {
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        if constexpr (isTail) {
            nSize = nSize % MAIN_LOOP_N_SIZE_2201;
        } else {
            if (nSize > MAIN_LOOP_N_SIZE_2201) {
                nSize = MAIN_LOOP_N_SIZE_2201;
            }
        }
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = true;
        auto params = Std::make_tuple(nSize, mSize, srcStride, dstStride, reluEn, unitFlag, isChannelSplit, nz2ndEn);
        return params;
    }

    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void FixpipeNZ2NDVectorCompute(const T& dst, const U& src, const V& quant, uint32_t nIterNum,
        uint32_t calNSize, uint32_t tailNSize, uint32_t dstOffset, uint32_t srcOffset)
    {
        auto mainLoopParam = GenParams<T, U, trait, false>(dst, src);
        auto srcLayout = src.Layout();
        using srcType = typename U::elementType;
        uint32_t srcN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                        GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t srcM = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                        GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto dstLayout = dst.Layout();
        using dstType = typename T::elementType;
        uint32_t dstN = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
                        GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        uint32_t dstM = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) *
                        GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);

        for (uint16_t i = 0; i < nIterNum; ++i) {
            CopyDeqTensorToFbufImpl(quant, calNSize, i);
            InsertSync();
            auto srcNew = reinterpret_cast<srcType *>(src.Engine().Begin().Get() + srcOffset * i);
            auto srcIterator = MakeL0CmemPtr(srcNew);
            auto srcMatrixLayout = MakeNZLayout<Std::ignore_t>(srcM, srcN);
            auto srcTensor = MakeTensor(srcIterator, srcMatrixLayout);
            auto dstNew = reinterpret_cast<dstType *>(dst.Engine().Begin().Get() + dstOffset * i);
            auto dstIterator = MakeGMmemPtr(dstNew);
            auto dstMatrixLayout = MakeRowMajorLayout<dstType>(dstM, dstN);
            auto dstTensor = MakeTensor(dstIterator, dstMatrixLayout);

            DataCopy<T, U, decltype(mainLoopParam), trait>(dstTensor, srcTensor, mainLoopParam);
        }
        auto tailParam = GenParams<T, U, trait, true>(dst, src);
        if (tailNSize) {
            CopyDeqTensorToFbufImpl(quant, tailNSize, nIterNum);
            InsertSync();
            auto srcNew = reinterpret_cast<srcType *>(src.Engine().Begin().Get() + srcOffset * nIterNum);
            auto srcIterator = MakeL0CmemPtr(srcNew);
            auto srcMatrixLayout = MakeNZLayout<Std::ignore_t>(srcM, srcN);
            auto srcTensor = MakeTensor(srcIterator, srcMatrixLayout);
            auto dstNew = reinterpret_cast<dstType *>(dst.Engine().Begin().Get() + dstOffset * nIterNum);
            auto dstIterator = MakeGMmemPtr(dstNew);
            auto dstMatrixLayout = MakeRowMajorLayout<dstType>(dstM, dstN);
            auto dstTensor = MakeTensor(dstIterator, dstMatrixLayout);

            DataCopy<T, U, decltype(tailParam), trait>(dstTensor, srcTensor, tailParam);
        }
    }

    template <typename T, typename U, typename V, const FixpipeTrait& trait, typename Coord>
    __aicore__ inline void FixpipeNZ2NDVectorCompute(const T& dst, const U& src, const V& quant, const Coord& coord,
        uint32_t nIterNum, uint32_t calNSize, uint32_t tailNSize, uint32_t dstOffset, uint32_t srcOffset)
    {
        auto mainLoopParam = GenParams<T, U, trait, false>(dst, src);
        auto srcLayout = src.Layout();
        using srcType = typename U::elementType;
        uint32_t srcN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                        GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t srcM = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                        GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        auto dstLayout = dst.Layout();
        using dstType = typename T::elementType;
        uint32_t dstN = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout) *
                        GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
        uint32_t dstM = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout) *
                        GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
        auto index = Crd2Idx(coord, dstLayout);

        for (uint16_t i = 0; i < nIterNum; ++i) {
            CopyDeqTensorToFbufImpl(quant, calNSize, i);
            InsertSync();
            auto srcNew = reinterpret_cast<srcType *>(src.Engine().Begin().Get() + srcOffset * i);
            auto srcIterator = MakeL0CmemPtr(srcNew);
            auto srcMatrixLayout = MakeNZLayout<Std::ignore_t>(srcM, srcN);
            auto srcTensor = MakeTensor(srcIterator, srcMatrixLayout);
            auto dstNew = reinterpret_cast<dstType *>(dst.Engine().Begin().Get() + dstOffset * i + index);
            auto dstNDIterator = MakeGMmemPtr(dstNew);
            auto dstNDMatrixLayout = MakeRowMajorLayout<dstType>(dstM, dstN);
            auto dstNDTensor = MakeTensor(dstNDIterator, dstNDMatrixLayout);

            DataCopy<T, U, decltype(mainLoopParam), trait>(dstNDTensor, srcTensor, mainLoopParam);
        }
        auto tailParam = GenParams<T, U, trait, true>(dst, src);
        if (tailNSize) {
            CopyDeqTensorToFbufImpl(quant, tailNSize, nIterNum);
            InsertSync();
            auto srcNew = reinterpret_cast<srcType *>(src.Engine().Begin().Get() + srcOffset * nIterNum);
            auto srcIterator = MakeL0CmemPtr(srcNew);
            auto srcMatrixLayout = MakeNZLayout<Std::ignore_t>(srcM, srcN);
            auto srcTensor = MakeTensor(srcIterator, srcMatrixLayout);
            auto dstNew = reinterpret_cast<dstType *>(dst.Engine().Begin().Get() + dstOffset * nIterNum + index);
            auto dstIterator = MakeGMmemPtr(dstNew);
            auto dstMatrixLayout = MakeRowMajorLayout<dstType>(dstM, dstN);
            auto dstTensor = MakeTensor(dstIterator, dstMatrixLayout);

            DataCopy<T, U, decltype(tailParam), trait>(dstTensor, srcTensor, tailParam);
        }
    }
};

class FixpipeNZ2ND2201VectorQuant : public FixpipeNZ2ND2201VectorBase, public SetRegister2201Base {
public:
    template <typename T, typename U, typename V, const FixpipeTrait& trait>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<decltype(registerParams)>(registerParams);
        auto params = GenParams<T, U, trait>(dst, src);
        FixpipeNZ2NDVectorEntrance<T, U, V, decltype(params), trait>(dst, src, quant, params);
    }

    template <typename T, typename U, typename V, const FixpipeTrait& trait, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Coord& coord)
    {
        auto registerParams = GenRegisterParams<T, U, trait>(dst, src);
        SetRegister<decltype(registerParams)>(registerParams);
        auto params = GenParams<T, U, trait>(dst, src);
        FixpipeNZ2NDVectorEntrance<T, U, V, decltype(params), trait, Coord>(dst, src, quant, coord, params);
    }

private:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::elementType;
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
        using srcType = typename U::elementType;
        using dstType = typename T::elementType;
        CheckNDTemplate<T>();
        CheckL0CNZTemplate<U>();
    }

    template <typename T, typename U, const FixpipeTrait& trait>
    __aicore__ inline auto GenParams(const T& dst, const U& src)
    {
        CheckTemplate<T, U, trait>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);

        uint16_t nIterNum = 1;
        uint32_t calNSize = nSize;
        uint32_t tailNSize = 0;
        uint32_t dstOffset = MAIN_LOOP_N_SIZE_2201;
        uint32_t srcOffset = CBURST_NUM_2201 * srcStride * BLOCK_CUBE;
        if (calNSize > MAIN_LOOP_N_SIZE_2201) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE_2201;
            tailNSize = nSize % MAIN_LOOP_N_SIZE_2201;
            calNSize = MAIN_LOOP_N_SIZE_2201;
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

}  // namespace TensorInternal
}  // namespace AscendC

#endif  // EXPERIMENTAL_TENSOR_API_DETAIL_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_2201_FIXPIPE_QUANT_NZ2ND_FOUR_DIM_2201_L0C_GM_H
