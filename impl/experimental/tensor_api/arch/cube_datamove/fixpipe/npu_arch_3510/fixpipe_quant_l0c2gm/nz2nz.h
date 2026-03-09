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
 * \file nz2nz.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2GM_NZ2NZ_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2GM_NZ2NZ_H

#include "impl/experimental/tensor_api/arch/cube_datamove/fixpipe/fixpipe_utils.h"
#include "impl/experimental/tensor_api/arch/cube_datamove/fixpipe/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class Fixpipe2GmNZ2NZSimpleQuant3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Coord& coord)
    {
        SetRegisterImpl<V>(quant);
        DataCopyImpl<trait, T, U>(dst, src);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        FormatCheckUtils3510 formatCheckInst;
        formatCheckInst.CheckNZTemplate<T>();
        formatCheckInst.CheckL0CNZTemplate<U>();
    }

    template <typename V>
    __aicore__ inline void SetRegisterImpl(const V& quant)
    {
        SetRegisterBase3510 setRegisterInst;
        setRegisterInst.SetRegister<V>(quant);
    }

    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src)
    {
        CheckTemplate<trait, T, U>();
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst.Data().Get());
        uint64_t quantPre = static_cast<uint64_t>(QuantMode_t::NoQuant);
        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        bool nz2dnEn = false;
        CopyMatrixCcToGmBase3510 copyInst;
        copyInst.DataCopy<trait, T, U>(dst, src,
            nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
    }
};


class Fixpipe2GmNZ2NZVectorBase3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename... Params>
    __aicore__ inline void FixpipeNZ2NZVectorEntrance(const T& dst, const U& src, const V& quant, const Params& ...params)
    {
        FixpipeNZ2NZVectorCompute<trait, T, U, V>(dst, src, quant, params...);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, bool isTail>
    __aicore__ inline auto GenParams(const T& dst, const U& src)
    {
        auto dstLayout = dst.Layout();
        auto srcLayout = src.Layout();
        uint32_t nSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
        if constexpr (isTail) {
            nSize = nSize % MAIN_LOOP_N_SIZE_3510;
        } else {
            if (nSize > MAIN_LOOP_N_SIZE_3510) {
                nSize = MAIN_LOOP_N_SIZE_3510;
            }
        }
        uint32_t mSize = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout) *
                         GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
        uint32_t srcStride =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout) / FRACTAL_FIXED;
        uint32_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout);
        uint8_t cacheMode = GetCacheModeFromTensor(dst.Data().Get());

        bool reluEn = false;
        uint8_t unitFlag = 0;
        bool isChannelSplit = false;
        bool nz2ndEn = false;
        bool nz2dnEn = false;
        auto params = Std::make_tuple(
            nSize, mSize, srcStride, dstStride, cacheMode, reluEn, unitFlag, isChannelSplit, nz2ndEn, nz2dnEn);
        return params;
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename V>
    __aicore__ inline void FixpipeNZ2NZVectorCompute(const T& dst, const U& src, const V& quant, uint32_t nIterNum,
        uint32_t calNSize, uint32_t tailNSize)
    {
        auto mainLoopParam = GenParams<trait, T, U, false>(dst, src);
        CopyMatrixCcToGmBase3510 copyInst;
        CopyDeqTensorToFbuf3510 copyDeqTensorInst;
        for (uint16_t i = 0; i < nIterNum; ++i) {
            copyDeqTensorInst.CopyDeqTensorToFbufImpl(quant, calNSize, i);
            InsertSync();
            auto srcCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, i * CBURST_NUM_3510));
            auto dstCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, i * CBURST_NUM_3510));
            DataCopyWrapper<trait>(copyInst, dst(dstCoord), src(srcCoord),
                mainLoopParam, tuple_sequence<decltype(mainLoopParam)>{});
        }
        if (tailNSize) {
            auto tailParam = GenParams<trait, T, U, true>(dst, src);
            copyDeqTensorInst.CopyDeqTensorToFbufImpl(quant, tailNSize, nIterNum);
            InsertSync();
            auto srcCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterNum * CBURST_NUM_3510));
            auto dstCoord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterNum * CBURST_NUM_3510));
            DataCopyWrapper<trait>(copyInst, dst(dstCoord), src(srcCoord),
                tailParam, tuple_sequence<decltype(tailParam)>{});
        }
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyWrapper(const CopyMatrixCcToGmBase3510& copyInst, const T& dst, const U& src,
        const V& tupleParams, Std::index_sequence<Is...>)
    {
        copyInst.DataCopy<trait>(dst, src, Std::get<Is>(tupleParams)...);
    }
};

class Fixpipe2GmNZ2NZVectorQuant3510 : public Fixpipe2GmNZ2NZVectorBase3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Coord& coord)
    {
        DataCopyImpl<trait, T, U, V>(dst, src, quant);
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U>
    __aicore__ inline constexpr void CheckTemplate()
    {
        FormatCheckUtils3510 formatCheckInst;
        formatCheckInst.CheckNZTemplate<T>();
        formatCheckInst.CheckL0CNZTemplate<U>();
    }

    template <const FixpipeTrait& trait, typename T, typename U, typename V>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& quant)
    {
        CheckTemplate<trait, T, U>();
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
        if (calNSize > MAIN_LOOP_N_SIZE_3510) {
            nIterNum = nSize / MAIN_LOOP_N_SIZE_3510;
            tailNSize = nSize % MAIN_LOOP_N_SIZE_3510;
            calNSize = MAIN_LOOP_N_SIZE_3510;
        }
        FixpipeNZ2NZVectorEntrance<trait, T, U, V>(dst, src, quant, nIterNum, calNSize, tailNSize);
    }
};


}  // namespace Te
}  // namespace AscendC

#endif  // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_QUANT_L0C2GM_NZ2NZ_H
