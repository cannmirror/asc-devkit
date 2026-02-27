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
 * \file fixpipe_base.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_BASE_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_BASE_H

#include "impl/experimental/tensor_api/arch/cube_datamove/fixpipe/fixpipe_utils.h"
#include "include/experimental/tensor_api/utils/utils.h"
#include "impl/experimental/tensor_api/tensor/pointer_impl.h"
#include "impl/experimental/tensor_api/tensor/local_tensor_impl.h"

namespace AscendC {
namespace Te {
constexpr uint32_t MAIN_LOOP_N_SIZE_3510 = 512;
constexpr uint32_t CBURST_NUM_3510 = MAIN_LOOP_N_SIZE_3510 / BLOCK_CUBE;

class CopyDeqTensorToFbuf3510 {
public:
    template <typename T>
    __aicore__ inline void CopyDeqTensorToFbufImpl(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        auto dstAddr = reinterpret_cast<__fbuf__ uint64_t*>(AllocTempBuf(calNSize));
        auto dst = MakeTensor(MakeFixbufmemPtr(dstAddr), src.Layout());
        auto tileSrc = TileSrcTensor(src, calNSize, nIterIndex);
        DataCopy(dst, tileSrc);
        SetFpc(dstAddr);
    }
private:
    template <typename T>
    __aicore__ inline decltype(auto) TileSrcTensor(const T& src, uint16_t calNSize, uint16_t nIterIndex) {
        auto coord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterIndex * MAIN_LOOP_N_SIZE_3510));
        auto shape = MakeShape(1, calNSize);
        return src(coord, shape);
    }
};

class CopyMatrixCcToGmBase3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const V& params)
    {
        DataCopyImpl<trait, T, U, V>(dst, src, params, tuple_sequence<decltype(params)>{});
    }

private:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, size_t... Is>
    __aicore__ inline void DataCopyImpl(const T& dst, const U& src, const V& tupleParams, Std::index_sequence<Is...>)
    {
        CopyMatrixCcToGm<trait.quantPre>(dst.Data().Get(), src.Data().Get(), Std::get<Is>(tupleParams)...);
    }

    template <QuantMode_t quantPre, typename T, typename U>
    __aicore__ inline void CopyMatrixCcToGm(__gm__ T *dst, __cc__ U *src, uint32_t nSize, uint32_t mSize,
        uint32_t srcStride, uint32_t dstStride, uint8_t cacheMode, bool reluEn, uint8_t unitFlag, bool isChannelSplit,
        bool nz2ndEn, bool nz2dnEn)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            copy_matrix_cc_to_gm(dst, src, 0, nSize, mSize, dstStride, srcStride, cacheMode, 0, unitFlag, static_cast<uint64_t>(quantPre),
                reluEn, isChannelSplit, nz2ndEn, static_cast<uint64_t>(QuantMode_post::NoConv), 0, false, false, 0, false, false, false, false, false, 
                nz2dnEn); 
        }
    }
};

class SetRegisterBase3510 {
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
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            set_quant_pre(quant);
        }
    }

    template <typename T>
    __aicore__ inline void SetParamsToRegister(uint32_t ndNum, uint32_t dstNDStride, uint32_t srcNDStride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
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
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
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

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_3510_FIXPIPE_BASE_H