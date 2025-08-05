/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_out_split_n_with_params.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_TILE_COPY_OUI_COPY_OUT_SPLIT_N_WITH_PARAMS_H
#define ACT_INCLUDE_MATMUL_TILE_COPY_OUI_COPY_OUT_SPLIT_N_WITH_PARAMS_H

#include "../tile_copy_policy.h"

namespace Act {
namespace Gemm {
namespace Tile {
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyOutSplitNWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<AscendC::PhyPosIsUB(OutputType::pos) // output to UB
                                      && (OutputType::format == CubeFormat::ND ||
                                          OutputType::format == CubeFormat::ND_ALIGN) // ND
                                      && !AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T,
                                                                                typename InputType::T>()>> { // no quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& co2Local,
                                      const AscendC::LocalTensor<SrcT>& co1Local, int curRow, int curCol,
                                      int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                      int orgM, int orgN, int orgKc, uint8_t id = 0)
    {
#if defined(__DAV_C310__)
        uint32_t dimN = orgKc != 0 ? orgKc : orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            dimN = AscendC::Ceil(dimN, blockCount) * blockCount;
        }
        auto stride = dimN;
        int64_t dstOffset = static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) +
                            static_cast<int64_t>(curCol * baseN * l0CTileHeight);
        dstOffset = dstOffset >> 1; // splitN

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.nSize = AscendC::Ceil(params.nSize, blockCount) * blockCount;
        }
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride >> 1; // splitN
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.dstStride = AscendC::Ceil(params.dstStride, blockCount) * blockCount;
        }
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.dualDstCtl = static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_N);
        params.subBlockId = id;
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_ROW_MAJOR_UB>(co2Local[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyOutSplitNWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<AscendC::PhyPosIsUB(OutputType::pos)    // output to UB
                                      && OutputType::format == CubeFormat::NZ // NZ
                                      && !AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T,
                                                                                typename InputType::T>()>> { // no quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& co2Local,
                                      const AscendC::LocalTensor<SrcT>& co1Local, int curRow, int curCol,
                                      int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                      int orgM, int orgN, int orgKc, uint8_t id = 0)
    {
#if defined(__DAV_C310__)
        uint32_t stride = static_cast<uint32_t>(orgM * AscendC::BLOCK_CUBE);
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;
        dstOffset = dstOffset >> 1; // splitM or splitN

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.dualDstCtl = static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_N);
        params.subBlockId = id;
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_NZ_UB>(co2Local[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif