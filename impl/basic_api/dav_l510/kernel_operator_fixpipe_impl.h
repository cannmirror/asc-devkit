/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_fixpipe_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H

#include "kernel_operator_set_spr_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * SPR                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
    bool isUnitFlag = false)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixPipeConfigImpl is not support on this version!"); });
}

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &pre, bool isUnitFlag = false)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixPipeConfigImpl is not support on this version!"); });
}

__aicore__ inline void SetFixPipeConfigImpl(uint64_t config)
{
    set_m_fpc(config);
}

__aicore__ inline void SetFixPipeConfigImpl(const FixPipeConfigParams &params)
{
        constexpr uint32_t POS_PRE_RELU = 0;
        constexpr uint32_t POS_PRE_QUANT = 8;
        constexpr uint32_t POS_POST_QUANT = 24;
        constexpr uint32_t POS_ANTIQ = 32;
        constexpr uint32_t SHIFT_SCALE = 40;

        uint64_t config = 0;

        config |= (uint64_t)params.preReluAddr << POS_PRE_RELU;
        config |= (uint64_t)params.preQuantAddr << POS_PRE_QUANT;
        config |= (uint64_t)params.postQuantAddr << POS_POST_QUANT;
        config |= (uint64_t)params.antiquantAddr << POS_ANTIQ;
        config |= (uint64_t)params.shiftValue << SHIFT_SCALE;

        set_m_fpc(config);
}

template<typename T>
__aicore__ inline void SetFixpipePreQuantFlagImpl(const FixPipePreQuantParams<T> &params)
{
    uint64_t config = 0;
    // [7:0] is higher bit of s17
    config |= ((((uint64_t)params.offset0 >> 9) & 0x7F) | ((params.offset0 >> 31) & 0x1 << 7));
    if (params.isSft) {
        // [20:16] is shift scale in u5
        config |= ((params.shiftScale) & 0x1F << 16);
        // [31:21] is mul scale in u11
        config |= ((params.mulScale) & 0x7FF << 21);
    } else {
        // [31:13] is M1 or u5 and u11 in req8_sft/deqs16_sft
        config |= ((((uint64_t)GetScalarBitcodeValue(params.scalarValue) >> 13) & 0x7FFFF) << 13);
    }
    // [45:37] is s9 or lower bit of s17, [41:37] is s5
    config |= (((uint64_t)params.offset0 & 0x1FF) |
        (((uint64_t)params.offset1 & 0xF) | ((((uint64_t)params.offset1 >> 7) & 0x1) << 4)) |
        (((uint64_t)params.offset2 & 0xFF) | ((((uint64_t)params.offset2 >> 15) & 0x1) << 8))) << 37;
    // max value of preClipRelu
    config |= ((uint64_t)GetScalarBitcodeValue(params.clipMaxValPre) & 0xFFFF) << 48;
    set_m_quant_pre(config);
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(uint64_t config)
{
    set_m_relu_alpha(config);
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(const FixPipeLeakyReluAlphaParams &params)
{
    uint64_t config = 0;
    config |= (uint64_t)params.lut & 0x7;
    // drop lower 13 mantissa bits of float 32
    config |= (((uint64_t)GetScalarBitcodeValue(params.preValue) >> 13) & 0x7FFFF) << 13;
    set_m_relu_alpha(config);
}

__aicore__ inline void SetFixpipeEltAnitqImpl(uint64_t config)
{
    set_m_elt_antiq_para(config);
}

template<typename T>
__aicore__ inline void SetFixpipeEltAnitqImpl(const FixPipeEltAntiqParams<T> &params)
{
    uint64_t config = 0;
    if (params.isSft) {
        // [4:0] is shift scale in u5
        config |= ((params.shiftScale) & 0x1F);
        // [15:5] is mul scale in u11
        config |= ((params.mulScale) & 0x7FF << 5);
    } else {
        config |= (uint64_t)GetScalarBitcodeValue(params.scalarValue) & 0xFFFF;
    }
    config |= (((uint64_t)params.s4Offset & 0x7) | ((((uint64_t)params.s4Offset >> 7) & 0x1) << 3)) << 16;
    config |= ((uint64_t)params.b8Offset & 0xFF) << 20;
    config |= ((uint64_t)params.s16Offset & 0xFFFF) << 28;
    config |= ((uint64_t)params.eltAntiqCfg & 0xF) << 44;
    set_m_elt_antiq_para(config);
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(uint64_t config)
{
    set_m_elt_src_para(config);
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(const FixpipeEltwiseAddrParams &params)
{
    uint64_t config = 0;
    config |= (uint64_t)params.c0Indicator & 0x1;
    config |= ((uint64_t)params.eltSrcAddr & 0xFFFF) << 16;
    config |= ((uint64_t)params.srcStride1 & 0xFFFF) << 32;
    config |= ((uint64_t)params.srcStride2 & 0xFFFF) << 48;
    set_m_elt_src_para(config);
}

__aicore__ inline void SetFixpipePostQuantImpl(uint64_t config)
{
    set_m_quant_post(config);
}

__aicore__ inline void SetFixpipePostQuantImpl(const FixPipePostQuantParams &params)
{
    uint64_t config = 0;
    // s5
    config |= (((uint64_t)params.offset0 & 0xF) | (((uint64_t)params.offset0 & 0x80) >> 3));
    // s9
    config |= (((uint64_t)params.offset1 & 0xFF) | (((uint64_t)params.offset1 & 0x8000) >> 7));
    // lower 9 bit of s17
    config |= ((uint64_t)params.offset2 & 0x1FF);
    // M3
    config |= (((uint64_t)GetScalarBitcodeValue(params.scalarValue) >> 13) & 0x7FFFF) << 13;
    // higher 8 bit of s17
    config |= ((((uint64_t)params.offset2 >> 9) & 0x7F) | ((((uint64_t)params.offset2 >> 31) & 0x1) << 7)) << 32;
    // shiftvalue
    config |= ((uint64_t)params.shiftValue & 0x1F) << 40;
    set_m_quant_post(config);
}

__aicore__ inline void SetFixpipeNz2ndFlagImpl(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "SetFixpipeNz2ndFlag is not support on this version!");
    }
    );
}

__aicore__ inline void SetFixpipePreQuantFlagImpl(uint64_t config)
{
    set_m_quant_pre(config);
}

__aicore__ inline void SetFixPipeClipReluImpl(uint64_t config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixPipeClipRelu is not support on this version!"); });
}

template <typename T, typename U>
__aicore__ inline void SetFixPipeClipReluImpl(const FixpipeClipReluParams<T, U> &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixPipeClipRelu is not support on this version!"); });
}

__aicore__ inline void SetFixpipeLoop3Impl(uint64_t config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeLoop3 is not support on this version!"); });
}

__aicore__ inline void SetFixpipeLoop3Impl(const FixpipeLoop3Params &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeLoop3 is not support on this version!"); });
}

__aicore__ inline void SetFixpipeLoop4Impl(uint64_t config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeLoop4 is not support on this version!"); });
}

__aicore__ inline void SetFixpipeLoop4Impl(const FixpipeLoop4Params &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeLoop4 is not support on this version!"); });
}

__aicore__ inline void SetFixpipeChannnelImpl(uint64_t config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeChannnel is not support on this version!"); });
}

__aicore__ inline void SetFixpipeChannnelImpl(const FixpipeChannelParams &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeChannnel is not support on this version!"); });
}

/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
// L0C->L1
template <typename T, typename U>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParams<U>& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe is not support on this version!"); });
}
// L0C->L1 deq tensor quant
template <typename T, typename U>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParams<U>& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe is not support on this version!"); });
}

// L0C->GM
template <typename T, typename U>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParams<U>& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe is not support on this version!"); });
}

// L0C->GM deq tensor quant
template <typename T, typename U>
__aicore__ inline void Fixpipe(const GlobalTensor<T> &dst, const LocalTensor<U> &src,
    const LocalTensor<uint64_t> &cbufWorkspace, const FixpipeParams<U> &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe is not support on this version!"); });
}

template <typename T, typename U, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ T *dst, __cc__ U *src, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

template <typename T, typename U, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2L1Impl(
    __cbuf__ T *dst, __cc__ U *src, __cbuf__ uint64_t *cbufWorkspace, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

template <typename T, typename U, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ T *dst, __cc__ U *src, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

template <typename T, typename U, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2GMImpl(
    __gm__ T *dst, __cc__ U *src, __cbuf__ uint64_t *cbufWorkspace, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
