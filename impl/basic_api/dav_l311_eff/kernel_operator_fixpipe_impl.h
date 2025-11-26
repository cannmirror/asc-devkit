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
    set_fpc(config);
}

__aicore__ inline void SetFixPipeConfigImpl(const FixPipeConfigParams &params)
{
    uint64_t config = 0;
    // 当前地址分配有问题，临时配合DataCopyL12FBImpl手动改成0，后续地址分配问题解决再删除
    config |= (((((uint64_t)0 & 0xFFFF) >> 6) & 0xFF) << 0);
    config |= (((((uint64_t)0 & 0xFFFF) >> 7) & 0xFF) << 8);
    config |= (((((uint64_t)0 & 0xFFFF) >> 7) & 0xFF) << 24);
    config |= (((((uint64_t)0 & 0xFFFF) >> 6) & 0xFF) << 32);
    set_fpc(config);
}

template<typename T>
__aicore__ inline void SetFixpipePreQuantFlagImpl(const FixPipePreQuantParams<T> &params)
{
    uint64_t config = 0;
    config |= (((((uint64_t)params.offset0 >> 9) & 0x7F) | ((params.offset0 >> 31) & 0x1 << 7)) |
        ((((uint64_t)GetScalarBitcodeValue(params.scalarValue) >> 13) & 0x7FFFF) << 13));
    config |= (((uint64_t)params.offset0 & 0x1FF) |
        (((uint64_t)params.offset1 & 0xF) | ((((uint64_t)params.offset1 >> 7) & 0x1) << 4)) |
        (((uint64_t)params.offset2 & 0xFF) | ((((uint64_t)params.offset2 >> 15) & 0x1) << 8))) << 37;
    config |= ((uint64_t)params.isSigned & 0x1) << 46;
    config |= ((uint64_t)GetScalarBitcodeValue(params.clipMaxValPre) & 0xFFFF) << 48;
    set_quant_pre(config);
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(uint64_t config)
{
    set_relu_alpha(config);
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(const FixPipeLeakyReluAlphaParams &params)
{
    uint64_t config = 0;
    config |= (uint64_t)params.lut & 0x7;
    config |= (((uint64_t)GetScalarBitcodeValue(params.preValue) >> 13) & 0x7FFFF) << 13;
    set_relu_alpha(config);
}

__aicore__ inline void SetFixpipeEltAnitqImpl(uint64_t config)
{
    set_elt_antiq_para(config);
}

template<typename T>
__aicore__ inline void SetFixpipeEltAnitqImpl(const FixPipeEltAntiqParams<T> &params)
{
    uint64_t config = 0;
    config |= (uint64_t)GetScalarBitcodeValue(params.scalarValue) & 0xFFFF;
    config |= ((static_cast<uint64_t>(params.s4Offset) & 0x7) | ((((uint64_t)params.s4Offset >> 7) & 0x1) << 3)) << 16;
    config |= (static_cast<uint64_t>(params.b8Offset) & 0xFF) << 20;
    config |= (static_cast<uint64_t>(params.s16Offset) & 0xFFFF) << 28;
    config |= (static_cast<uint64_t>(params.eltAntiqCfg) & 0xF) << 44;
    set_elt_antiq_para(config);
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(uint64_t config)
{
    set_elt_src_para(config);
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(const FixpipeEltwiseAddrParams &params)
{
    uint64_t config = 0;
    config |= static_cast<uint64_t>(params.c0Indicator) & 0x1;
    config |= (static_cast<uint64_t>(params.eltSrcAddr) & 0xFFFF) << 16;
    config |= (static_cast<uint64_t>(params.srcStride1) & 0xFFFF) << 32;
    config |= (static_cast<uint64_t>(params.srcStride2) & 0xFFFF) << 48;
    set_elt_src_para(config);
}

__aicore__ inline void SetFixpipePostQuantImpl(uint64_t config)
{
    set_quant_post(config);
}

__aicore__ inline void SetFixpipePostQuantImpl(const FixPipePostQuantParams &params)
{
    uint64_t config = 0;
    config |= ((((uint64_t)params.offset0 & 0xF) | (((uint64_t)params.offset0 & 0x80) >> 3)) |
        (((uint64_t)params.offset1 & 0xFF) | (((uint64_t)params.offset1 & 0x8000) >> 7)) |
        ((uint64_t)params.offset2 & 0x1FF));
    config |= ((uint64_t)params.isSigned & 0x1) << 9;
    config |= (((uint64_t)GetScalarBitcodeValue(params.scalarValue) >> 13) & 0x7FFFF) << 13;
    config |= ((((uint64_t)params.offset2 >> 9) & 0x7F) | ((((uint64_t)params.offset2 >> 31) & 0x1) << 7)) << 32;
    config |= ((uint64_t)params.shiftValue & 0x1F) << 40;
    set_quant_post(config);
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
    set_quant_pre(config);
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
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe is not support on this version!"); })
}
// L0C->L1 deq tensor quant
template <typename T, typename U>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParams<U>& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe is not support on this version!"); })
}

// L0C->GM
template <typename T, typename U>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParams<U>& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe is not support on this version!"); })
}

// L0C->GM deq tensor quant
template <typename T, typename U>
__aicore__ inline void Fixpipe(const GlobalTensor<T> &dst, const LocalTensor<U> &src,
    const LocalTensor<uint64_t> &cbufWorkspace, const FixpipeParams<U> &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe is not support on this version!"); })
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
