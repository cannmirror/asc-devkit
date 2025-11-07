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

#define MaxDeqNums 256
#define ReluPreAddr 1
#define ReluPostAddr 2
#define QuantPostAddr 3
#define AntiqAddr 4

namespace AscendC {

/* **************************************************************************************************
 * SPR                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void SetFixPipeConfigImpl(
    const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre, bool isUnitFlag = false)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixPipeConfig is not support on this version!"); });
}

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &preTensor, bool isUnitFlag = false)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixPipeConfig is not support on this version!"); });
}

__aicore__ inline void SetFixPipeConfigImpl(uint64_t config)
{
    set_fpc(config);
}

__aicore__ inline void SetFixPipeConfigImpl(const FixPipeConfigParams &params)
{
    uint64_t config = 0;
    // 当前地址分配有问题，临时配合DataCopyL12FBImpl手动改成0，后续地址分配问题解决再删除
    config |= ((((static_cast<uint64_t>(0) & 0xFFFF) >> 6) & 0xFF) << 0);
    config |= ((((static_cast<uint64_t>(0) & 0xFFFF) >> 7) & 0xFF) << 8);
    config |= ((((static_cast<uint64_t>(0) & 0xFFFF) >> 6) & 0xFF) << 16);
    config |= ((((static_cast<uint64_t>(0) & 0xFFFF) >> 7) & 0xFF) << 24);
    config |= ((((static_cast<uint64_t>(0) & 0xFFFF) >> 6) & 0xFF) << 32);
    config |= ((static_cast<uint64_t>(params.unitFlag) & 0x1) << 63);
    set_fpc(config);
}

template <typename T>
__aicore__ inline void SetFixpipePreQuantFlagImpl(const FixPipePreQuantParams<T> &params)
{
    uint64_t config = 0;
    config |= ((((static_cast<uint64_t>(params.offset0) >> 9) & 0x7F) | ((((uint64_t)params.offset0 >> 31) & 0x1) << 7)) |
               ((uint64_t)GetScalarBitcodeValue(params.f16Multiplier) & 0xFFFF) |
               ((((uint64_t)GetScalarBitcodeValue(params.scalarValue) >> 13) & 0x7FFFF) << 13));
    config |= (((uint64_t)params.offset0 & 0x1FF) |
                  (((uint64_t)params.offset1 & 0xF) | ((((uint64_t)params.offset1 >> 7) & 0x1) << 4)) |
                  (((uint64_t)params.offset2 & 0xFF) | ((((uint64_t)params.offset2 >> 15) & 0x1) << 8)))
              << 37;
    config |= ((uint64_t)params.isSigned & 0x1) << 46;
    config |= ((uint64_t)params.bitMaskNum & 0x7) << 47;
    set_quant_pre(config);
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(uint64_t config)
{
    set_relu_alpha(config);
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(const FixPipeLeakyReluAlphaParams &params)
{
    uint64_t config = 0;
    config |= (((uint64_t)GetScalarBitcodeValue(params.preValue) >> 13) & 0x7FFFF) << 13;
    config |= (((uint64_t)GetScalarBitcodeValue(params.postValue) >> 13) & 0x7FFFF) << 45;
    set_relu_alpha(config);
}

__aicore__ inline void SetFixpipeEltAnitqImpl(uint64_t config)
{
    set_elt_antiq_para(config);
}

template <typename T>
__aicore__ inline void SetFixpipeEltAnitqImpl(const FixPipeEltAntiqParams<T> &params)
{
    uint64_t config = 0;
    config |= (uint64_t)GetScalarBitcodeValue(params.scalarValue) & 0xFFFF;
    config |= (((uint64_t)params.s4Offset & 0x7) | ((((uint64_t)params.s4Offset >> 7) & 0x1) << 3)) << 16;
    config |= ((uint64_t)params.b8Offset & 0xFF) << 20;
    config |= ((uint64_t)params.s16Offset & 0xFFFF) << 28;
    config |= ((uint64_t)params.eltAntiqCfg & 0xF) << 44;
    set_elt_antiq_para(config);
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(uint64_t config)
{
    set_elt_src_para(config);
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(const FixpipeEltwiseAddrParams &params)
{
    uint64_t config = 0;
    config |= (uint64_t)params.c0ChannelStride & 0xFFFF;
    config |= ((uint64_t)params.eltSrcAddr & 0xFFFF) << 16;
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
    uint64_t config = 0;
    config = config | ((uint64_t)ndNum);              // ND_PARA[15:0], nd number.
    config = config | ((uint64_t)srcNdStride << 16);  // ND_PARA[31:16], src nd stride.
    config = config | ((uint64_t)dstNdStride << 32);  // ND_PARA[47:32], dst nd stride.
    set_loop3_para(config);
}

__aicore__ inline void SetFixpipePreQuantFlagImpl(uint64_t config)
{
    set_quant_pre(config);
}

__aicore__ inline void SetFixPipeClipReluImpl(uint64_t config)
{
    set_fix_clip_relu(config);
}

template <typename T, typename U>
__aicore__ inline void SetFixPipeClipReluImpl(const FixpipeClipReluParams<T, U> &intriParams)
{
    uint64_t config = 0;
    config = config | ((uint64_t)intriParams.preValue);
    config = config | (((uint64_t)intriParams.postValue) << 16);
    set_fix_clip_relu(config);
}

__aicore__ inline void SetFixpipeLoop3Impl(uint64_t config)
{
    set_loop3_para(config);
}

__aicore__ inline void SetFixpipeLoop3Impl(const FixpipeLoop3Params &intriParams)
{
    uint64_t config = 0;
    config = config | ((uint64_t)intriParams.loopSize);
    config = config | (((uint64_t)intriParams.srcStride) << 16);
    config = config | (((uint64_t)intriParams.dstStride) << 32);
    set_loop3_para(config);
}

__aicore__ inline void SetFixpipeLoop4Impl(uint64_t config)
{
    set_loop4_para(config);
}

__aicore__ inline void SetFixpipeLoop4Impl(const FixpipeLoop4Params &intriParams)
{
    uint64_t config = 0;
    config = config | ((uint64_t)intriParams.loopSize);
    config = config | (((uint64_t)intriParams.srcStride) << 16);
    config = config | (((uint64_t)intriParams.dstStride) << 32);
    set_loop4_para(config);
}

__aicore__ inline void SetFixpipeChannnelImpl(uint64_t config)
{
    set_channel_para(config);
}

__aicore__ inline void SetFixpipeChannnelImpl(const FixpipeChannelParams &intriParams)
{
    uint64_t config = 0;
    config = config | ((uint64_t)intriParams.c0ChannelStride);
    config = config | (((uint64_t)intriParams.channelMergeStride) << 16);
    config = config | (((uint64_t)intriParams.c0ChannelIndicator) << 32);
    config = config | (((uint64_t)intriParams.srcStride) << 48);
    set_channel_para(config);
}

/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
struct FixpipeTiling {
    uint16_t nIterNum = 0;
    uint16_t nSize = 0;
    bool isDb = false;
    uint16_t tailNSize = 0;
};

// fixpipe tiling calculating
__aicore__ inline FixpipeTiling GenFixpipeTiling(uint16_t n)
{
    FixpipeTiling tiling;
    if (n <= MaxDeqNums) {
        tiling.nIterNum = 1;
        tiling.nSize = n;
        tiling.isDb = false;
        tiling.tailNSize = 0;
    } else {
        tiling.isDb = true;
        uint16_t dbMaxDeqNums = MaxDeqNums / 2;
        tiling.nIterNum = n / dbMaxDeqNums;
        tiling.nSize = dbMaxDeqNums;
        tiling.tailNSize = n % dbMaxDeqNums;
    }
    return tiling;
}

template <typename T>
struct FixpipeInfoParams {
    __aicore__ inline FixpipeInfoParams()
    {}

    __aicore__ inline FixpipeInfoParams(
        const FixpipeParams<T> &intriParams, const uint8_t srcByteSize, const uint8_t dstByteSize)
    {
        quantCfg = intriParams.quantCfg;
        tiling = GenFixpipeTiling(intriParams.nSize);
        totalN = intriParams.nSize;
        n = intriParams.nSize / tiling.nIterNum;
        srcOffset = intriParams.mSize * n * srcByteSize;
        if (intriParams.nz2ndEnable == false) {
            dstOffset = intriParams.mSize * n * dstByteSize;
        } else {
            dstOffset = n * dstByteSize;
        }
    }
    uint64_t totalN = 0;
    uint16_t n = 0;
    uint16_t srcOffset = 0;
    uint16_t dstOffset = 0;
    uint64_t quantCfg = 0;
    __cbuf__ uint64_t *cbufWorkspace;
    // fixpipe tiling
    FixpipeTiling tiling;
};

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams)
{
    copy_matrix_cc_to_cbuf((__cbuf__ T *)dst,
        (__cc__ U *)src,
        0,
        intriParams.nSize,
        intriParams.mSize,
        intriParams.dstStride,
        intriParams.srcStride,
        intriParams.preClipReluMode,
        intriParams.unitFlag,
        intriParams.quantParams.preQuantMode,
        intriParams.preReluMode,
        intriParams.channelSplitEnable,
        intriParams.nz2ndEnable,
        intriParams.quantParams.postQuantMode,
        intriParams.postReluMode,
        intriParams.postClipReluMode,
        intriParams.loopEnhanceEnable,
        intriParams.eltwiseOp,
        intriParams.eltwiseAntiqEnable,
        intriParams.loopEnhanceMergeEnable,
        intriParams.c0PadEnable,
        intriParams.postWinoEnable,
        intriParams.broadCastEnable,
        intriParams.nz2dnEnable);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams)
{
    copy_matrix_cc_to_gm((__gm__ T *)dst,
        (__cc__ U *)src,
        0,
        intriParams.nSize,
        intriParams.mSize,
        intriParams.dstStride,
        intriParams.srcStride,
        intriParams.preClipReluMode,
        intriParams.unitFlag,
        intriParams.quantParams.preQuantMode,
        intriParams.preReluMode,
        intriParams.channelSplitEnable,
        intriParams.nz2ndEnable,
        intriParams.quantParams.postQuantMode,
        intriParams.postReluMode,
        intriParams.postClipReluMode,
        intriParams.loopEnhanceEnable,
        intriParams.eltwiseOp,
        intriParams.eltwiseAntiqEnable,
        intriParams.loopEnhanceMergeEnable,
        intriParams.c0PadEnable,
        intriParams.postWinoEnable,
        intriParams.broadCastEnable,
        intriParams.nz2dnEnable);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams)
{
    copy_matrix_cc_to_ub((__ubuf__ T *)dst,
        (__cc__ U *)src,
        0,
        intriParams.nSize,
        intriParams.mSize,
        intriParams.dstStride,
        intriParams.srcStride,
        intriParams.preClipReluMode,
        intriParams.unitFlag,
        intriParams.quantParams.preQuantMode,
        intriParams.preReluMode,
        intriParams.channelSplitEnable,
        intriParams.nz2ndEnable,
        intriParams.quantParams.postQuantMode,
        intriParams.postReluMode,
        intriParams.postClipReluMode,
        intriParams.loopEnhanceEnable,
        intriParams.eltwiseOp,
        intriParams.eltwiseAntiqEnable,
        intriParams.loopEnhanceMergeEnable,
        intriParams.c0PadEnable,
        intriParams.postWinoEnable,
        intriParams.broadCastEnable,
        intriParams.nz2dnEnable);
}

// L0C->L1/UB
template <typename T, typename U>
__aicore__ inline void Fixpipe(
    const LocalTensor<T> &dstLocal, const LocalTensor<U> &srcLocal, const FixpipeParams<U> &intriParams)
{
    if constexpr ((!IsSameType<U, int32_t>::value) && (!IsSameType<U, half>::value)) {
        ASCENDC_ASSERT(
            false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<T, int32_t>::value) && (!IsSameType<T, half>::value) &&
                         (!IsSameType<T, int8_t>::value) && (!IsSameType<T, int16_t>::value)) {
        ASCENDC_ASSERT(false,
            { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        const Hardware dstHWPos = GetPhyType((QuePosition)dstLocal.GetPosition());
        if (dstHWPos == Hardware::UB) {
            FixpipeL0C2UBImpl(
                (__ubuf__ T *)dstLocal.GetPhyAddr(), (__cc__ U *)srcLocal.GetPhyAddr(), intriParams);
        } else {
            FixpipeL0C2L1Impl(
                (__cbuf__ T *)dstLocal.GetPhyAddr(), (__cc__ U *)srcLocal.GetPhyAddr(), intriParams);
        }
    }
}

// L0C->GM
template <typename T, typename U>
__aicore__ inline void Fixpipe(
    const GlobalTensor<T> &dstGlobal, const LocalTensor<U> &srcLocal, const FixpipeParams<U> &intriParams)
{
#ifdef __CCE_KT_TEST__
    bool isUsedProcessLock = false;
    if (g_isAtomic == true) {
        ProcessLock::GetProcessLock()->Write();
        isUsedProcessLock = true;
    }
#endif  // __CCE_KT_TEST__
    if constexpr ((!IsSameType<U, int32_t>::value) && (!IsSameType<U, half>::value)) {
        ASCENDC_ASSERT(
            false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<T, int32_t>::value) && (!IsSameType<T, half>::value) &&
                         (!IsSameType<T, int8_t>::value) && (!IsSameType<T, int16_t>::value)) {
        ASCENDC_ASSERT(false,
            { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        FixpipeL0C2GMImpl((__gm__ T *)dstGlobal.GetPhyAddr(), (__cc__ U *)srcLocal.GetPhyAddr(), intriParams);
    }
#ifdef __CCE_KT_TEST__
    if (isUsedProcessLock == true) {
        isUsedProcessLock = false;
        ProcessLock::GetProcessLock()->Unlock();
    }
#endif  // __CCE_KT_TEST__
}

template <typename T>
__aicore__ inline void CopyDeqTensorToFbuf(
    const FixpipeInfoParams<T> &fixpipeInfo, uint16_t calNSize, uint16_t nIterIndex)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint64_t fpcAddr = 0;
    uint64_t cbufOffset = 0;
    uint64_t frontAddrU64 = (nIterIndex & 0x1) * 1024;
    uint64_t frontAddrU32 = (nIterIndex & 0x1) * 512;

    if (GetBit(fixpipeInfo.quantCfg, QUANTPRE_SCALE_VECTOR_CFGBIT)) {
        uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), 128) * 128;
        __fbuf__ uint64_t *deqTensorTempBuf =
            AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(frontAddrU64, deqDataSize / sizeof(uint64_t));
        uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
        uint16_t fbufBurstLen = deqDataSize / 64;
        copy_cbuf_to_fbuf(deqTensorTempBuf, fixpipeInfo.cbufWorkspace + deqValueOffset, 1, fbufBurstLen, 0, 0);
        AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
        fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf) >> (uint64_t)7) << 8;
        cbufOffset += fixpipeInfo.totalN * sizeof(uint64_t) / 8;
    }

    if (GetBit(fixpipeInfo.quantCfg, QUANTPOST_SCALE_VECTOR_CFGBIT)) {
        uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), 128) * 128;
        __fbuf__ uint64_t *deqTensorTempBuf = AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(
            frontAddrU64 + (QuantPostAddr << 16), deqDataSize / sizeof(uint64_t));
        uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
        uint16_t fbufBurstLen = deqDataSize / 64;
        copy_cbuf_to_fbuf(
            deqTensorTempBuf, fixpipeInfo.cbufWorkspace + cbufOffset + deqValueOffset, 1, fbufBurstLen, 0, 0);
        AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
        fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf) >> (uint64_t)7) << 24;
        cbufOffset += fixpipeInfo.totalN * sizeof(uint64_t) / 8;
    }

    if (GetBit(fixpipeInfo.quantCfg, PRERELU_SCALE_VECTOR_CFGBIT)) {
        uint16_t reluDataSize = DivCeil(calNSize * sizeof(uint32_t), 64) * 64;
        __fbuf__ uint64_t *deqTensorTempBuf = AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(
            frontAddrU32 + (ReluPreAddr << 16), reluDataSize / sizeof(uint32_t));
        uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
        uint16_t fbufBurstLen = reluDataSize / 64;
        copy_cbuf_to_fbuf(
            deqTensorTempBuf, fixpipeInfo.cbufWorkspace + cbufOffset + deqValueOffset, 1, fbufBurstLen, 0, 0);
        AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
        fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf) >> (uint64_t)6);
        cbufOffset += fixpipeInfo.totalN * sizeof(uint32_t) / 8;
    }

    if (GetBit(fixpipeInfo.quantCfg, POSTREELU_SCALE_VECTOR_CFGBIT)) {
        uint16_t reluDataSize = DivCeil(calNSize * sizeof(uint32_t), 64) * 64;
        __fbuf__ uint64_t *deqTensorTempBuf = AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(
            frontAddrU32 + (ReluPostAddr << 16), reluDataSize / sizeof(uint32_t));
        uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
        uint16_t fbufBurstLen = reluDataSize / 64;
        copy_cbuf_to_fbuf(
            deqTensorTempBuf, fixpipeInfo.cbufWorkspace + cbufOffset + deqValueOffset, 1, fbufBurstLen, 0, 0);
        AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
        fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf) >> (uint64_t)6) << 16;
        cbufOffset += fixpipeInfo.totalN * sizeof(uint32_t) / 8;
    }

    if (GetBit(fixpipeInfo.quantCfg, ELTWISEANTIQ_SCALE_VECTOR_CFGBIT)) {
        uint16_t antiqDataSize = DivCeil(calNSize * sizeof(uint32_t), 64) * 64;
        __fbuf__ uint64_t *deqTensorTempBuf = AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(
            frontAddrU32 + (AntiqAddr << 16), antiqDataSize / sizeof(uint32_t));
        uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
        uint16_t fbufBurstLen = antiqDataSize / 64;
        copy_cbuf_to_fbuf(
            deqTensorTempBuf, fixpipeInfo.cbufWorkspace + cbufOffset + deqValueOffset, 1, fbufBurstLen, 0, 0);
        AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
        fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf) >> (uint64_t)6) << 32;
        cbufOffset += fixpipeInfo.totalN * sizeof(uint32_t) / 8;
    }
    set_fpc(fpcAddr);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams,
    const FixpipeInfoParams<U> &fixpipeInfo)
{
    copy_matrix_cc_to_gm((__gm__ T *)dst,
        (__cc__ U *)src,
        0,
        fixpipeInfo.n,
        intriParams.mSize,
        intriParams.dstStride,
        intriParams.srcStride,
        intriParams.preClipReluMode,
        intriParams.unitFlag,
        intriParams.quantParams.preQuantMode,
        intriParams.preReluMode,
        intriParams.channelSplitEnable,
        intriParams.nz2ndEnable,
        intriParams.quantParams.postQuantMode,
        intriParams.postReluMode,
        intriParams.postClipReluMode,
        intriParams.loopEnhanceEnable,
        intriParams.eltwiseOp,
        intriParams.eltwiseAntiqEnable,
        intriParams.loopEnhanceMergeEnable,
        intriParams.c0PadEnable,
        intriParams.postWinoEnable,
        intriParams.broadCastEnable,
        intriParams.nz2dnEnable);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams,
    const FixpipeInfoParams<U> &fixpipeInfo)
{
    copy_matrix_cc_to_ub((__ubuf__ T *)dst,
        (__cc__ U *)src,
        0,
        fixpipeInfo.n,
        intriParams.mSize,
        intriParams.dstStride,
        intriParams.srcStride,
        intriParams.preClipReluMode,
        intriParams.unitFlag,
        intriParams.quantParams.preQuantMode,
        intriParams.preReluMode,
        intriParams.channelSplitEnable,
        intriParams.nz2ndEnable,
        intriParams.quantParams.postQuantMode,
        intriParams.postReluMode,
        intriParams.postClipReluMode,
        intriParams.loopEnhanceEnable,
        intriParams.eltwiseOp,
        intriParams.eltwiseAntiqEnable,
        intriParams.loopEnhanceMergeEnable,
        intriParams.c0PadEnable,
        intriParams.postWinoEnable,
        intriParams.broadCastEnable,
        intriParams.nz2dnEnable);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams,
    const FixpipeInfoParams<U> &fixpipeInfo)
{
    copy_matrix_cc_to_cbuf((__cbuf__ T *)dst,
        (__cc__ U *)src,
        0,
        fixpipeInfo.n,
        intriParams.mSize,
        intriParams.dstStride,
        intriParams.srcStride,
        intriParams.preClipReluMode,
        intriParams.unitFlag,
        intriParams.quantParams.preQuantMode,
        intriParams.preReluMode,
        intriParams.channelSplitEnable,
        intriParams.nz2ndEnable,
        intriParams.quantParams.postQuantMode,
        intriParams.postReluMode,
        intriParams.postClipReluMode,
        intriParams.loopEnhanceEnable,
        intriParams.eltwiseOp,
        intriParams.eltwiseAntiqEnable,
        intriParams.loopEnhanceMergeEnable,
        intriParams.c0PadEnable,
        intriParams.postWinoEnable,
        intriParams.broadCastEnable,
        intriParams.nz2dnEnable);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2GMImplN(const GlobalTensor<T> &dstGlobal, const LocalTensor<U> &srcLocal,
    const FixpipeInfoParams<U> &fixpipeInfo, const FixpipeParams<U> &intriParams, uint16_t calNSize,
    uint16_t nIterIndex)
{
    CopyDeqTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    uint32_t srcOffset = nIterIndex * fixpipeInfo.srcOffset;
    uint32_t dstOffset = nIterIndex * fixpipeInfo.dstOffset;
    FixpipeL0C2GMImpl((__gm__ T *)(dstGlobal.GetPhyAddr() + dstOffset),
        (__cc__ U *)(srcLocal.GetPhyAddr() + srcOffset),
        intriParams,
        fixpipeInfo);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2UBImplN(const LocalTensor<T> &dstGlobal, const LocalTensor<U> &srcLocal,
    const FixpipeInfoParams<U> &fixpipeInfo, const FixpipeParams<U> &intriParams, uint16_t calNSize,
    uint16_t nIterIndex)
{
    CopyDeqTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    uint32_t srcOffset = nIterIndex * fixpipeInfo.srcOffset;
    uint32_t dstOffset = nIterIndex * fixpipeInfo.dstOffset;
    FixpipeL0C2UBImpl((__ubuf__ T *)(dstGlobal.GetPhyAddr() + dstOffset),
        (__cc__ U *)(srcLocal.GetPhyAddr() + srcOffset),
        intriParams,
        fixpipeInfo);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2L1ImplN(const LocalTensor<T> &dstGlobal, const LocalTensor<U> &srcLocal,
    const FixpipeInfoParams<U> &fixpipeInfo, const FixpipeParams<U> &intriParams, uint16_t calNSize,
    uint16_t nIterIndex)
{
    CopyDeqTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    uint32_t srcOffset = nIterIndex * fixpipeInfo.srcOffset;
    uint32_t dstOffset = nIterIndex * fixpipeInfo.dstOffset;
    FixpipeL0C2L1Impl((__cbuf__ T *)(dstGlobal.GetPhyAddr() + dstOffset),
        (__cc__ U *)(srcLocal.GetPhyAddr() + srcOffset),
        intriParams,
        fixpipeInfo);
}

template <typename T, typename U>
__aicore__ inline void Fixpipe(const GlobalTensor<T> &dstGlobal, const LocalTensor<U> &srcLocal,
    const LocalTensor<uint64_t> &cbufWorkspace, const FixpipeParams<U> &intriParams)
{
    if constexpr ((!IsSameType<U, int32_t>::value) && (!IsSameType<U, half>::value)) {
        ASCENDC_ASSERT(
            false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<T, int32_t>::value) && (!IsSameType<T, half>::value) &&
                         (!IsSameType<T, int8_t>::value) && (!IsSameType<T, int16_t>::value)) {
        ASCENDC_ASSERT(false,
            { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        FixpipeInfoParams<U> fixpipeInfo(intriParams, sizeof(U), sizeof(T));
        fixpipeInfo.cbufWorkspace = (__cbuf__ uint64_t *)cbufWorkspace.GetPhyAddr();
        for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
            FixpipeL0C2GMImplN(dstGlobal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.nSize, i);
        }
        if (fixpipeInfo.tiling.tailNSize > 0) {
            FixpipeL0C2GMImplN(dstGlobal,
                srcLocal,
                fixpipeInfo,
                intriParams,
                fixpipeInfo.tiling.tailNSize,
                fixpipeInfo.tiling.nIterNum);
        }
    }
}

// L0C->L1/UB deq tensor quant
template <typename T, typename U>
__aicore__ inline void Fixpipe(const LocalTensor<T> &dstLocal, const LocalTensor<U> &srcLocal,
    const LocalTensor<uint64_t> &cbufWorkspace, const FixpipeParams<U> &intriParams)
{
    if constexpr ((!IsSameType<U, int32_t>::value) && (!IsSameType<U, half>::value)) {
        ASCENDC_ASSERT(
            false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<T, int32_t>::value) && (!IsSameType<T, half>::value) &&
                         (!IsSameType<T, int8_t>::value) && (!IsSameType<T, int16_t>::value)) {
        ASCENDC_ASSERT(false,
            { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        const Hardware dstHWPos = GetPhyType((QuePosition)dstLocal.GetPosition());
        FixpipeInfoParams<U> fixpipeInfo(intriParams, sizeof(U), sizeof(T));
        fixpipeInfo.cbufWorkspace = (__cbuf__ uint64_t *)cbufWorkspace.GetPhyAddr();
        if (dstHWPos == Hardware::UB) {
            for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
                FixpipeL0C2UBImplN(dstLocal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.nSize, i);
            }
            if (fixpipeInfo.tiling.tailNSize > 0) {
                FixpipeL0C2UBImplN(dstLocal,
                    srcLocal,
                    fixpipeInfo,
                    intriParams,
                    fixpipeInfo.tiling.tailNSize,
                    fixpipeInfo.tiling.nIterNum);
            }
        } else {
            for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
                FixpipeL0C2L1ImplN(dstLocal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.nSize, i);
            }
            if (fixpipeInfo.tiling.tailNSize > 0) {
                FixpipeL0C2L1ImplN(dstLocal,
                    srcLocal,
                    fixpipeInfo,
                    intriParams,
                    fixpipeInfo.tiling.tailNSize,
                    fixpipeInfo.tiling.nIterNum);
            }
        }
    }
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

}  // namespace AscendC
#endif  // ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
