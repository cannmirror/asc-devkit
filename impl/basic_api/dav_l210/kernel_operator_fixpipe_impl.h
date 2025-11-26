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
    uint64_t config = 0;
    config = config | ((uint64_t)reluPre.GetPhyAddr() >> 7);         // align with 128bit, FPC[7:0], ReluPreAddr
    config = config | (((uint64_t)quantPre.GetPhyAddr() >> 7) << 8); // align with 128bit, FPC[15:8], QuantPreAddr.
    config = config | ((uint64_t)isUnitFlag << 59);                  // FPC[59], UnitFlag.
    set_fpc(config);
}

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &pre, bool isUnitFlag = false)
{
    uint64_t config = 0;
    if constexpr (setRelu) {
        config = config | ((uint64_t)pre.GetPhyAddr() >> 7); // align with 128bit, FPC[7:0], ReluPreAddr.
    } else {
        config =
            config | (((uint64_t)pre.GetPhyAddr() >> 7) << 8); // align with 128bit, FPC[15:8], QuantPreAddr.
    }
    config = config | ((uint64_t)isUnitFlag << 59); // FPC[59], UnitFlag.
    set_fpc(config);
}

__aicore__ inline void SetFixPipeConfigImpl(uint64_t config)
{
    set_fpc(config);
}

__aicore__ inline void SetFixPipeConfigImpl(const FixPipeConfigParams &params)
{
    uint64_t config = 0;
    // 当前地址分配有问题，临时配合DataCopyL12FBImpl手动改成0，后续地址分配问题解决再删除
    config |= (((((uint64_t)0 & 0xFFFF) >> 7) & 0xFF) << 0);
    config |= (((((uint64_t)0 & 0xFFFF) >> 7) & 0xFF) << 8);
    config |= (((((uint64_t)0 & 0xFFFF) >> 7) & 0xFF) << 16);
    config |= (((uint64_t)params.upsamplingCo & 0xFF) << 40);
    config |= (((uint64_t)params.upsamplingParam & 0xFF) << 48);
    config |= ((uint64_t)params.avgPoolingInitEnable << 56);
    config |= ((uint64_t)params.avgPoolingWrittenEnable << 57);
    config |= (((uint64_t)params.convReluMode & 0x1) << 58);
    config |= ((uint64_t)params.unitFlag << 59);
    set_fpc(config);
}

template<typename T>
__aicore__ inline void SetFixpipePreQuantFlagImpl(const FixPipePreQuantParams<T> &params)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipePreQuant on this version");
    });
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(uint64_t config)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipeLeakyReluAlpha on this version");
    });
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(const FixPipeLeakyReluAlphaParams &params)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipeLeakyReluAlpha on this version");
    });
}

__aicore__ inline void SetFixpipeEltAnitqImpl(uint64_t config)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipeEltAnitq on this version");
    });
}

template<typename T>
__aicore__ inline void SetFixpipeEltAnitqImpl(const FixPipeEltAntiqParams<T> &params)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipeEltAnitq on this version");
    });
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(uint64_t config)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipeEltwiseAddr on this version");
    });
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(const FixpipeEltwiseAddrParams &params)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipeEltwiseAddr on this version");
    });
}

__aicore__ inline void SetFixpipePostQuantImpl(uint64_t config)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipePostQuant on this version");
    });
}

__aicore__ inline void SetFixpipePostQuantImpl(const FixPipePostQuantParams &params)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipePostQuant on this version");
    });
}

__aicore__ inline void SetFixpipeNz2ndFlagImpl(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetFixpipeNz2ndFlag on this version");
    });
}

__aicore__ inline void SetFixpipePreQuantFlagImpl(uint64_t config)
{
    set_quant_pre(config);
}

__aicore__ inline void SetFixPipeClipReluImpl(uint64_t config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixPipeClipRelu is not support on l210!"); });
}

template <typename T, typename U>
__aicore__ inline void SetFixPipeClipReluImpl(const FixpipeClipReluParams<T, U> &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixPipeClipRelu is not support on l210!"); });
}

__aicore__ inline void SetFixpipeLoop3Impl(uint64_t config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeLoop3 is not support on l210!"); });
}

__aicore__ inline void SetFixpipeLoop3Impl(const FixpipeLoop3Params &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeLoop3 is not support on l210!"); });
}

__aicore__ inline void SetFixpipeLoop4Impl(uint64_t config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeLoop4 is not support on l210!"); });
}

__aicore__ inline void SetFixpipeLoop4Impl(const FixpipeLoop4Params &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeLoop4 is not support on l210!"); });
}

__aicore__ inline void SetFixpipeChannnelImpl(uint64_t config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeChannnel is not support on l210!"); });
}

__aicore__ inline void SetFixpipeChannnelImpl(const FixpipeChannelParams &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetFixpipeChannnel is not support on l210!"); });
}
/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */

// tiling params
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
    // deqTensor/reluTensor in FB valid num is 256
    constexpr uint16_t maxDeqNums = 256;
    if (n <= maxDeqNums) {
        tiling.nIterNum = 1;
        tiling.nSize = n;
        tiling.isDb = false;
        tiling.tailNSize = 0;
    } else {
        tiling.isDb = true;
        uint16_t dbMaxDeqNums = maxDeqNums / 2;
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
        // 存放tiling信息
        quantCfg = intriParams.quantCfg;
        biasEnable = intriParams.biasEnable;
        tiling = GenFixpipeTiling(intriParams.nSize);
        totalN = intriParams.nSize;
        n = intriParams.nSize / tiling.nIterNum;
        // src offset和是否使能NZ2ND没有关系
        srcOffset = intriParams.mSize * n * srcByteSize;
        // dstOffset如果是ND输出，要按照N往后排列
        dstOffset = n * dstByteSize;
    }
    uint64_t totalN = 0;
    uint16_t n = 0;
    uint16_t srcOffset = 0;
    uint16_t dstOffset = 0;
    uint64_t quantCfg = 0;
    bool biasEnable = false;
    __cbuf__ uint64_t *cbufWorkspace;
    // fixpipe tiling
    FixpipeTiling tiling;
};

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams)
{
    uint64_t xm = ((uint64_t)(intriParams.unitFlag & 0x3ULL)) |
                  ((uint64_t)((uint64_t)intriParams.eltwiseOp & 0x3ULL) << 2) |
                  ((uint64_t)(intriParams.nSize & 0xFFFULL) << 4) |
                  ((uint64_t)(intriParams.mSize & 0xFFFFULL) << 16) |
                  ((uint64_t)(intriParams.srcBurstGap & 0xFFFFULL) << 32) |
                  ((uint64_t)(intriParams.dstBurstGap & 0xFFFFULL) << 48);
    uint64_t xt = ((uint64_t)(intriParams.quantParams.preQuantMode & 0xFULL)) |
                  ((uint64_t)(intriParams.biasEnable & 0x1ULL) << 4) |
                  ((uint64_t)(intriParams.preReluMode & 0x3ULL) << 5) |
                  ((uint64_t)(intriParams.eltwiseEnable & 0x1ULL) << 7) |
                  ((uint64_t)(intriParams.postReluMode & 0x3ULL) << 8) |
                  ((uint64_t)(intriParams.poolMode & 0x3ULL) << 10) |
                  ((uint64_t)(intriParams.quantParams.postQuantMode & 0x3ULL) << 12) |
                  ((uint64_t)(intriParams.dualMode & 0xFULL) << 16) |
                  ((uint64_t)(intriParams.ws & 0xFFFFULL) << 32) |
                  ((uint64_t)(intriParams.wSize & 0xFFFFULL) << 48);
    fix_matrix_cc_to_ubuf((__ubuf__ T *)dst, (__cc__ U *)src, xm, xt);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams)
{
    uint64_t xm = ((uint64_t)(intriParams.unitFlag & 0x3ULL)) |
                  ((uint64_t)((uint64_t)intriParams.eltwiseOp & 0x3ULL) << 2) |
                  ((uint64_t)(intriParams.nSize & 0xFFFULL) << 4) |
                  ((uint64_t)(intriParams.mSize & 0xFFFFULL) << 16) |
                  ((uint64_t)(intriParams.srcBurstGap & 0xFFFFULL) << 32) |
                  ((uint64_t)(intriParams.dstBurstGap & 0xFFFFULL) << 48);
    uint64_t xt = ((uint64_t)(intriParams.quantParams.preQuantMode & 0xFULL)) |
                  ((uint64_t)(intriParams.biasEnable & 0x1ULL) << 4) |
                  ((uint64_t)(intriParams.preReluMode & 0x3ULL) << 5) |
                  ((uint64_t)(intriParams.eltwiseEnable & 0x1ULL) << 7) |
                  ((uint64_t)(intriParams.postReluMode & 0x3ULL) << 8) |
                  ((uint64_t)(intriParams.poolMode & 0x3ULL) << 10) |
                  ((uint64_t)(intriParams.quantParams.postQuantMode & 0x3ULL) << 12) |
                  ((uint64_t)(intriParams.dualMode & 0xFULL) << 16) |
                  ((uint64_t)(intriParams.ws & 0xFFFFULL) << 32) |
                  ((uint64_t)(intriParams.wSize & 0xFFFFULL) << 48);
    fix_matrix_cc_to_cbuf((__cbuf__ T *)dst, (__cc__ U *)src, xm, xt);
}

// L0C->L1
template <typename T, typename U>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParams<U>& intriParams)
{
    if constexpr ((!IsSameType<U, int32_t>::value) && (!IsSameType<U, half>::value)) {
        ASCENDC_ASSERT(
            false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<T, int32_t>::value) && (!IsSameType<T, half>::value) &&
                         (!IsSameType<T, int8_t>::value) && (!IsSameType<T, uint8_t>::value) &&
                         (!IsSameType<T, int16_t>::value)) {
        ASCENDC_ASSERT(false,
            { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        const Hardware dstHWPos = GetPhyType((QuePosition)dst.GetPosition());
        if (dstHWPos == Hardware::UB) {
            FixpipeL0C2UBImpl(
                (__ubuf__ T *)dst.GetPhyAddr(), (__cc__ U *)src.GetPhyAddr(), intriParams);
        } else {
            FixpipeL0C2L1Impl(
                (__cbuf__ T *)dst.GetPhyAddr(), (__cc__ U *)src.GetPhyAddr(), intriParams);
        }
    }
}

template <typename T>
__aicore__ inline void CopyTensorToFbuf(
    const FixpipeInfoParams<T> &fixpipeInfo, uint16_t calNSize, uint16_t nIterIndex)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint64_t fpcAddr = 0;
    // L1上workspace的固定偏移
    uint64_t cbufOffset = 0;
    // FB上的起始地址
    uint64_t frontAddrReLUbBias = (nIterIndex & 0x1) * 1024;
    uint64_t frontAddrDeq = 2048 + (nIterIndex & 0x1) * 1024;
    uint64_t frontAddrReLUaReq = 4096 + (nIterIndex & 0x1) * 1024;

    if (GetBit(fixpipeInfo.quantCfg, PRERELU_SCALE_VECTOR_CFGBIT) || fixpipeInfo.biasEnable) {
        uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), 128) * 128; // 搬移对齐到128，数据类型写死u64
        __fbuf__ uint64_t *deqTensorTempBuf =
            AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(frontAddrReLUbBias, deqDataSize / sizeof(uint64_t));
        uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
        // L1 -> FB
        uint16_t fbufBurstLen = deqDataSize / 32;  // burst length，单位32B
        copy_cbuf_to_fbuf(deqTensorTempBuf, fixpipeInfo.cbufWorkspace + deqValueOffset, 1, fbufBurstLen, 0, 0);
        AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
        fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf & 0xFFFF) >> (uint64_t)7); // 右移7位，in unit of 128B
        cbufOffset += fixpipeInfo.totalN * sizeof(uint64_t) / 8;
    }

    if (GetBit(fixpipeInfo.quantCfg, QUANTPRE_SCALE_VECTOR_CFGBIT)) {
        uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), 128) * 128; // 搬移对齐到128，数据类型写死u64
        __fbuf__ uint64_t *deqTensorTempBuf =
            AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(frontAddrDeq, deqDataSize / sizeof(uint64_t));
        uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
        // L1 -> FB
        uint16_t fbufBurstLen = deqDataSize / 32;  // burst length，单位32B
        copy_cbuf_to_fbuf(
            deqTensorTempBuf, fixpipeInfo.cbufWorkspace + cbufOffset + deqValueOffset, 1, fbufBurstLen, 0, 0);
        AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
        fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf & 0xFFFF) >> (uint64_t)7) << 8; // 右移7位，in unit of 128B
        cbufOffset += fixpipeInfo.totalN * sizeof(uint64_t) / 8;
    }

    if (GetBit(fixpipeInfo.quantCfg, POSTREELU_SCALE_VECTOR_CFGBIT) ||
        GetBit(fixpipeInfo.quantCfg, QUANTPOST_SCALE_VECTOR_CFGBIT)) {
        uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), 128) * 128; // 搬移对齐到128，数据类型写死u64
        __fbuf__ uint64_t *deqTensorTempBuf =
            AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(frontAddrReLUaReq, deqDataSize / sizeof(uint64_t));
        uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
        // L1 -> FB
        uint16_t fbufBurstLen = deqDataSize / 32;  // burst length，单位32B
        copy_cbuf_to_fbuf(
            deqTensorTempBuf, fixpipeInfo.cbufWorkspace + cbufOffset + deqValueOffset, 1, fbufBurstLen, 0, 0);
        AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
        fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf & 0xFFFF) >> (uint64_t)7) << 16; // 右移7位，in unit of 128B
        cbufOffset += fixpipeInfo.totalN * sizeof(uint64_t) / 8;
    }
    set_fpc(fpcAddr);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams,
    const FixpipeInfoParams<U> &fixpipeInfo)
{
    uint64_t xm = ((uint64_t)(intriParams.unitFlag & 0x3ULL)) |
                  ((uint64_t)((uint64_t)intriParams.eltwiseOp & 0x3ULL) << 2) |
                  ((uint64_t)(fixpipeInfo.n & 0xFFFULL) << 4) |
                  ((uint64_t)(intriParams.mSize & 0xFFFFULL) << 16) |
                  ((uint64_t)(intriParams.srcBurstGap & 0xFFFFULL) << 32) |
                  ((uint64_t)(intriParams.dstBurstGap & 0xFFFFULL) << 48);
    uint64_t xt = ((uint64_t)(intriParams.quantParams.preQuantMode & 0xFULL)) |
                  ((uint64_t)(intriParams.biasEnable & 0x1ULL) << 4) |
                  ((uint64_t)(intriParams.preReluMode & 0x3ULL) << 5) |
                  ((uint64_t)(intriParams.eltwiseEnable & 0x1ULL) << 7) |
                  ((uint64_t)(intriParams.postReluMode & 0x3ULL) << 8) |
                  ((uint64_t)(intriParams.poolMode & 0x3ULL) << 10) |
                  ((uint64_t)(intriParams.quantParams.postQuantMode & 0x3ULL) << 12) |
                  ((uint64_t)(intriParams.dualMode & 0xFULL) << 16) |
                  ((uint64_t)(intriParams.ws & 0xFFFFULL) << 32) |
                  ((uint64_t)(intriParams.wSize & 0xFFFFULL) << 48);
    fix_matrix_cc_to_ubuf((__ubuf__ T *)dst, (__cc__ U *)src, xm, xt);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ T *dst, __cc__ U *src, const FixpipeParams<U> &intriParams,
    const FixpipeInfoParams<U> &fixpipeInfo)
{
    uint64_t xm = ((uint64_t)(intriParams.unitFlag & 0x3ULL)) |
                  ((uint64_t)((uint64_t)intriParams.eltwiseOp & 0x3ULL) << 2) |
                  ((uint64_t)(fixpipeInfo.n & 0xFFFULL) << 4) |
                  ((uint64_t)(intriParams.mSize & 0xFFFFULL) << 16) |
                  ((uint64_t)(intriParams.srcBurstGap & 0xFFFFULL) << 32) |
                  ((uint64_t)(intriParams.dstBurstGap & 0xFFFFULL) << 48);
    uint64_t xt = ((uint64_t)(intriParams.quantParams.preQuantMode & 0xFULL)) |
                  ((uint64_t)(intriParams.biasEnable & 0x1ULL) << 4) |
                  ((uint64_t)(intriParams.preReluMode & 0x3ULL) << 5) |
                  ((uint64_t)(intriParams.eltwiseEnable & 0x1ULL) << 7) |
                  ((uint64_t)(intriParams.postReluMode & 0x3ULL) << 8) |
                  ((uint64_t)(intriParams.poolMode & 0x3ULL) << 10) |
                  ((uint64_t)(intriParams.quantParams.postQuantMode & 0x3ULL) << 12) |
                  ((uint64_t)(intriParams.dualMode & 0xFULL) << 16) |
                  ((uint64_t)(intriParams.ws & 0xFFFFULL) << 32) |
                  ((uint64_t)(intriParams.wSize & 0xFFFFULL) << 48);
    fix_matrix_cc_to_cbuf((__cbuf__ T *)dst, (__cc__ U *)src, xm, xt);
}

//  FixpipeInfoParams---存放二次计算后被多次使用的参数
//  FixpipeParams -- 直接存放原来赋值后就用到的参数
template <typename T, typename U>
__aicore__ inline void FixpipeL0C2UBImplN(const LocalTensor<T> &dst, const LocalTensor<U> &src,
    const FixpipeInfoParams<U> &fixpipeInfo, const FixpipeParams<U> &intriParams, uint16_t calNSize,
    uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    uint32_t srcOffset = nIterIndex * fixpipeInfo.srcOffset;
    uint32_t dstOffset = nIterIndex * fixpipeInfo.dstOffset;
    FixpipeL0C2UBImpl((__ubuf__ T *)(dst.GetPhyAddr() + dstOffset),
        (__cc__ U *)(src.GetPhyAddr() + srcOffset),
        intriParams,
        fixpipeInfo);
}

template <typename T, typename U>
__aicore__ inline void FixpipeL0C2L1ImplN(const LocalTensor<T> &dst, const LocalTensor<U> &src,
    const FixpipeInfoParams<U> &fixpipeInfo, const FixpipeParams<U> &intriParams, uint16_t calNSize,
    uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    uint32_t srcOffset = nIterIndex * fixpipeInfo.srcOffset;
    uint32_t dstOffset = nIterIndex * fixpipeInfo.dstOffset;
    FixpipeL0C2L1Impl((__cbuf__ T *)(dst.GetPhyAddr() + dstOffset),
        (__cc__ U *)(src.GetPhyAddr() + srcOffset),
        intriParams,
        fixpipeInfo);
}

// L0C->L1 deq tensor quant
template <typename T, typename U>
__aicore__ inline void Fixpipe(const LocalTensor<T> &dst, const LocalTensor<U> &src,
    const LocalTensor<uint64_t> &cbufWorkspace, const FixpipeParams<U> &intriParams)
{
    if constexpr ((!IsSameType<U, int32_t>::value) && (!IsSameType<U, half>::value)) {
        ASCENDC_ASSERT(
            false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<T, int32_t>::value) && (!IsSameType<T, half>::value) &&
                         (!IsSameType<T, int8_t>::value) && (!IsSameType<T, uint8_t>::value) &&
                         (!IsSameType<T, int16_t>::value)) {
        ASCENDC_ASSERT(false,
            { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        const Hardware dstHWPos = GetPhyType((QuePosition)dst.GetPosition());
        FixpipeInfoParams<U> fixpipeInfo(intriParams, sizeof(U), sizeof(T));
        // 这里获取的fixpipe tiling里面只有deq的最大size；
        // 存放workspace addr
        fixpipeInfo.cbufWorkspace = (__cbuf__ uint64_t *)cbufWorkspace.GetPhyAddr();
        if (dstHWPos == Hardware::UB) {
            // 整块正常搬移
            for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
                FixpipeL0C2UBImplN(dst, src, fixpipeInfo, intriParams, fixpipeInfo.tiling.nSize, i);
            }
            // 尾块多搬一次
            if (fixpipeInfo.tiling.tailNSize > 0) {
                FixpipeL0C2UBImplN(dst,
                    src,
                    fixpipeInfo,
                    intriParams,
                    fixpipeInfo.tiling.tailNSize,
                    fixpipeInfo.tiling.nIterNum);
            }
        } else {
            // 整块正常搬移
            for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
                FixpipeL0C2L1ImplN(dst, src, fixpipeInfo, intriParams, fixpipeInfo.tiling.nSize, i);
            }
            // 尾块多搬一次
            if (fixpipeInfo.tiling.tailNSize > 0) {
                FixpipeL0C2L1ImplN(dst,
                    src,
                    fixpipeInfo,
                    intriParams,
                    fixpipeInfo.tiling.tailNSize,
                    fixpipeInfo.tiling.nIterNum);
            }
        }
    }
}

// L0C->GM
template <typename T, typename U>
__aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src,
    const FixpipeParams<U>& intriParams)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

// L0C->GM deq tensor quant
template <typename T, typename U>
__aicore__ inline void Fixpipe(const GlobalTensor<T> &dst, const LocalTensor<U> &src,
    const LocalTensor<uint64_t> &cbufWorkspace, const FixpipeParams<U> &intriParams)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
