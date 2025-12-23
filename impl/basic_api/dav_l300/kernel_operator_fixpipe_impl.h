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
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
    bool isUnitFlag = false)
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

__aicore__ inline void SetFixpipePreQuantFlagImpl(uint64_t config)
{
    set_quant_pre(config);
}

__aicore__ inline void SetFixpipeLeakyReluAlphaImpl(uint64_t config)
{
    set_relu_alpha(config);
}

__aicore__ inline void SetFixpipeEltAnitqImpl(uint64_t config)
{
    set_elt_antiq_para(config);
}

__aicore__ inline void SetFixpipeEltwiseAddrImpl(uint64_t config)
{
    set_elt_src_para(config);
}

__aicore__ inline void SetFixpipePostQuantImpl(uint64_t config)
{
    set_quant_post(config);
}

__aicore__ inline void SetFixpipeNz2ndFlagImpl(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    uint64_t config = 0;
    config = config | (static_cast<uint64_t>(ndNum));             // ND_PARA[15:0], nd number.
    config = config | (static_cast<uint64_t>(srcNdStride) << 16); // ND_PARA[31:16], src nd stride.
    config = config | (static_cast<uint64_t>(dstNdStride) << 32); // ND_PARA[47:32], dst nd stride.
    set_nd_para(config);
}

__aicore__ inline void SetFixPipeClipReluImpl(uint64_t config)
{
    set_fix_clip_relu(config);
}

__aicore__ inline void SetFixpipeLoop3Impl(uint64_t config)
{
    set_loop3_para(config);
}


__aicore__ inline void SetFixpipeLoop4Impl(uint64_t config)
{
    set_loop4_para(config);
}

__aicore__ inline void SetFixpipeChannnelImpl(uint64_t config)
{
    set_channel_para(config);
}

/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
struct FixpipeTiling {
    uint16_t nIterNum = 0; // n方向循环的次数
    uint16_t nSize = 0;  // 每次循环的n的size
    bool isDb = false;
    uint16_t tailNSize = 0;  // 尾块的n的size
};

// fixpipe tiling calculating
__aicore__ inline FixpipeTiling GenFixpipeTiling(uint16_t n)
{
    FixpipeTiling tiling;
    // deqTensor/reluTensor in FB valid num is 256
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
        tiling.tailNSize = n % dbMaxDeqNums;  // 尾块的大小
    }
    return tiling;
}

template <typename SrcT>
struct FixpipeInfoParams {
    __aicore__ inline FixpipeInfoParams() {}

    __aicore__ inline FixpipeInfoParams(const FixpipeParams<SrcT>& intriParams, const uint8_t srcByteSize, const uint8_t dstByteSize)
    {
        // 存放tiling信息
        quantCfg = intriParams.quantCfg;
        tiling = GenFixpipeTiling(intriParams.nSize);
        totalN = intriParams.nSize;
        n = intriParams.nSize / tiling.nIterNum;
        // src offset和是否使能NZ2ND没有关系
        srcOffset = intriParams.mSize * n * srcByteSize;
        if (intriParams.nz2ndEnable == false) {
            // 如果搬N次，N次的srcStride应该是固定的值；
            // 最终搬移的src offset = srcStride * n_index * nsize
            dstOffset = intriParams.mSize * n * dstByteSize;
        } else {
            // dstOffset如果是ND输出，要按照N往后排列
            dstOffset = n * dstByteSize;
        }
    }
    uint64_t totalN = 0;
    uint16_t n = 0;
    uint16_t srcOffset = 0;
    uint16_t dstOffset = 0;
    uint64_t quantCfg = 0;
    __cbuf__ uint64_t* cbufWorkspace;
    // fixpipe tiling
    FixpipeTiling tiling;
};

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ DstT* dst, __cc__ SrcT* src, const FixpipeParams<SrcT>& intriParams)
{
    uint8_t antiqMode = (intriParams.eltwiseAntiqMode == eltwise_antiq_t::NO_ANTIQ) ?
        0 : ((static_cast<uint8_t>(intriParams.eltwiseAntiqMode) >> 2) + 1);
    copy_matrix_cc_to_cbuf((__cbuf__ DstT*)dst, (__cc__ SrcT*)src, 0,
        intriParams.nSize, intriParams.mSize, intriParams.dstStride,
        intriParams.srcStride, intriParams.preClipReluMode, intriParams.unitFlag,
        intriParams.quantParams.preQuantMode, intriParams.preReluMode, intriParams.channelSplitEnable,
        intriParams.nz2ndEnable, intriParams.quantParams.postQuantMode, intriParams.postReluMode,
        intriParams.postClipReluMode, intriParams.loopEnhanceEnable, intriParams.eltwiseOp,
        antiqMode, intriParams.loopEnhanceMergeEnable, intriParams.c0PadEnable,
        intriParams.postWinoEnable);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ DstT* dst, __cc__ SrcT* src, const FixpipeParams<SrcT>& intriParams)
{
    uint8_t antiqMode = (intriParams.eltwiseAntiqMode == eltwise_antiq_t::NO_ANTIQ) ?
        0 : ((static_cast<uint8_t>(intriParams.eltwiseAntiqMode) >> 2) + 1);
    copy_matrix_cc_to_gm((__gm__ DstT*)dst, (__cc__ SrcT*)src, 0,
        intriParams.nSize, intriParams.mSize, intriParams.dstStride,
        intriParams.srcStride, intriParams.preClipReluMode, intriParams.unitFlag,
        intriParams.quantParams.preQuantMode, intriParams.preReluMode, intriParams.channelSplitEnable,
        intriParams.nz2ndEnable, intriParams.quantParams.postQuantMode, intriParams.postReluMode,
        intriParams.postClipReluMode, intriParams.loopEnhanceEnable, intriParams.eltwiseOp,
        antiqMode, intriParams.loopEnhanceMergeEnable, intriParams.c0PadEnable,
        intriParams.postWinoEnable);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ DstT* dst, __cc__ SrcT* src, const FixpipeParams<SrcT>& intriParams)
{
    uint8_t antiqMode = (intriParams.eltwiseAntiqMode == eltwise_antiq_t::NO_ANTIQ) ?
        0 : ((static_cast<uint8_t>(intriParams.eltwiseAntiqMode) >> 2) + 1);
    copy_matrix_cc_to_ub((__ubuf__ DstT*)dst, (__cc__ SrcT*)src, 0,
        intriParams.nSize, intriParams.mSize, intriParams.dstStride,
        intriParams.srcStride, intriParams.preClipReluMode, intriParams.unitFlag,
        intriParams.quantParams.preQuantMode, intriParams.preReluMode, intriParams.channelSplitEnable,
        intriParams.nz2ndEnable, intriParams.quantParams.postQuantMode, intriParams.postReluMode,
        intriParams.postClipReluMode, intriParams.loopEnhanceEnable, intriParams.eltwiseOp,
        antiqMode, intriParams.loopEnhanceMergeEnable, intriParams.c0PadEnable,
        intriParams.postWinoEnable);
}

// L0C->L1/UB
template <typename DstT, typename SrcT>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParams<SrcT>& intriParams)
{
    if constexpr ((!IsSameType<SrcT, int32_t>::value) && (!IsSameType<SrcT, half>::value)) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<DstT, int32_t>::value) && (!IsSameType<DstT, half>::value)&&
        (!IsSameType<DstT, int8_t>::value) && (!IsSameType<DstT, int16_t>::value)) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        const Hardware dstHWPos = GetPhyType((QuePosition)dstLocal.GetPosition());
        if (dstHWPos == Hardware::UB) {
            FixpipeL0C2UBImpl((__ubuf__ DstT*)dstLocal.GetPhyAddr(), (__cc__ SrcT*)srcLocal.GetPhyAddr(), intriParams);
        } else {
            FixpipeL0C2L1Impl((__cbuf__ DstT*)dstLocal.GetPhyAddr(), (__cc__ SrcT*)srcLocal.GetPhyAddr(), intriParams);
        }
    }
}

// L0C->GM
template <typename DstT, typename SrcT>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParams<SrcT>& intriParams)
{
#ifdef __CCE_KT_TEST__
    bool isUsedProcessLock = false;
    if (g_isAtomic == true) {
        ProcessLock::GetProcessLock()->Write();
        isUsedProcessLock = true;
    }
#endif // __CCE_KT_TEST__
    if constexpr ((!IsSameType<SrcT, int32_t>::value)&&(!IsSameType<SrcT, half>::value)) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<DstT, int32_t>::value)&&(!IsSameType<DstT, half>::value)&&
        (!IsSameType<DstT, int8_t>::value)&&(!IsSameType<DstT, int16_t>::value)) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        FixpipeL0C2GMImpl((__gm__ DstT*)dstGlobal.GetPhyAddr(), (__cc__ SrcT*)srcLocal.GetPhyAddr(), intriParams);
    }
#ifdef __CCE_KT_TEST__
    if (isUsedProcessLock == true) {
        isUsedProcessLock = false;
        ProcessLock::GetProcessLock()->Unlock();
    }
#endif // __CCE_KT_TEST__
}

template <typename SrcT>
__aicore__ inline void CopyDeqTensorToFbuf(const FixpipeInfoParams<SrcT>& fixpipeInfo, uint16_t calNSize,
    uint16_t nIterIndex)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint64_t fpcAddr = 0;
    // L1上workspace的固定偏移
    uint64_t cbufOffset = 0;
    // FB上的起始地址
    uint64_t frontAddrU64 = (nIterIndex & 0x1) * 1024;
    uint64_t frontAddrU32 = (nIterIndex & 0x1) * 512;

    uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), 128) * 128;  // 搬移对齐到128，数据类型写死u64
    __fbuf__ uint64_t* deqTensorTempBuf =
        AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(frontAddrU64, deqDataSize / sizeof(uint64_t));
    uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
    // L1 -> FB
    uint16_t fbufBurstLen = deqDataSize / 128;  // burst length，单位128B
    copy_cbuf_to_fbuf(deqTensorTempBuf, fixpipeInfo.cbufWorkspace + deqValueOffset, 1, fbufBurstLen, 0, 0);
    AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
    fpcAddr = fpcAddr | (((uint64_t)deqTensorTempBuf & 0xFFFF) >> (uint64_t)7) << 8; // 右移7位，地址也in unit of 128B
    cbufOffset += fixpipeInfo.totalN * sizeof(uint64_t) / 8;
    
    set_fpc(fpcAddr);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ DstT* dst, __cc__ SrcT* src, const FixpipeParams<SrcT>& intriParams,
    const FixpipeInfoParams<SrcT>& fixpipeInfo)
{
    uint8_t antiqMode = (intriParams.eltwiseAntiqMode == eltwise_antiq_t::NO_ANTIQ) ?
        0 : ((static_cast<uint8_t>(intriParams.eltwiseAntiqMode) >> 2) + 1);
    copy_matrix_cc_to_gm((__gm__ DstT*)dst, (__cc__ SrcT*)src, 0,
        fixpipeInfo.n, intriParams.mSize, intriParams.dstStride,
        intriParams.srcStride, intriParams.preClipReluMode, intriParams.unitFlag,
        intriParams.quantParams.preQuantMode, intriParams.preReluMode, intriParams.channelSplitEnable,
        intriParams.nz2ndEnable, intriParams.quantParams.postQuantMode, intriParams.postReluMode,
        intriParams.postClipReluMode, intriParams.loopEnhanceEnable, intriParams.eltwiseOp,
        antiqMode, intriParams.loopEnhanceMergeEnable, intriParams.c0PadEnable,
        intriParams.postWinoEnable);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ DstT* dst, __cc__ SrcT* src, const FixpipeParams<SrcT>& intriParams,
    const FixpipeInfoParams<SrcT>& fixpipeInfo)
{
    uint8_t antiqMode = (intriParams.eltwiseAntiqMode == eltwise_antiq_t::NO_ANTIQ) ?
        0 : ((static_cast<uint8_t>(intriParams.eltwiseAntiqMode) >> 2) + 1);
    copy_matrix_cc_to_ub((__ubuf__ DstT*)dst, (__cc__ SrcT*)src, 0,
        fixpipeInfo.n, intriParams.mSize, intriParams.dstStride,
        intriParams.srcStride, intriParams.preClipReluMode, intriParams.unitFlag,
        intriParams.quantParams.preQuantMode, intriParams.preReluMode, intriParams.channelSplitEnable,
        intriParams.nz2ndEnable, intriParams.quantParams.postQuantMode, intriParams.postReluMode,
        intriParams.postClipReluMode, intriParams.loopEnhanceEnable, intriParams.eltwiseOp,
        antiqMode, intriParams.loopEnhanceMergeEnable, intriParams.c0PadEnable,
        intriParams.postWinoEnable);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ DstT* dst, __cc__ SrcT* src, const FixpipeParams<SrcT>& intriParams,
    const FixpipeInfoParams<SrcT>& fixpipeInfo)
{
    uint8_t antiqMode = (intriParams.eltwiseAntiqMode == eltwise_antiq_t::NO_ANTIQ) ?
        0 : ((static_cast<uint8_t>(intriParams.eltwiseAntiqMode) >> 2) + 1);
    copy_matrix_cc_to_cbuf((__cbuf__ DstT*)dst, (__cc__ SrcT*)src, 0,
        fixpipeInfo.n, intriParams.mSize, intriParams.dstStride,
        intriParams.srcStride, intriParams.preClipReluMode, intriParams.unitFlag,
        intriParams.quantParams.preQuantMode, intriParams.preReluMode, intriParams.channelSplitEnable,
        intriParams.nz2ndEnable, intriParams.quantParams.postQuantMode, intriParams.postReluMode,
        intriParams.postClipReluMode, intriParams.loopEnhanceEnable, intriParams.eltwiseOp,
        antiqMode, intriParams.loopEnhanceMergeEnable, intriParams.c0PadEnable,
        intriParams.postWinoEnable);
}

//  FixpipeInfoParams---存放二次计算后被多次使用的参数
//  FixpipeParams -- 直接存放原来赋值后就用到的参数
template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2GMImplN(const GlobalTensor<DstT> &dstGlobal, const LocalTensor<SrcT> &srcLocal,
    const FixpipeInfoParams<SrcT>& fixpipeInfo, const FixpipeParams<SrcT> &intriParams, uint16_t calNSize, uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyDeqTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    // offset偏移的计算，区分了有无NZ2ND场景
    // NZ2ND的差异已经在fixpipeInfo中解决，在这里只乘nIndex
    uint32_t srcOffset = nIterIndex * fixpipeInfo.srcOffset;
    uint32_t dstOffset = nIterIndex * fixpipeInfo.dstOffset;
    FixpipeL0C2GMImpl((__gm__ DstT*)(dstGlobal.GetPhyAddr() + dstOffset), (__cc__ SrcT*)(srcLocal.GetPhyAddr() + srcOffset),
        intriParams, fixpipeInfo);
}

//  FixpipeInfoParams---存放二次计算后被多次使用的参数
//  FixpipeParams -- 直接存放原来赋值后就用到的参数
template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2UBImplN(const LocalTensor<DstT> &dstGlobal, const LocalTensor<SrcT> &srcLocal,
    const FixpipeInfoParams<SrcT>& fixpipeInfo, const FixpipeParams<SrcT> &intriParams, uint16_t calNSize, uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyDeqTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    // offset偏移的计算，区分了有无NZ2ND场景
    // NZ2ND的差异已经在fixpipeInfo中解决，在这里只乘nIndex
    uint32_t srcOffset = nIterIndex * fixpipeInfo.srcOffset;
    uint32_t dstOffset = nIterIndex * fixpipeInfo.dstOffset;
    FixpipeL0C2UBImpl((__ubuf__ DstT*)(dstGlobal.GetPhyAddr() + dstOffset), (__cc__ SrcT*)(srcLocal.GetPhyAddr() + srcOffset),
        intriParams, fixpipeInfo);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2L1ImplN(const LocalTensor<DstT> &dstGlobal, const LocalTensor<SrcT> &srcLocal,
    const FixpipeInfoParams<SrcT>& fixpipeInfo, const FixpipeParams<SrcT> &intriParams, uint16_t calNSize, uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyDeqTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    // offset偏移的计算，区分了有无NZ2ND场景
    // NZ2ND的差异已经在fixpipeInfo中解决，在这里只乘nIndex
    uint32_t srcOffset = nIterIndex * fixpipeInfo.srcOffset;
    uint32_t dstOffset = nIterIndex * fixpipeInfo.dstOffset;
    FixpipeL0C2L1Impl((__cbuf__ DstT*)(dstGlobal.GetPhyAddr() + dstOffset), (__cc__ SrcT*)(srcLocal.GetPhyAddr() + srcOffset),
        intriParams, fixpipeInfo);
}

// L0C->GM deq tensor quant
template <typename DstT, typename SrcT>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT> &dstGlobal, const LocalTensor<SrcT> &srcLocal,
    const LocalTensor<uint64_t> &cbufWorkspace, const FixpipeParams<SrcT> &intriParams)
{
    if constexpr ((!IsSameType<SrcT, int32_t>::value)&&(!IsSameType<SrcT, half>::value)) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<DstT, int32_t>::value)&&(!IsSameType<DstT, half>::value)&&
        (!IsSameType<DstT, int8_t>::value)&&(!IsSameType<DstT, int16_t>::value)) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        FixpipeInfoParams<SrcT> fixpipeInfo(intriParams, sizeof(SrcT), sizeof(DstT));
        // 这里获取的fixpipe tiling里面只有deq的最大size；
        // 存放workspace addr
        fixpipeInfo.cbufWorkspace = (__cbuf__ uint64_t *)cbufWorkspace.GetPhyAddr();
        // 整块正常搬移
        for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
            FixpipeL0C2GMImplN(dstGlobal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.nSize, i);
        }
        // 尾块多搬一次
        if (fixpipeInfo.tiling.tailNSize > 0) {
            FixpipeL0C2GMImplN(dstGlobal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.tailNSize, fixpipeInfo.tiling.nIterNum);
        }
    }
}

// L0C->L1/UB deq tensor quant
template <typename DstT, typename SrcT>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParams<SrcT>& intriParams)
{
    if constexpr ((!IsSameType<SrcT, int32_t>::value) && (!IsSameType<SrcT, half>::value)) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe src data type only support fp16/s32 on this version!"); });
    } else if constexpr ((!IsSameType<DstT, int32_t>::value) && (!IsSameType<DstT, half>::value)&&
        (!IsSameType<DstT, int8_t>::value) && (!IsSameType<DstT, int16_t>::value)) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Fixpipe dst data type only support fp16/s8/s16/s32 on this version!"); });
    } else {
        const Hardware dstHWPos = GetPhyType((QuePosition)dstLocal.GetPosition());
        FixpipeInfoParams<SrcT> fixpipeInfo(intriParams, sizeof(SrcT), sizeof(DstT));
            // 这里获取的fixpipe tiling里面只有deq的最大size；
            // 存放workspace addr
        fixpipeInfo.cbufWorkspace = (__cbuf__ uint64_t *)cbufWorkspace.GetPhyAddr();
        if (dstHWPos == Hardware::UB) {
            // 整块正常搬移
            for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
                FixpipeL0C2UBImplN(dstLocal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.nSize, i);
            }
            // 尾块多搬一次
            if (fixpipeInfo.tiling.tailNSize > 0) {
                FixpipeL0C2UBImplN(dstLocal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.tailNSize, fixpipeInfo.tiling.nIterNum);
            }
        } else {
            // 整块正常搬移
            for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
                FixpipeL0C2L1ImplN(dstLocal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.nSize, i);
            }
            // 尾块多搬一次
            if (fixpipeInfo.tiling.tailNSize > 0) {
                FixpipeL0C2L1ImplN(dstLocal, srcLocal, fixpipeInfo, intriParams, fixpipeInfo.tiling.tailNSize, fixpipeInfo.tiling.nIterNum);
            }
        }
    }
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ DstT* dst, __cc__ SrcT* src, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ DstT* dst, __cc__ SrcT* src, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ DstT* dst, __cc__ SrcT* src, __cbuf__ uint64_t* cbufWorkspace,
    const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ DstT* dst, __cc__ SrcT* src, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ DstT* dst, __cc__ SrcT* src, __cbuf__ uint64_t* cbufWorkspace,
    const FixpipeParamsV220 &intriParams)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported fixpipe"); });
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
