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
 * \file kernel_utils_struct_dma_params.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_STRUCT_DMA_PARAMS_H
#define ASCENDC_MODULE_UTILS_STRUCT_DMA_PARAMS_H
#include "utils/kernel_utils_mode.h"

namespace AscendC {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
enum class eltwise_antiq_t {
    S8ANTIQ_Scalar = 0,  // s8 antiquant scalarģʽ
    S8ANTIQ_Vector,  // s8 antiquant vectorģʽ
    S4ANTIQ_Scalar,  // s4 antiquant scalarģʽ
    S4ANTIQ_Vector,  // s4 antiquant vectorģʽ
    U8ANTIQ_Scalar,  // u8 antiquant scalarģʽ
    U8ANTIQ_Vector,  // u8 antiquant vectorģʽ
    S16ANTIQ_Scalar,  // s16 antiquant scalarģʽ
    S16ANTIQ_Vector,  // s16 antiquant vectorģʽ
    NO_ANTIQ
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
#else
struct QuantParams {
    __aicore__ QuantParams() {}
    __aicore__ QuantParams(const QuantMode_t quantPreIn) : quantPre(quantPreIn) {}
    __aicore__ QuantParams(const QuantMode_t quantPreIn, const uint64_t deqScalarIn)
        : quantPre(quantPreIn), deqScalar(deqScalarIn) {}
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    uint64_t deqScalar;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
#if (__NPU_ARCH__ == 2103)
struct QuantParamsV210 {
    __aicore__ QuantParamsV210() {}
    __aicore__ QuantParamsV210(const ConvReluFix_t preQuantIn) : preQuantMode(preQuantIn) {}
    __aicore__ QuantParamsV210(const ConvReluFix_t preQuantIn, const uint64_t preScalarIn)
        : preQuantMode(preQuantIn), preScalarValue(preScalarIn) {}
    __aicore__ QuantParamsV210(const ConvReluFix_t preQuantIn, const uint64_t preScalarIn,
        const Req_t postQuantIn, const uint64_t postScalarIn)
        : preQuantMode(preQuantIn), preScalarValue(preScalarIn),
        postQuantMode(postQuantIn), postScalarValue(postScalarIn) {}
    ConvReluFix_t preQuantMode = ConvReluFix_t::CRFMODE_NONE;
    Req_t postQuantMode = Req_t::NoREQ;
    uint64_t preScalarValue = 0;
    uint64_t postScalarValue = 0;
};

using QuantParams = QuantParamsV210;
#endif

#if (__NPU_ARCH__ == 3003)
struct QuantParamsV300 {
    __aicore__ QuantParamsV300() {}
    __aicore__ QuantParamsV300(const QuantMode_t preQuantIn) : preQuantMode(preQuantIn) {}
    __aicore__ QuantParamsV300(const QuantMode_t preQuantIn, const uint64_t preScalarIn)
        : preQuantMode(preQuantIn), preScalarValue(preScalarIn) {}
    __aicore__ QuantParamsV300(const QuantMode_t preQuantIn, const uint64_t preScalarIn,
        const QuantMode_post postQuantIn, const uint64_t postScalarIn)
        : preQuantMode(preQuantIn), preScalarValue(preScalarIn),
        postQuantMode(postQuantIn), postScalarValue(postScalarIn) {}
    QuantMode_t preQuantMode = QuantMode_t::NoQuant;
    QuantMode_post postQuantMode = QuantMode_post::NoConv;
    uint64_t preScalarValue = 0;
    uint64_t postScalarValue = 0;
};

using QuantParams = QuantParamsV300;
#endif

#if (__NPU_ARCH__ == 3103)
struct QuantParamsV310 {
    __aicore__ QuantParamsV310() {}
    __aicore__ QuantParamsV310(const QuantMode_t preQuantIn) : preQuantMode(preQuantIn) {}
    __aicore__ QuantParamsV310(const QuantMode_t preQuantIn, const uint64_t preScalarIn)
        : preQuantMode(preQuantIn), preScalarValue(preScalarIn) {}
    __aicore__ QuantParamsV310(const QuantMode_t preQuantIn, const uint64_t preScalarIn,
        const QuantMode_post postQuantIn, const uint64_t postScalarIn)
        : preQuantMode(preQuantIn), preScalarValue(preScalarIn),
        postQuantMode(postQuantIn), postScalarValue(postScalarIn) {}
    QuantMode_t preQuantMode = QuantMode_t::NoQuant;
    QuantMode_post postQuantMode = QuantMode_post::NoConv;
    uint64_t preScalarValue = 0;
    uint64_t postScalarValue = 0;
};

using QuantParams = QuantParamsV310;
#endif

#if (__NPU_ARCH__ == 3113)
struct QuantParamsV311 {
    __aicore__ QuantParamsV311() {}
    __aicore__ QuantParamsV311(const QuantMode_t preQuantIn) : preQuantMode(preQuantIn) {}
    __aicore__ QuantParamsV311(const QuantMode_t preQuantIn, const uint64_t preScalarIn)
        : preQuantMode(preQuantIn), preScalarValue(preScalarIn) {}
    __aicore__ QuantParamsV311(const QuantMode_t preQuantIn, const uint64_t preScalarIn,
        const QuantMode_post postQuantIn, const uint64_t postScalarIn)
        : preQuantMode(preQuantIn), preScalarValue(preScalarIn),
        postQuantMode(postQuantIn), postScalarValue(postScalarIn) {}
    QuantMode_t preQuantMode = QuantMode_t::NoQuant;
    QuantMode_post postQuantMode = QuantMode_post::NoConv;
    uint64_t preScalarValue = 0;
    uint64_t postScalarValue = 0;
};

using QuantParams = QuantParamsV311;
#endif

#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
struct Nz2NdParams {
    __aicore__ Nz2NdParams() {}

    __aicore__ Nz2NdParams(const uint16_t ndNumIn, const uint16_t srcNdStrideIn, const uint32_t dstNdStrideIn)
    {
        ndNum = ndNumIn;
        srcNdStride = srcNdStrideIn;
        dstNdStride = dstNdStrideIn;
    }

    uint16_t ndNum = 1; // loop3Size
    uint16_t srcNdStride = 0; // loop3SrcStride
    uint32_t dstNdStride = 0; // loop3DstStride
};
#else
struct Nz2NdParams {
    __aicore__ Nz2NdParams()
    {
        nz2ndEn = false;
        ndNum = 1;
        srcNdStride = 0;
        dstNdStride = 0;
        originalNSize = 0;
    }

    __aicore__ Nz2NdParams(const bool nz2ndEnIn, const uint16_t ndNumIn, const uint16_t srcNdStrideIn,
        const uint16_t dstNdStrideIn, const uint16_t originalNSizeIn)
    {
        nz2ndEn = nz2ndEnIn;
        ndNum = ndNumIn;
        srcNdStride = srcNdStrideIn;
        dstNdStride = dstNdStrideIn;
        originalNSize = originalNSizeIn;
    }

    bool nz2ndEn = false;
    uint16_t ndNum = 1;
    uint16_t srcNdStride = 0;
    uint16_t dstNdStride = 0;
    uint16_t originalNSize = 0;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
struct Nz2DnParams {
    __aicore__ Nz2DnParams() {}

    __aicore__ Nz2DnParams(const uint16_t dnNumIn, const uint16_t srcNzMatrixStrideIn,
        const uint32_t dstDnMatrixStrideIn, const uint16_t srcNzC0StrideIn)
    {
        dnNum = dnNumIn;
        srcNzMatrixStride = srcNzMatrixStrideIn;
        dstDnMatrixStride = dstDnMatrixStrideIn;
        srcNzC0Stride = srcNzC0StrideIn;
    }

    uint16_t dnNum = 1; // loop3Size
    uint16_t srcNzMatrixStride = 0; // loop3SrcStride
    uint32_t dstDnMatrixStride = 0; // loop3DstStride
    uint16_t srcNzC0Stride = 0; // loop0SrcStride
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
#else
template <typename T = int32_t>
struct FixpipeParams {
    __aicore__ FixpipeParams()
    {
        cburstNum = DEFAULT_DATA_COPY_NBURST;
        burstLen = 1;
        srcStride = DEFAULT_DATA_COPY_STRIDE;
        dstStride = DEFAULT_DATA_COPY_STRIDE;
        reluEn = false;
        unitFlag = 0;
    }

    __aicore__ FixpipeParams(const uint16_t count, const uint16_t len, const uint16_t srcStrideIn,
        const uint32_t dstStrideIn)
    {
        cburstNum = count;
        burstLen = len;
        dstStride = dstStrideIn;
        srcStride = srcStrideIn;
    }

    uint16_t cburstNum = 0;
    uint16_t burstLen = 0;
    uint32_t dstStride = 0;
    uint16_t srcStride = 0;
    // extend param
    QuantParams quantParams;
    bool reluEn = false;
    Nz2NdParams nz2ndParams;
    uint8_t unitFlag = 0;
};
#endif

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) || \
    (__NPU_ARCH__ == 3113))
#if (__NPU_ARCH__ == 2103)
template <typename T = int32_t>
struct FixpipeParamsV210 {
    __aicore__ FixpipeParamsV210()
    {
        unitFlag = unit_flag_t::UFMode0;
        eltwiseOp = eltwise_op_t::No_Eltwise;
        nSize = 0;
        mSize = 0;
        srcBurstGap = 0;
        dstBurstGap = 0;
        biasEnable = false;
        preReluMode = Relu_t::NoRELU;
        eltwiseEnable = false;
        postReluMode = Relu_t::NoRELU;
        poolMode = Pool_t::NoPooling;
        dualMode = DualMode_t::DUAL_MODE0;
        ws = 0;
        wSize = 0;
        quantCfg = 0;
    }

    __aicore__ FixpipeParamsV210(const uint16_t nSize_, const uint16_t mSize_, const uint16_t srcBurstGap_,
        const uint32_t dstBurstGap_)
    {
        nSize = nSize_;
        mSize = mSize_;
        srcBurstGap = srcBurstGap_;
        dstBurstGap = dstBurstGap_;
    }

    __aicore__ inline void SetUnitFlag(const unit_flag_t unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    __aicore__ inline void SetEltwiseOp(const eltwise_op_t eltwiseOp_)
    {
        eltwiseOp = eltwiseOp_;
    }

    __aicore__ inline void SetNSize(const uint16_t nSize_)
    {
        nSize = nSize_;
    }

    __aicore__ inline void SetMSize(const uint16_t mSize_)
    {
        mSize = mSize_;
    }

    __aicore__ inline void SetSrcBurstGap(const uint16_t srcBurstGap_)
    {
        srcBurstGap = srcBurstGap_;
    }

    __aicore__ inline void SetDstBurstGap(const uint16_t dstBurstGap_)
    {
        dstBurstGap = dstBurstGap_;
    }

    __aicore__ inline void SetPreQuantMode(const ConvReluFix_t preQuantMode_)
    {
        quantParams.preQuantMode = preQuantMode_;
    }

    __aicore__ inline void SetPostQuantMode(const Req_t postQuantMode_)
    {
        quantParams.postQuantMode = postQuantMode_;
    }

    __aicore__ inline void SetBiasEnable(const bool biasEnable_)
    {
        biasEnable = biasEnable_;
    }

    __aicore__ inline void SetPreReluMode(const Relu_t preReluMode_)
    {
        preReluMode = preReluMode_;
    }

    __aicore__ inline void SetEltwiseEnable(const bool eltwiseEnable_)
    {
        eltwiseEnable = eltwiseEnable_;
    }

    __aicore__ inline void SetPostReluMode(const Relu_t postReluMode_)
    {
        postReluMode = postReluMode_;
    }

    __aicore__ inline void SetPoolMode(const Pool_t poolMode_)
    {
        poolMode = poolMode_;
    }

    __aicore__ inline void SetDualMode(const DualMode_t dualMode_)
    {
        dualMode = dualMode_;
    }

    __aicore__ inline void SetWs(const uint16_t ws_)
    {
        ws = ws_;
    }

    __aicore__ inline void SetWSize(const uint16_t WSize_)
    {
        wSize = WSize_;
    }

    __aicore__ inline void SetQuantCfg(const uint64_t quantCfg_)
    {
        quantCfg = quantCfg_;
    }

    unit_flag_t unitFlag = unit_flag_t::UFMode0;
    eltwise_op_t eltwiseOp = eltwise_op_t::No_Eltwise;
    uint16_t nSize = 0;
    uint16_t mSize = 0;
    uint16_t srcBurstGap = 0;
    uint16_t dstBurstGap = 0;
    QuantParams quantParams;
    bool biasEnable = false;
    Relu_t preReluMode = Relu_t::NoRELU;
    bool eltwiseEnable = false;
    Relu_t postReluMode = Relu_t::NoRELU;
    Pool_t poolMode = Pool_t::NoPooling;
    DualMode_t dualMode = DualMode_t::DUAL_MODE0;
    uint16_t ws = 0;
    uint16_t wSize = 0;
    uint64_t quantCfg = 0;
};
#endif

#if (__NPU_ARCH__ == 3003)
template <typename T = int32_t>
struct FixpipeParamsV300 {
    __aicore__ FixpipeParamsV300()
    {
        nSize = 0;
        mSize = 0;
        srcStride = 0;
        dstStride = 0;
        preReluMode = ReluMode_t::NoRelu;
        postReluMode = ReluMode_t::NoRelu;
        preClipReluMode = ClipReluMode_t::NoClipRelu;
        postClipReluMode = ClipReluMode_t::NoClipRelu;
        eltwiseOp = eltwise_op_t::No_Eltwise;
        eltwiseAntiqMode = eltwise_antiq_t::NO_ANTIQ;
        loopEnhanceEnable = false;
        loopEnhanceMergeEnable = false;
        unitFlag = 0;
        c0PadEnable = false;
        postWinoEnable = false;
        channelSplitEnable = false;
        nz2ndEnable = false;
        quantCfg = 0;
    }

    __aicore__ FixpipeParamsV300(const uint16_t nSize_, const uint16_t mSize_, const uint16_t srcStride_,
        const uint32_t dstStride_)
    {
        nSize = nSize_;
        mSize = mSize_;
        srcStride = srcStride_;
        dstStride = dstStride_;
    }

    __aicore__ inline void SetNSize(const uint16_t nSize_)
    {
        nSize = nSize_;
    }

    __aicore__ inline void SetMSize(const uint16_t mSize_)
    {
        mSize = mSize_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint32_t dstStride_)
    {
        dstStride = dstStride_;
    }

    __aicore__ inline void SetPreQuantMode(const QuantMode_t preQuantMode_)
    {
        quantParams.preQuantMode = preQuantMode_;
    }

    __aicore__ inline void SetPostQuantMode(const QuantMode_post postQuantMode_)
    {
        quantParams.postQuantMode = postQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(const ReluMode_t preReluMode_)
    {
        preReluMode = preReluMode_;
    }

    __aicore__ inline void SetPostReluMode(const ReluMode_t postReluMode_)
    {
        postReluMode = postReluMode_;
    }

    __aicore__ inline void SetPreClipReluMode(const ClipReluMode_t preClipReluMode_)
    {
        preClipReluMode = preClipReluMode_;
    }

    __aicore__ inline void SetPostClipReluMode(const ClipReluMode_t postClipReluMode_)
    {
        postClipReluMode = postClipReluMode_;
    }

    __aicore__ inline void SetEltwiseOp(const eltwise_op_t eltwiseOp_)
    {
        eltwiseOp = eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(const bool eltwiseAntiqEnable_)
    {
        (void)eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseAntiqMode(const eltwise_antiq_t eltwiseAntiqMode_)
    {
        eltwiseAntiqMode = eltwiseAntiqMode_;
    }

    __aicore__ inline void SetBroadCastEnable(const bool broadCastEnable_)
    {
        (void)broadCastEnable_;
    }

    __aicore__ inline void SetLoopEnhanceEnable(const bool loopEnhanceEnable_)
    {
        loopEnhanceEnable = loopEnhanceEnable_;
    }

    __aicore__ inline void SetLoopEnhanceMergeEnable(const bool loopEnhanceMergeEnable_)
    {
        loopEnhanceMergeEnable = loopEnhanceMergeEnable_;
    }

    __aicore__ inline void SetUnitFlag(const uint8_t unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    __aicore__ inline void SetC0PadEnable(const bool c0PadEnable_)
    {
        c0PadEnable = c0PadEnable_;
    }

    __aicore__ inline void SetPostWinoEnable(const bool postWinoEnable_)
    {
        postWinoEnable = postWinoEnable_;
    }

    __aicore__ inline void SetChannelSplitEnable(const bool channelSplitEnable_)
    {
        channelSplitEnable = channelSplitEnable_;
    }

    __aicore__ inline void SetNz2ndEnable(const bool nz2ndEnable_)
    {
        nz2ndEnable = nz2ndEnable_;
    }

    __aicore__ inline void SetNz2dnEnable(const bool nz2dnEnable_)
    {
        (void)nz2dnEnable_;
    }

    __aicore__ inline void SetPreScalarValue(const uint64_t preScalarValue_)
    {
        (void)preScalarValue_;
    }

    __aicore__ inline void SetPostScalarValue(const uint64_t postScalarValue_)
    {
        (void)postScalarValue_;
    }

    __aicore__ inline void SetcBurstNum(const uint16_t cburstNum_)
    {
        (void)cburstNum_;
    }

    __aicore__ inline void SetBurstLen(const uint16_t burstLen_)
    {
        (void)burstLen_;
    }

    __aicore__ inline void SetQuantCfg(const uint64_t quantCfg_)
    {
        quantCfg = quantCfg_;
    }

    uint16_t nSize = 0;
    uint16_t mSize = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    QuantParams quantParams;
    ReluMode_t preReluMode = ReluMode_t::NoRelu;
    ReluMode_t postReluMode = ReluMode_t::NoRelu;
    ClipReluMode_t preClipReluMode = ClipReluMode_t::NoClipRelu;
    ClipReluMode_t postClipReluMode = ClipReluMode_t::NoClipRelu;
    eltwise_op_t eltwiseOp = eltwise_op_t::No_Eltwise;
    eltwise_antiq_t eltwiseAntiqMode = eltwise_antiq_t::NO_ANTIQ;
    bool loopEnhanceEnable = false;
    bool loopEnhanceMergeEnable = false;
    uint8_t unitFlag = 0;
    bool c0PadEnable = false;
    bool postWinoEnable = false;
    bool channelSplitEnable = false;
    Nz2NdParams nz2ndParams;
    bool nz2ndEnable = false;
    uint64_t quantCfg = 0;
};
#endif

#if (__NPU_ARCH__ == 3103) 
template <typename T = int32_t>
struct FixpipeParamsV310 {
    __aicore__ FixpipeParamsV310()
    {
        nSize = 0;
        mSize = 0;
        srcStride = 0;
        dstStride = 0;
        preReluMode = ReluMode_t::NoRelu;
        postReluMode = ReluMode_t::NoRelu;
        preClipReluMode = ClipReluMode_t::NoClipRelu;
        postClipReluMode = ClipReluMode_t::NoClipRelu;
        eltwiseOp = eltwise_op_t::No_Eltwise;
        eltwiseAntiqEnable = false;
        broadCastEnable = false;
        loopEnhanceEnable = false;
        loopEnhanceMergeEnable = false;
        unitFlag = 0;
        c0PadEnable = false;
        postWinoEnable = false;
        channelSplitEnable = false;
        nz2ndEnable = false;
        quantCfg = 0;
    }

    __aicore__ FixpipeParamsV310(const uint16_t nSize_, const uint16_t mSize_, const uint16_t srcStride_,
        const uint32_t dstStride_)
    {
        nSize = nSize_;
        mSize = mSize_;
        srcStride = srcStride_;
        dstStride = dstStride_;
    }

    __aicore__ inline void SetNSize(const uint16_t nSize_)
    {
        nSize = nSize_;
    }

    __aicore__ inline void SetMSize(const uint16_t mSize_)
    {
        mSize = mSize_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint32_t dstStride_)
    {
        dstStride = dstStride_;
    }

    __aicore__ inline void SetPreQuantMode(const QuantMode_t preQuantMode_)
    {
        quantParams.preQuantMode = preQuantMode_;
    }

    __aicore__ inline void SetPostQuantMode(const QuantMode_post postQuantMode_)
    {
        quantParams.postQuantMode = postQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(const ReluMode_t preReluMode_)
    {
        preReluMode = preReluMode_;
    }

    __aicore__ inline void SetPostReluMode(const ReluMode_t postReluMode_)
    {
        postReluMode = postReluMode_;
    }

    __aicore__ inline void SetPreClipReluMode(const ClipReluMode_t preClipReluMode_)
    {
        preClipReluMode = preClipReluMode_;
    }

    __aicore__ inline void SetPostClipReluMode(const ClipReluMode_t postClipReluMode_)
    {
        postClipReluMode = postClipReluMode_;
    }

    __aicore__ inline void SetEltwiseOp(const eltwise_op_t eltwiseOp_)
    {
        eltwiseOp = eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(const bool eltwiseAntiqEnable_)
    {
        eltwiseAntiqEnable = eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseAntiqMode(const eltwise_antiq_t eltwiseAntiqMode_)
    {
        (void)eltwiseAntiqMode_;
    }

    __aicore__ inline void SetBroadCastEnable(const bool broadCastEnable_)
    {
        broadCastEnable = broadCastEnable_;
    }

    __aicore__ inline void SetLoopEnhanceEnable(const bool loopEnhanceEnable_)
    {
        loopEnhanceEnable = loopEnhanceEnable_;
    }

    __aicore__ inline void SetLoopEnhanceMergeEnable(const bool loopEnhanceMergeEnable_)
    {
        loopEnhanceMergeEnable = loopEnhanceMergeEnable_;
    }

    __aicore__ inline void SetUnitFlag(const uint8_t unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    __aicore__ inline void SetC0PadEnable(const bool c0PadEnable_)
    {
        c0PadEnable = c0PadEnable_;
    }

    __aicore__ inline void SetPostWinoEnable(const bool postWinoEnable_)
    {
        postWinoEnable = postWinoEnable_;
    }

    __aicore__ inline void SetChannelSplitEnable(const bool channelSplitEnable_)
    {
        channelSplitEnable = channelSplitEnable_;
    }

    __aicore__ inline void SetNz2ndEnable(const bool nz2ndEnable_)
    {
        nz2ndEnable = nz2ndEnable_;
    }

    __aicore__ inline void SetNz2dnEnable(const bool nz2dnEnable_)
    {
        nz2dnEnable = nz2dnEnable_;
    }

    __aicore__ inline void SetPreScalarValue(const uint64_t preScalarValue_)
    {
        (void)preScalarValue_;
    }

    __aicore__ inline void SetPostScalarValue(const uint64_t postScalarValue_)
    {
        (void)postScalarValue_;
    }

    __aicore__ inline void SetcBurstNum(const uint16_t cburstNum_)
    {
        (void)cburstNum_;
    }

    __aicore__ inline void SetBurstLen(const uint16_t burstLen_)
    {
        (void)burstLen_;
    }

    __aicore__ inline void SetQuantCfg(const uint64_t quantCfg_)
    {
        quantCfg = quantCfg_;
    }

    uint16_t nSize = 0;
    uint16_t mSize = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    QuantParams quantParams;
    ReluMode_t preReluMode = ReluMode_t::NoRelu;
    ReluMode_t postReluMode = ReluMode_t::NoRelu;
    ClipReluMode_t preClipReluMode = ClipReluMode_t::NoClipRelu;
    ClipReluMode_t postClipReluMode = ClipReluMode_t::NoClipRelu;
    eltwise_op_t eltwiseOp = eltwise_op_t::No_Eltwise;
    bool eltwiseAntiqEnable = false;
    bool broadCastEnable = false;
    bool loopEnhanceEnable = false;
    bool loopEnhanceMergeEnable = false;
    uint8_t unitFlag = 0;
    bool c0PadEnable = false;
    bool postWinoEnable = false;
    bool channelSplitEnable = false;
    Nz2NdParams nz2ndParams;
    bool nz2ndEnable = false;
    bool nz2dnEnable = false;
    uint64_t quantCfg = 0;
};
#endif

#if (__NPU_ARCH__ == 3113)
template <typename T = int32_t>
struct FixpipeParamsV311Gen {
    __aicore__ FixpipeParamsV311Gen()
    {
        nSize = 0;
        mSize = 0;
        srcStride = 0;
        dstStride = 0;
        preReluMode = ReluMode_t::NoRelu;
        postReluMode = ReluMode_t::NoRelu;
        preClipReluMode = ClipReluMode_t::NoClipRelu;
        postClipReluMode = ClipReluMode_t::NoClipRelu;
        eltwiseOp = eltwise_op_t::No_Eltwise;
        eltwiseAntiqEnable = false;
        broadCastEnable = false;
        loopEnhanceEnable = false;
        loopEnhanceMergeEnable = false;
        unitFlag = 0;
        c0PadEnable = false;
        postWinoEnable = false;
        channelSplitEnable = false;
        nz2ndEnable = false;
        nz2dnEnable = false;
        quantCfg = 0;
    }

    __aicore__ FixpipeParamsV311Gen(const uint16_t nSize_, const uint16_t mSize_, const uint16_t srcStride_,
        const uint32_t dstStride_)
    {
        nSize = nSize_;
        mSize = mSize_;
        srcStride = srcStride_;
        dstStride = dstStride_;
    }

    __aicore__ inline void SetNSize(const uint16_t nSize_)
    {
        nSize = nSize_;
    }

    __aicore__ inline void SetMSize(const uint16_t mSize_)
    {
        mSize = mSize_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint32_t dstStride_)
    {
        dstStride = dstStride_;
    }

    __aicore__ inline void SetPreQuantMode(const QuantMode_t preQuantMode_)
    {
        quantParams.preQuantMode = preQuantMode_;
    }

    __aicore__ inline void SetPostQuantMode(const QuantMode_post postQuantMode_)
    {
        quantParams.postQuantMode = postQuantMode_;
    }

    __aicore__ inline void SetPreReluMode(const ReluMode_t preReluMode_)
    {
        preReluMode = preReluMode_;
    }

    __aicore__ inline void SetPostReluMode(const ReluMode_t postReluMode_)
    {
        postReluMode = postReluMode_;
    }

    __aicore__ inline void SetPreClipReluMode(const ClipReluMode_t preClipReluMode_)
    {
        preClipReluMode = preClipReluMode_;
    }

    __aicore__ inline void SetPostClipReluMode(const ClipReluMode_t postClipReluMode_)
    {
        postClipReluMode = postClipReluMode_;
    }

    __aicore__ inline void SetEltwiseOp(const eltwise_op_t eltwiseOp_)
    {
        eltwiseOp = eltwiseOp_;
    }

    __aicore__ inline void SetEltwiseAntiqEnable(const bool eltwiseAntiqEnable_)
    {
        eltwiseAntiqEnable = eltwiseAntiqEnable_;
    }

    __aicore__ inline void SetEltwiseAntiqMode(const eltwise_antiq_t eltwiseAntiqMode_)
    {
        (void)eltwiseAntiqMode_;
    }

    __aicore__ inline void SetBroadCastEnable(const bool broadCastEnable_)
    {
        broadCastEnable = broadCastEnable_;
    }

    __aicore__ inline void SetLoopEnhanceEnable(const bool loopEnhanceEnable_)
    {
        loopEnhanceEnable = loopEnhanceEnable_;
    }

    __aicore__ inline void SetLoopEnhanceMergeEnable(const bool loopEnhanceMergeEnable_)
    {
        loopEnhanceMergeEnable = loopEnhanceMergeEnable_;
    }

    __aicore__ inline void SetUnitFlag(const uint8_t unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    __aicore__ inline void SetC0PadEnable(const bool c0PadEnable_)
    {
        c0PadEnable = c0PadEnable_;
    }

    __aicore__ inline void SetPostWinoEnable(const bool postWinoEnable_)
    {
        postWinoEnable = postWinoEnable_;
    }

    __aicore__ inline void SetChannelSplitEnable(const bool channelSplitEnable_)
    {
        channelSplitEnable = channelSplitEnable_;
    }

    __aicore__ inline void SetNz2ndEnable(const bool nz2ndEnable_)
    {
        nz2ndEnable = nz2ndEnable_;
    }

    __aicore__ inline void SetNz2dnEnable(const bool nz2dnEnable_)
    {
        nz2dnEnable = nz2dnEnable_;
    }

    __aicore__ inline void SetPreScalarValue(const uint64_t preScalarValue_)
    {
        (void)preScalarValue_;
    }

    __aicore__ inline void SetPostScalarValue(const uint64_t postScalarValue_)
    {
        (void)postScalarValue_;
    }

    __aicore__ inline void SetcBurstNum(const uint16_t cburstNum_)
    {
        (void)cburstNum_;
    }

    __aicore__ inline void SetBurstLen(const uint16_t burstLen_)
    {
        (void)burstLen_;
    }

    __aicore__ inline void SetQuantCfg(const uint64_t quantCfg_)
    {
        quantCfg = quantCfg_;
    }

    uint16_t nSize = 0;
    uint16_t mSize = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    QuantParams quantParams;
    ReluMode_t preReluMode = ReluMode_t::NoRelu;
    ReluMode_t postReluMode = ReluMode_t::NoRelu;
    ClipReluMode_t preClipReluMode = ClipReluMode_t::NoClipRelu;
    ClipReluMode_t postClipReluMode = ClipReluMode_t::NoClipRelu;
    eltwise_op_t eltwiseOp = eltwise_op_t::No_Eltwise;
    bool eltwiseAntiqEnable = false;
    bool broadCastEnable = false;
    bool loopEnhanceEnable = false;
    bool loopEnhanceMergeEnable = false;
    uint8_t unitFlag = 0;
    bool c0PadEnable = false;
    bool postWinoEnable = false;
    bool channelSplitEnable = false;
    Nz2NdParams nz2ndParams;
    bool nz2ndEnable = false;
    bool nz2dnEnable = false;
    uint64_t quantCfg = 0;
};
#endif

#if (__NPU_ARCH__ == 2103)
template <typename T>
using FixpipeParams = FixpipeParamsV210<T>;
#endif

#if (__NPU_ARCH__ == 3003)
template <typename T>
using FixpipeParams = FixpipeParamsV300<T>;
#endif

#if (__NPU_ARCH__ == 3103) 
template <typename T>
using FixpipeParams = FixpipeParamsV310<T>;
#endif

#if ((__NPU_ARCH__ == 3113))
template <typename T>
using FixpipeParams = FixpipeParamsV311Gen<T>;
#endif

struct FixPipeConfigParamsV210 {
    __aicore__ FixPipeConfigParamsV210()
    {
        relubBiasAddr = 0;
        dequantAddr = 2048;
        reluaRequantAddr = 4096;
        upsamplingCo = 0;
        upsamplingParam = 0;
        avgPoolingInitEnable = false;
        avgPoolingWrittenEnable = false;
        convReluMode = ReluMode_t::NormalRelu;
        unitFlag = false;
    }

    __aicore__ inline void SetRelubBiasAddr(const uint32_t relubBiasAddr_)
    {
        relubBiasAddr = relubBiasAddr_;
    }

    __aicore__ inline void SetDequantAddr(const uint32_t dequantAddr_)
    {
        dequantAddr = dequantAddr_;
    }

    __aicore__ inline void SetReluaRequantAddr(const uint32_t reluaRequantAddr_)
    {
        reluaRequantAddr = reluaRequantAddr_;
    }

    __aicore__ inline void SetUpsamplingCo(const uint32_t upsamplingCo_)
    {
        upsamplingCo = upsamplingCo_;
    }

    __aicore__ inline void SetUpsamplingParam(const uint32_t upsamplingParam_)
    {
        upsamplingParam = upsamplingParam_;
    }

    __aicore__ inline void SetAvgPoolingInitEnable(const bool avgPoolingInitEnable_)
    {
        avgPoolingInitEnable = avgPoolingInitEnable_;
    }

    __aicore__ inline void SetAvgPoolingWrittenEnable(const bool avgPoolingWrittenEnable_)
    {
        avgPoolingWrittenEnable = avgPoolingWrittenEnable_;
    }

    __aicore__ inline void SetConvReluMode(const ReluMode_t convReluMode_)
    {
        convReluMode = convReluMode_;
    }

    __aicore__ inline void SetUnitFlag(const bool unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    uint32_t relubBiasAddr = 0;
    uint32_t dequantAddr = 2048;
    uint32_t reluaRequantAddr = 4096;
    uint32_t upsamplingCo = 0;
    uint32_t upsamplingParam = 0;
    bool avgPoolingInitEnable = false;
    bool avgPoolingWrittenEnable = false;
    ReluMode_t convReluMode = ReluMode_t::NormalRelu;
    bool unitFlag = false;
};

struct FixPipeConfigParamsV300 {
    __aicore__ FixPipeConfigParamsV300()
    {
        preReluAddr = 65536;
        preQuantAddr = 0;
        postReluAddr = 131072;
        postQuantAddr = 196608;
        unitFlag = false;
    }

    __aicore__ inline void SetPreReluAddr(const uint32_t preReluAddr_)
    {
        preReluAddr = preReluAddr_;
    }

    __aicore__ inline void SetPreQuantAddr(const uint32_t preQuantAddr_)
    {
        preQuantAddr = preQuantAddr_;
    }

    __aicore__ inline void SetPostReluAddr(const uint32_t postReluAddr_)
    {
        postReluAddr = postReluAddr_;
    }

    __aicore__ inline void SetPostQuantAddr(const uint32_t postQuantAddr_)
    {
        postQuantAddr = postQuantAddr_;
    }

    __aicore__ inline void SetAntiquantAddr(const uint32_t antiquantAddr_)
    {
        (void)antiquantAddr_;
    }

    __aicore__ inline void SetUnitFlag(const bool unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    uint32_t preReluAddr = 65536;
    uint32_t preQuantAddr = 0;
    uint32_t postReluAddr = 131072;
    uint32_t postQuantAddr = 196608;
    bool unitFlag = false;
};

struct FixPipeConfigParamsV310Gen {
    __aicore__ FixPipeConfigParamsV310Gen()
    {
        preReluAddr = 65536;
        preQuantAddr = 0;
        postReluAddr = 131072;
        postQuantAddr = 196608;
        antiquantAddr = 262144;
        unitFlag = false;
    }

    __aicore__ inline void SetPreReluAddr(const uint32_t preReluAddr_)
    {
        preReluAddr = preReluAddr_;
    }

    __aicore__ inline void SetPreQuantAddr(const uint32_t preQuantAddr_)
    {
        preQuantAddr = preQuantAddr_;
    }

    __aicore__ inline void SetPostReluAddr(const uint32_t postReluAddr_)
    {
        postReluAddr = postReluAddr_;
    }

    __aicore__ inline void SetPostQuantAddr(const uint32_t postQuantAddr_)
    {
        postQuantAddr = postQuantAddr_;
    }

    __aicore__ inline void SetAntiquantAddr(const uint32_t antiquantAddr_)
    {
        antiquantAddr = antiquantAddr_;
    }

    __aicore__ inline void SetUnitFlag(const bool unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    uint32_t preReluAddr = 65536;
    uint32_t preQuantAddr = 0;
    uint32_t postReluAddr = 131072;
    uint32_t postQuantAddr = 196608;
    uint32_t antiquantAddr = 262144;
    bool unitFlag = false;
};


struct FixPipeConfigParamsV311Gen {
    __aicore__ FixPipeConfigParamsV311Gen()
    {
        preReluAddr = 65536;
        preQuantAddr = 0;
        postReluAddr = 131072;
        postQuantAddr = 196608;
        antiquantAddr = 262144;
        unitFlag = false;
    }

    __aicore__ inline void SetPreReluAddr(const uint32_t preReluAddr_)
    {
        preReluAddr = preReluAddr_;
    }

    __aicore__ inline void SetPreQuantAddr(const uint32_t preQuantAddr_)
    {
        preQuantAddr = preQuantAddr_;
    }

    __aicore__ inline void SetPostReluAddr(const uint32_t postReluAddr_)
    {
        postReluAddr = postReluAddr_;
    }

    __aicore__ inline void SetPostQuantAddr(const uint32_t postQuantAddr_)
    {
        postQuantAddr = postQuantAddr_;
    }

    __aicore__ inline void SetAntiquantAddr(const uint32_t antiquantAddr_)
    {
        antiquantAddr = antiquantAddr_;
    }

    __aicore__ inline void SetUnitFlag(const bool unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    uint32_t preReluAddr = 65536;
    uint32_t preQuantAddr = 0;
    uint32_t postReluAddr = 131072;
    uint32_t postQuantAddr = 196608;
    uint32_t antiquantAddr = 262144;
    bool unitFlag = false;
};



#if ((__NPU_ARCH__ == 3113))
using FixPipeConfigParams = FixPipeConfigParamsV311Gen;
#elif ((__NPU_ARCH__ == 3103))
using FixPipeConfigParams = FixPipeConfigParamsV310Gen;
#elif (__NPU_ARCH__ == 3003)
using FixPipeConfigParams = FixPipeConfigParamsV300;
#elif (__NPU_ARCH__ == 2103)
using FixPipeConfigParams = FixPipeConfigParamsV210;
#else
struct FixPipeConfigParams {
    __aicore__ FixPipeConfigParams()
    {
        preReluAddr = 65536;
        preQuantAddr = 0;
        postReluAddr = 131072;
        postQuantAddr = 196608;
        antiquantAddr = 262144;
        unitFlag = false;
    }

    __aicore__ inline void SetPreReluAddr(const uint32_t preReluAddr_)
    {
        preReluAddr = preReluAddr_;
    }

    __aicore__ inline void SetPreQuantAddr(const uint32_t preQuantAddr_)
    {
        preQuantAddr = preQuantAddr_;
    }

    __aicore__ inline void SetPostReluAddr(const uint32_t postReluAddr_)
    {
        postReluAddr = postReluAddr_;
    }

    __aicore__ inline void SetPostQuantAddr(const uint32_t postQuantAddr_)
    {
        postQuantAddr = postQuantAddr_;
    }

    __aicore__ inline void SetAntiquantAddr(const uint32_t antiquantAddr_)
    {
        antiquantAddr = antiquantAddr_;
    }

    __aicore__ inline void SetUnitFlag(const bool unitFlag_)
    {
        unitFlag = unitFlag_;
    }

    uint32_t preReluAddr = 65536;
    uint32_t preQuantAddr = 0;
    uint32_t postReluAddr = 131072;
    uint32_t postQuantAddr = 196608;
    uint32_t antiquantAddr = 262144;
    bool unitFlag = false;
};
#endif

template<typename T = int16_t>
struct FixPipePreQuantParamsV300 {
    __aicore__ FixPipePreQuantParamsV300()
    {
        offset0 = 0;
        scalarValue = 0.0;
        offset1 = 0;
        offset2 = 0;
        isSigned = false;
    }

    __aicore__ inline void SetOffset0(int32_t offset0_)
    {
        offset0 = offset0_;
    }

    __aicore__ inline void SetF16Multiplier(float16_t f16Multiplier_) {
    }

    __aicore__ inline void SetScalarValue(float32_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetOffset1(int8_t offset1_)
    {
        offset1 = offset1_;
    }

    __aicore__ inline void SetOffset2(int16_t offset2_)
    {
        offset2 = offset2_;
    }

    __aicore__ inline void SetIsSigned(bool isSigned_)
    {
        isSigned = isSigned_;
    }

    __aicore__ inline void SetBitMaskNum(lsb_mask_t bitMaskNum_) {
    }

    __aicore__ inline void SetClipMaxValPre(T clipMaxValPre_) {
    }

    int32_t offset0 = 0;
    float32_t scalarValue = 0.0;
    int8_t offset1 = 0;
    int16_t offset2 = 0;
    bool isSigned = false;
};

template<typename T = int16_t>
struct FixPipePreQuantParamsV310Gen {
    __aicore__ FixPipePreQuantParamsV310Gen()
    {
        offset0 = 0;
        f16Multiplier = 0.0;
        scalarValue = 0.0;
        offset1 = 0;
        offset2 = 0;
        isSigned = false;
        bitMaskNum = lsb_mask_t::Disable;
    }

    __aicore__ inline void SetOffset0(int32_t offset0_)
    {
        offset0 = offset0_;
    }

    __aicore__ inline void SetF16Multiplier(float16_t f16Multiplier_)
    {
        f16Multiplier = f16Multiplier_;
    }

    __aicore__ inline void SetScalarValue(float32_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetOffset1(int8_t offset1_)
    {
        offset1 = offset1_;
    }

    __aicore__ inline void SetOffset2(int16_t offset2_)
    {
        offset2 = offset2_;
    }

    __aicore__ inline void SetIsSigned(bool isSigned_)
    {
        isSigned = isSigned_;
    }

    __aicore__ inline void SetBitMaskNum(lsb_mask_t bitMaskNum_)
    {
        bitMaskNum = bitMaskNum_;
    }

    __aicore__ inline void SetClipMaxValPre(T clipMaxValPre_)
    {
        (void)clipMaxValPre_;
    }

    int32_t offset0 = 0;
    float16_t f16Multiplier = 0.0;
    float32_t scalarValue = 0.0;
    int8_t offset1 = 0;
    int16_t offset2 = 0;
    bool isSigned = false;
    lsb_mask_t bitMaskNum = lsb_mask_t::Disable;
};


template<typename T = int16_t>
struct FixPipePreQuantParamsV311Gen {
    __aicore__ FixPipePreQuantParamsV311Gen()
    {
        offset0 = 0;
        f16Multiplier = 0.0;
        scalarValue = 0.0;
        offset1 = 0;
        offset2 = 0;
        isSigned = false;
        bitMaskNum = lsb_mask_t::Disable;
    }

    __aicore__ inline void SetOffset0(int32_t offset0_)
    {
        offset0 = offset0_;
    }

    __aicore__ inline void SetF16Multiplier(float16_t f16Multiplier_)
    {
        f16Multiplier = f16Multiplier_;
    }

    __aicore__ inline void SetScalarValue(float32_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetOffset1(int8_t offset1_)
    {
        offset1 = offset1_;
    }

    __aicore__ inline void SetOffset2(int16_t offset2_)
    {
        offset2 = offset2_;
    }

    __aicore__ inline void SetIsSigned(bool isSigned_)
    {
        isSigned = isSigned_;
    }

    __aicore__ inline void SetBitMaskNum(lsb_mask_t bitMaskNum_)
    {
        bitMaskNum = bitMaskNum_;
    }

    __aicore__ inline void SetClipMaxValPre(T clipMaxValPre_)
    {
        (void)clipMaxValPre_;
    }

    int32_t offset0 = 0;
    float16_t f16Multiplier = 0.0;
    float32_t scalarValue = 0.0;
    int8_t offset1 = 0;
    int16_t offset2 = 0;
    bool isSigned = false;
    lsb_mask_t bitMaskNum = lsb_mask_t::Disable;
};


#if ((__NPU_ARCH__ == 3113))
template<typename T = int16_t>
using FixPipePreQuantParams = FixPipePreQuantParamsV311Gen<T>;
#elif ((__NPU_ARCH__ == 3103))
template<typename T = int16_t>
using FixPipePreQuantParams = FixPipePreQuantParamsV310Gen<T>;
#elif (__NPU_ARCH__ == 3003)
template<typename T = int16_t>
using FixPipePreQuantParams = FixPipePreQuantParamsV300<T>;
#else
template<typename T = int16_t>
struct FixPipePreQuantParams {
    __aicore__ FixPipePreQuantParams()
    {
        offset0 = 0;
        f16Multiplier = 0.0;
        scalarValue = 0.0;
        offset1 = 0;
        offset2 = 0;
        isSigned = false;
        bitMaskNum = lsb_mask_t::Disable;
    }

    __aicore__ inline void SetOffset0(int32_t offset0_)
    {
        offset0 = offset0_;
    }

    __aicore__ inline void SetF16Multiplier(float16_t f16Multiplier_)
    {
        f16Multiplier = f16Multiplier_;
    }

    __aicore__ inline void SetScalarValue(float32_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetOffset1(int8_t offset1_)
    {
        offset1 = offset1_;
    }

    __aicore__ inline void SetOffset2(int16_t offset2_)
    {
        offset2 = offset2_;
    }

    __aicore__ inline void SetIsSigned(bool isSigned_)
    {
        isSigned = isSigned_;
    }

    __aicore__ inline void SetBitMaskNum(lsb_mask_t bitMaskNum_)
    {
        bitMaskNum = bitMaskNum_;
    }

    __aicore__ inline void SetClipMaxValPre(T clipMaxValPre_)
    {
        (void)clipMaxValPre_;
    }

    int32_t offset0 = 0;
    float16_t f16Multiplier = 0.0;
    float32_t scalarValue = 0.0;
    int8_t offset1 = 0;
    int16_t offset2 = 0;
    bool isSigned = false;
    lsb_mask_t bitMaskNum = lsb_mask_t::Disable;
};
#endif

enum class lut_mode_t {
    Gelu,
    Silu,
    Sigmoid,
    Tanh
};

struct FixPipeLeakyReluAlphaParamsV300 {
    __aicore__ FixPipeLeakyReluAlphaParamsV300()
    {
        preValue = 0.0;
        postValue = 0.0;
    }

    __aicore__ inline void SetPreValue(float32_t preValue_)
    {
        preValue = preValue_;
    }

    __aicore__ inline void SetPostValue(float32_t postValue_)
    {
        postValue = postValue_;
    }

    __aicore__ inline void SetLut(lut_mode_t lut_)
    {
        (void)lut_;
    }

    float32_t preValue = 0.0;
    float32_t postValue = 0.0;
};

struct FixPipeLeakyReluAlphaParamsV310Gen {
    __aicore__ FixPipeLeakyReluAlphaParamsV310Gen()
    {
        preValue = 0.0;
        postValue = 0.0;
    }

    __aicore__ inline void SetPreValue(float32_t preValue_)
    {
        preValue = preValue_;
    }

    __aicore__ inline void SetPostValue(float32_t postValue_)
    {
        postValue = postValue_;
    }

    __aicore__ inline void SetLut(lut_mode_t lut_)
    {
        (void)lut_;
    }

    float32_t preValue = 0.0;
    float32_t postValue = 0.0;
};


struct FixPipeLeakyReluAlphaParamsV311Gen {
    __aicore__ FixPipeLeakyReluAlphaParamsV311Gen()
    {
        preValue = 0.0;
        postValue = 0.0;
    }

    __aicore__ inline void SetPreValue(float32_t preValue_)
    {
        preValue = preValue_;
    }

    __aicore__ inline void SetPostValue(float32_t postValue_)
    {
        postValue = postValue_;
    }

    __aicore__ inline void SetLut(lut_mode_t lut_)
    {
        (void)lut_;
    }

    float32_t preValue = 0.0;
    float32_t postValue = 0.0;
};


#if (__NPU_ARCH__ == 3113)
using FixPipeLeakyReluAlphaParams = FixPipeLeakyReluAlphaParamsV311Gen;
#elif (__NPU_ARCH__ == 3103)
using FixPipeLeakyReluAlphaParams = FixPipeLeakyReluAlphaParamsV310Gen;
#elif (__NPU_ARCH__ == 3003)
using FixPipeLeakyReluAlphaParams = FixPipeLeakyReluAlphaParamsV300;
#else
struct FixPipeLeakyReluAlphaParams {
    __aicore__ FixPipeLeakyReluAlphaParams()
    {
        preValue = 0.0;
        postValue = 0.0;
    }

    __aicore__ inline void SetPreValue(float32_t preValue_)
    {
        preValue = preValue_;
    }

    __aicore__ inline void SetPostValue(float32_t postValue_)
    {
        postValue = postValue_;
    }

    __aicore__ inline void SetLut(lut_mode_t lut_)
    {
        (void)lut_;
    }

    float32_t preValue = 0.0;
    float32_t postValue = 0.0;
};
#endif

template<typename T = int8_t>
struct FixPipeEltAntiqParamsV300 {
    __aicore__ FixPipeEltAntiqParamsV300()
    {
        scalarValue = 0.0;
        s4Offset = 0;
        b8Offset = (T)0;
    }

    __aicore__ inline void SetScalarValue(float16_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetS4Offset(int8_t s4Offset_)
    {
        s4Offset = s4Offset_;
    }

    __aicore__ inline void SetB8Offset(T b8Offset_)
    {
        b8Offset = b8Offset_;
    }

    __aicore__ inline void SetS16Offset(int16_t s16Offset_) {
    }

    __aicore__ inline void SetEltAntiqCfg(eltwise_antiq_t eltAntiqCfg_) {
    }

    float16_t scalarValue = 0.0;
    int8_t s4Offset = 0;
    T b8Offset = (T)0;
};

template<typename T = int8_t>
struct FixPipeEltAntiqParamsV310Gen {
    __aicore__ FixPipeEltAntiqParamsV310Gen()
    {
        scalarValue = 0.0;
        s4Offset = 0;
        b8Offset = static_cast<T>(0);
        s16Offset = 0;
        eltAntiqCfg = eltwise_antiq_t::S8ANTIQ_Scalar;
    }

    __aicore__ inline void SetScalarValue(float16_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetS4Offset(int8_t s4Offset_)
    {
        s4Offset = s4Offset_;
    }

    __aicore__ inline void SetB8Offset(T b8Offset_)
    {
        b8Offset = b8Offset_;
    }

    __aicore__ inline void SetS16Offset(int16_t s16Offset_)
    {
        s16Offset = s16Offset_;
    }

    __aicore__ inline void SetEltAntiqCfg(eltwise_antiq_t eltAntiqCfg_)
    {
        eltAntiqCfg = eltAntiqCfg_;
    }

    float16_t scalarValue = 0.0;
    int8_t s4Offset = 0;
    T b8Offset = static_cast<T>(0);
    int16_t s16Offset = 0;
    eltwise_antiq_t eltAntiqCfg = eltwise_antiq_t::S8ANTIQ_Scalar;
};


template<typename T = int8_t>
struct FixPipeEltAntiqParamsV311Gen {
    __aicore__ FixPipeEltAntiqParamsV311Gen()
    {
        scalarValue = 0.0;
        s4Offset = 0;
        b8Offset = static_cast<T>(0);
        s16Offset = 0;
        eltAntiqCfg = eltwise_antiq_t::S8ANTIQ_Scalar;
    }

    __aicore__ inline void SetScalarValue(float16_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetS4Offset(int8_t s4Offset_)
    {
        s4Offset = s4Offset_;
    }

    __aicore__ inline void SetB8Offset(T b8Offset_)
    {
        b8Offset = b8Offset_;
    }

    __aicore__ inline void SetS16Offset(int16_t s16Offset_)
    {
        s16Offset = s16Offset_;
    }

    __aicore__ inline void SetEltAntiqCfg(eltwise_antiq_t eltAntiqCfg_)
    {
        eltAntiqCfg = eltAntiqCfg_;
    }

    float16_t scalarValue = 0.0;
    int8_t s4Offset = 0;
    T b8Offset = static_cast<T>(0);
    int16_t s16Offset = 0;
    eltwise_antiq_t eltAntiqCfg = eltwise_antiq_t::S8ANTIQ_Scalar;
};


#if ((__NPU_ARCH__ == 3113))
template<typename T = int8_t>
using FixPipeEltAntiqParams = FixPipeEltAntiqParamsV311Gen<T>;
#elif ((__NPU_ARCH__ == 3103))
template<typename T = int8_t>
using FixPipeEltAntiqParams = FixPipeEltAntiqParamsV310Gen<T>;
#elif (__NPU_ARCH__ == 3003)
template<typename T = int8_t>
using FixPipeEltAntiqParams = FixPipeEltAntiqParamsV300<T>;
#else
template<typename T = int8_t>
struct FixPipeEltAntiqParams {
    __aicore__ FixPipeEltAntiqParams()
    {
        scalarValue = 0.0;
        s4Offset = 0;
        b8Offset = static_cast<T>(0);
        s16Offset = 0;
        eltAntiqCfg = eltwise_antiq_t::S8ANTIQ_Scalar;
    }

    __aicore__ inline void SetScalarValue(float16_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetS4Offset(int8_t s4Offset_)
    {
        s4Offset = s4Offset_;
    }

    __aicore__ inline void SetB8Offset(T b8Offset_)
    {
        b8Offset = b8Offset_;
    }

    __aicore__ inline void SetS16Offset(int16_t s16Offset_)
    {
        s16Offset = s16Offset_;
    }

    __aicore__ inline void SetEltAntiqCfg(eltwise_antiq_t eltAntiqCfg_)
    {
        eltAntiqCfg = eltAntiqCfg_;
    }

    float16_t scalarValue = 0.0;
    int8_t s4Offset = 0;
    T b8Offset = static_cast<T>(0);
    int16_t s16Offset = 0;
    eltwise_antiq_t eltAntiqCfg = eltwise_antiq_t::S8ANTIQ_Scalar;
};
#endif

struct FixpipeEltwiseAddrParamsV300 {
    __aicore__ FixpipeEltwiseAddrParamsV300()
    {
        c0ChannelStride = 0;
        eltSrcAddr = 0;
    }

    __aicore__ inline void SetC0Indicator(bool c0Indicator_)
    {
        (void)c0Indicator_;
    }

    __aicore__ inline void SetC0ChannelStride(uint32_t c0ChannelStride_)
    {
        c0ChannelStride = c0ChannelStride_;
    }

    __aicore__ inline void SetEltSrcAddr(uint32_t eltSrcAddr_)
    {
        eltSrcAddr = eltSrcAddr_;
    }

    __aicore__ inline void SetSrcStride1(uint32_t srcStride1_)
    {
        (void)srcStride1_;
    }

    __aicore__ inline void SetSrcStride2(uint32_t srcStride2_)
    {
        (void)srcStride2_;
    }

    uint32_t c0ChannelStride = 0;
    uint32_t eltSrcAddr = 0;
};

struct FixpipeEltwiseAddrParamsV310Gen {
    __aicore__ FixpipeEltwiseAddrParamsV310Gen()
    {
        c0ChannelStride = 0;
        eltSrcAddr = 0;
    }

    __aicore__ inline void SetC0Indicator(bool c0Indicator_)
    {
        (void)c0Indicator_;
    }

    __aicore__ inline void SetC0ChannelStride(uint32_t c0ChannelStride_)
    {
        c0ChannelStride = c0ChannelStride_;
    }

    __aicore__ inline void SetEltSrcAddr(uint32_t eltSrcAddr_)
    {
        eltSrcAddr = eltSrcAddr_;
    }

    __aicore__ inline void SetSrcStride1(uint32_t srcStride1_)
    {
        (void)srcStride1_;
    }

    __aicore__ inline void SetSrcStride2(uint32_t srcStride2_)
    {
        (void)srcStride2_;
    }

    uint32_t c0ChannelStride = 0;
    uint32_t eltSrcAddr = 0;
};


struct FixpipeEltwiseAddrParamsV311Gen {
    __aicore__ FixpipeEltwiseAddrParamsV311Gen()
    {
        c0ChannelStride = 0;
        eltSrcAddr = 0;
    }

    __aicore__ inline void SetC0Indicator(bool c0Indicator_)
    {
        (void)c0Indicator_;
    }

    __aicore__ inline void SetC0ChannelStride(uint32_t c0ChannelStride_)
    {
        c0ChannelStride = c0ChannelStride_;
    }

    __aicore__ inline void SetEltSrcAddr(uint32_t eltSrcAddr_)
    {
        eltSrcAddr = eltSrcAddr_;
    }

    __aicore__ inline void SetSrcStride1(uint32_t srcStride1_)
    {
        (void)srcStride1_;
    }

    __aicore__ inline void SetSrcStride2(uint32_t srcStride2_)
    {
        (void)srcStride2_;
    }

    uint32_t c0ChannelStride = 0;
    uint32_t eltSrcAddr = 0;
};


#if ((__NPU_ARCH__ == 3113))
using FixpipeEltwiseAddrParams = FixpipeEltwiseAddrParamsV311Gen;
#elif ((__NPU_ARCH__ == 3103))
using FixpipeEltwiseAddrParams = FixpipeEltwiseAddrParamsV310Gen;
#elif (__NPU_ARCH__ == 3003)
using FixpipeEltwiseAddrParams = FixpipeEltwiseAddrParamsV300;
#else
struct FixpipeEltwiseAddrParams {
    __aicore__ FixpipeEltwiseAddrParams()
    {
        c0ChannelStride = 0;
        eltSrcAddr = 0;
    }

    __aicore__ inline void SetC0Indicator(bool c0Indicator_)
    {
        (void)c0Indicator_;
    }

    __aicore__ inline void SetC0ChannelStride(uint32_t c0ChannelStride_)
    {
        c0ChannelStride = c0ChannelStride_;
    }

    __aicore__ inline void SetEltSrcAddr(uint32_t eltSrcAddr_)
    {
        eltSrcAddr = eltSrcAddr_;
    }

    __aicore__ inline void SetSrcStride1(uint32_t srcStride1_)
    {
        (void)srcStride1_;
    }

    __aicore__ inline void SetSrcStride2(uint32_t srcStride2_)
    {
        (void)srcStride2_;
    }

    uint32_t c0ChannelStride = 0;
    uint32_t eltSrcAddr = 0;
};
#endif

struct FixPipePostQuantParamsV300 {
    __aicore__ FixPipePostQuantParamsV300()
    {
        offset0 = 0;
        offset1 = 0;
        isSigned = false;
        scalarValue = 0.0;
        offset2 = 0;
    }

    __aicore__ inline void SetOffset0(int8_t offset0_)
    {
        offset0 = offset0_;
    }

    __aicore__ inline void SetOffset1(int16_t offset1_)
    {
        offset1 = offset1_;
    }

    __aicore__ inline void SetIsSigned(bool isSigned_)
    {
        isSigned = isSigned_;
    }

    __aicore__ inline void SetScalarValue(float32_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetOffset2(int32_t offset2_)
    {
        offset2 = offset2_;
    }

    __aicore__ inline void SetShiftValue(uint8_t shiftValue_) {
    }

    int8_t offset0 = 0;
    int16_t offset1 = 0;
    bool isSigned = false;
    float32_t scalarValue = 0.0;
    int32_t offset2 = 0;
};

struct FixPipePostQuantParamsV310Gen {
    __aicore__ FixPipePostQuantParamsV310Gen()
    {
        offset0 = 0;
        offset1 = 0;
        isSigned = false;
        scalarValue = 0.0;
        offset2 = 0;
        shiftValue = 0;
    }

    __aicore__ inline void SetOffset0(int8_t offset0_)
    {
        offset0 = offset0_;
    }

    __aicore__ inline void SetOffset1(int16_t offset1_)
    {
        offset1 = offset1_;
    }

    __aicore__ inline void SetIsSigned(bool isSigned_)
    {
        isSigned = isSigned_;
    }

    __aicore__ inline void SetScalarValue(float32_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetOffset2(int32_t offset2_)
    {
        offset2 = offset2_;
    }

    __aicore__ inline void SetShiftValue(uint8_t shiftValue_)
    {
        shiftValue = shiftValue_;
    }

    int8_t offset0 = 0;
    int16_t offset1 = 0;
    bool isSigned = false;
    float32_t scalarValue = 0.0;
    int32_t offset2 = 0;
    uint8_t shiftValue = 0;
};

struct FixPipePostQuantParamsV311Gen {
    __aicore__ FixPipePostQuantParamsV311Gen()
    {
        offset0 = 0;
        offset1 = 0;
        isSigned = false;
        scalarValue = 0.0;
        offset2 = 0;
        shiftValue = 0;
    }

    __aicore__ inline void SetOffset0(int8_t offset0_)
    {
        offset0 = offset0_;
    }

    __aicore__ inline void SetOffset1(int16_t offset1_)
    {
        offset1 = offset1_;
    }

    __aicore__ inline void SetIsSigned(bool isSigned_)
    {
        isSigned = isSigned_;
    }

    __aicore__ inline void SetScalarValue(float32_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetOffset2(int32_t offset2_)
    {
        offset2 = offset2_;
    }

    __aicore__ inline void SetShiftValue(uint8_t shiftValue_)
    {
        shiftValue = shiftValue_;
    }

    int8_t offset0 = 0;
    int16_t offset1 = 0;
    bool isSigned = false;
    float32_t scalarValue = 0.0;
    int32_t offset2 = 0;
    uint8_t shiftValue = 0;
};

#if ((__NPU_ARCH__ == 3113))
using FixPipePostQuantParams = FixPipePostQuantParamsV311Gen;
#elif ((__NPU_ARCH__ == 3103))
using FixPipePostQuantParams = FixPipePostQuantParamsV310Gen;
#elif (__NPU_ARCH__ == 3003)
using FixPipePostQuantParams = FixPipePostQuantParamsV300;
#else
struct FixPipePostQuantParams {
    __aicore__ FixPipePostQuantParams()
    {
        offset0 = 0;
        offset1 = 0;
        isSigned = false;
        scalarValue = 0.0;
        offset2 = 0;
        shiftValue = 0;
    }

    __aicore__ inline void SetOffset0(int8_t offset0_)
    {
        offset0 = offset0_;
    }

    __aicore__ inline void SetOffset1(int16_t offset1_)
    {
        offset1 = offset1_;
    }

    __aicore__ inline void SetIsSigned(bool isSigned_)
    {
        isSigned = isSigned_;
    }

    __aicore__ inline void SetScalarValue(float32_t scalarValue_)
    {
        scalarValue = scalarValue_;
    }

    __aicore__ inline void SetOffset2(int32_t offset2_)
    {
        offset2 = offset2_;
    }

    __aicore__ inline void SetShiftValue(uint8_t shiftValue_)
    {
        shiftValue = shiftValue_;
    }

    int8_t offset0 = 0;
    int16_t offset1 = 0;
    bool isSigned = false;
    float32_t scalarValue = 0.0;
    int32_t offset2 = 0;
    uint8_t shiftValue = 0;
};
#endif

#if (__NPU_ARCH__ != 3113)
struct FixpipeChannelParams {
    __aicore__ FixpipeChannelParams()
    {
        c0ChannelStride = 0;
        channelMergeStride = 0;
        c0ChannelIndicator = 0;
        srcStride = 0;
    }

    __aicore__ FixpipeChannelParams(uint16_t c0ChannelStride_, uint16_t channelMergeStride_, bool c0ChannelIndicator_, uint16_t srcStride_)
    {
        c0ChannelStride = c0ChannelStride_;
        channelMergeStride = channelMergeStride_;
        c0ChannelIndicator = c0ChannelIndicator_;
        srcStride = srcStride_;
    }

    __aicore__ inline void SetC0ChannelStride(const uint16_t c0ChannelStride_)
    {
        c0ChannelStride = c0ChannelStride_;
    }

    __aicore__ inline void SetChannelMergeStride(const uint16_t channelMergeStride_)
    {
        channelMergeStride = channelMergeStride_;
    }

    __aicore__ inline void SetC0ChannelIndicator(const bool c0ChannelIndicator_)
    {
        c0ChannelIndicator = c0ChannelIndicator_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    uint16_t c0ChannelStride = 0;
    uint16_t channelMergeStride = 0;
    bool c0ChannelIndicator = 0;
    uint16_t srcStride = 0;
};
#endif

struct FixpipeChannelParamsV311Gen {
    __aicore__ FixpipeChannelParamsV311Gen()
    {
        c0ChannelStride = 0;
        channelMergeStride = 0;
        c0ChannelIndicator = 0;
        srcStride = 0;
    }

    __aicore__ FixpipeChannelParamsV311Gen(uint16_t c0ChannelStride_, uint16_t channelMergeStride_, bool c0ChannelIndicator_, uint16_t srcStride_)
    {
        c0ChannelStride = c0ChannelStride_;
        channelMergeStride = channelMergeStride_;
        c0ChannelIndicator = c0ChannelIndicator_;
        srcStride = srcStride_;
    }

    __aicore__ inline void SetC0ChannelStride(const uint16_t c0ChannelStride_)
    {
        c0ChannelStride = c0ChannelStride_;
    }

    __aicore__ inline void SetChannelMergeStride(const uint16_t channelMergeStride_)
    {
        channelMergeStride = channelMergeStride_;
    }

    __aicore__ inline void SetC0ChannelIndicator(const bool c0ChannelIndicator_)
    {
        c0ChannelIndicator = c0ChannelIndicator_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    uint16_t c0ChannelStride = 0;
    uint16_t channelMergeStride = 0;
    bool c0ChannelIndicator = 0;
    uint16_t srcStride = 0;
};

#if ((__NPU_ARCH__ == 3113))
using FixpipeChannelParams = FixpipeChannelParamsV311Gen;
#endif

#if (__NPU_ARCH__ != 3113)
template <typename T, typename U>
struct FixpipeClipReluParams {
    __aicore__ FixpipeClipReluParams()
    {
        preValue = 0;
        postValue = 0;
    }

    __aicore__ FixpipeClipReluParams(const T preValue_, const U postValue_)
    {
        preValue = preValue_;
        postValue = postValue_;
    }

    __aicore__ inline void SetPreValue(const T preValue_)
    {
        preValue = preValue_;
    }

    __aicore__ inline void SetPostValue(const U postValue_)
    {
        postValue = postValue_;
    }

    T preValue = 0;
    U postValue = 0;
};
#endif

template <typename T, typename U>
struct FixpipeClipReluParamsV311Gen {
    __aicore__ FixpipeClipReluParamsV311Gen()
    {
        preValue = 0;
        postValue = 0;
    }

    __aicore__ FixpipeClipReluParamsV311Gen(const T preValue_, const U postValue_)
    {
        preValue = preValue_;
        postValue = postValue_;
    }

    __aicore__ inline void SetPreValue(const T preValue_)
    {
        preValue = preValue_;
    }

    __aicore__ inline void SetPostValue(const U postValue_)
    {
        postValue = postValue_;
    }

    T preValue = 0;
    U postValue = 0;
};


#if ((__NPU_ARCH__ == 3113))
template <typename T, typename U>
using FixpipeClipReluParams = FixpipeClipReluParamsV311Gen<T, U>;
#endif

#if (__NPU_ARCH__ != 3113)
struct FixpipeLoop3Params {
    __aicore__ FixpipeLoop3Params()
    {
        loopSize = 0;
        srcStride = 0;
        dstStride = 0;
    }

    __aicore__ FixpipeLoop3Params(uint16_t loopSize_, uint16_t srcStride_, uint32_t dstStride_)
    {
        loopSize = loopSize_;
        srcStride = srcStride_;
        dstStride = dstStride_;
    }

    __aicore__ inline void SetLoopSize(const uint16_t loopSize_)
    {
        loopSize = loopSize_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint32_t dstStride_)
    {
        dstStride = dstStride_;
    }

    uint16_t loopSize = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
};
#endif

struct FixpipeLoop3ParamsV311Gen {
    __aicore__ FixpipeLoop3ParamsV311Gen()
    {
        loopSize = 0;
        srcStride = 0;
        dstStride = 0;
    }

    __aicore__ FixpipeLoop3ParamsV311Gen(uint16_t loopSize_, uint16_t srcStride_, uint32_t dstStride_)
    {
        loopSize = loopSize_;
        srcStride = srcStride_;
        dstStride = dstStride_;
    }

    __aicore__ inline void SetLoopSize(const uint16_t loopSize_)
    {
        loopSize = loopSize_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint32_t dstStride_)
    {
        dstStride = dstStride_;
    }

    uint16_t loopSize = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
};

#if ((__NPU_ARCH__ == 3113))
using FixpipeLoop3Params = FixpipeLoop3ParamsV311Gen;
#endif

#if (__NPU_ARCH__ != 3113)
struct FixpipeLoop4Params {
    __aicore__ FixpipeLoop4Params()
    {
        loopSize = 0;
        srcStride = 0;
        dstStride = 0;
    }

    __aicore__ FixpipeLoop4Params(uint16_t loopSize_, uint16_t srcStride_, uint32_t dstStride_)
    {
        loopSize = loopSize_;
        srcStride = srcStride_;
        dstStride = dstStride_;
    }

    __aicore__ inline void SetLoopSize(const uint16_t loopSize_)
    {
        loopSize = loopSize_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint32_t dstStride_)
    {
        dstStride = dstStride_;
    }

    uint16_t loopSize = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
};
#endif

struct FixpipeLoop4ParamsV311Gen {
    __aicore__ FixpipeLoop4ParamsV311Gen()
    {
        loopSize = 0;
        srcStride = 0;
        dstStride = 0;
    }

    __aicore__ FixpipeLoop4ParamsV311Gen(uint16_t loopSize_, uint16_t srcStride_, uint32_t dstStride_)
    {
        loopSize = loopSize_;
        srcStride = srcStride_;
        dstStride = dstStride_;
    }

    __aicore__ inline void SetLoopSize(const uint16_t loopSize_)
    {
        loopSize = loopSize_;
    }

    __aicore__ inline void SetSrcStride(const uint16_t srcStride_)
    {
        srcStride = srcStride_;
    }

    __aicore__ inline void SetDstStride(const uint32_t dstStride_)
    {
        dstStride = dstStride_;
    }

    uint16_t loopSize = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
};

#if ((__NPU_ARCH__ == 3113))
using FixpipeLoop4Params = FixpipeLoop4ParamsV311Gen;
#endif

#endif

} // namespace AscendC
#endif // ASCENDC_MODULE_UTILS_STRUCT_DMA_PARAMS_H