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
 * \file block_conv_impl.h
 * \brief
 */
#ifndef BLOCK_CONV_IMPL_H
#define BLOCK_CONV_IMPL_H

#include "../utils/conv_common_utils.h"
#include "../utils/conv_host_utils.h"
#include "../utils/conv_status_utils.h"
#include "../utils/conv_layout_utils.h"
#include "../utils/conv_status_utils.h"

#define CONV2D_BLOCK_CLASS_LOCAL_PARAMS                                                                              \
    template<class ProblemShape_, class L1TileShape_, class L0TileShape_, class AconvType_, class BconvType_,        \
    class CConvType_, class L0CType_, class BiasConvType_, class TileAttr_>

#define CONV2D_BLOCK_FUNC_LOCAL_PARAM                                                                               \
    ProblemShape_, L1TileShape_, L0TileShape_, AconvType_, BconvType_, CConvType_, L0CType_, BiasConvType_, TileAttr_

namespace Act {
namespace Conv {
namespace Block {

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
class ConvImpl {
    using ProblemShape = ProblemShape_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using AType =AconvType_;
    using BType = BconvType_;
    using CType = CConvType_;
    using BiasType =  BiasConvType_;
    using TileAttr = TileAttr_;
    using L0CType = L0CType_;

public:
    const static uint64_t C0_SIZE = 32;
    constexpr static uint64_t k0 = C0_SIZE / sizeof(AType);
    constexpr static uint8_t sizeOfBias = sizeof(BiasType);
    const static uint64_t offsetx = 0;
    static constexpr int64_t mL0 = GetIntegralConstant<0, L0TileShape>();
    static constexpr int64_t nL0 = GetIntegralConstant<1, L0TileShape>();
    static constexpr int64_t kL0 = GetIntegralConstant<2, L0TileShape>();
    static constexpr int64_t mAL1 = GetIntegralConstant<0, L1TileShape>();
    static constexpr int64_t nBL1 = GetIntegralConstant<1, L1TileShape>();
    static constexpr int64_t kAL1Tiling = GetIntegralConstant<2, L1TileShape>();
    static constexpr int64_t kBL1Tiling = GetIntegralConstant<3, L1TileShape>();

    __aicore__ inline void CopyGmToA1MMode(AscendC::GlobalTensor<AType>& aGm, AscendC::LocalTensor<AType>& aL1,
                                           ConvInterate& iterate, const TempIters& tempIters, const uint64_t hiIdx,
                                           uint64_t padTopL1, uint64_t hiLoadL1);
    __aicore__ inline void CopyA1ToA2MMode(AscendC::LocalTensor<AType>& aL1, AscendC::LocalTensor<AType>& aL0,
                                           const TempIters& tempIters, const ConvInterate& iterate,
                                           const uint64_t mStartPos, const bool isFirst);
    __aicore__ inline void CopyGmToB1(AscendC::GlobalTensor<BType>& bGm, AscendC::LocalTensor<BType>& bL1,
                                      ConvInterate& iterate, const TempIters& tempIters);
    __aicore__ inline void CopyB1ToB2(AscendC::LocalTensor<BType>& bL1, AscendC::LocalTensor<BType>& bL0,
                                      const TempIters& tempIters, bool isFirst);
    __aicore__ inline void CopyGmToA1Bias(AscendC::GlobalTensor<BiasType>& biasGM,
                                          AscendC::LocalTensor<BiasType>& biasL1,
                                          uint64_t loadNum, uint64_t gmStartAddr);
    __aicore__ inline void CopyA1ToC2Bias(AscendC::LocalTensor<BiasType>& biasL1,
                                          AscendC::LocalTensor<L0CType>& biasBt, ConvInterate& iterate);
    __aicore__ inline void Mmad(AscendC::LocalTensor<AType>& aL0, AscendC::LocalTensor<BType>& bL0,
                                AscendC::LocalTensor<L0CType>& cL0, const TempIters& tempIters, ConvInterate& iterate);
    __aicore__ inline void CopyCo1ToGmMMode(AscendC::LocalTensor<L0CType>& cL0, AscendC::GlobalTensor<CType>& cGm,
                                            ConvInterate& iterate);
    __aicore__ inline void SetLoad3dFMatrix(uint64_t padTopL1, uint64_t hiLoadL1);
    #if defined(__DAV_C310__) // for Ascend910_95
    __aicore__ inline void SetDn2NzIntriParams(AscendC::Dn2NzParams &intriParams, uint64_t kAL1Iter, uint64_t hiLoadL1);
    #endif
    __aicore__ inline void SetParamsCommon();
    __aicore__ inline void SetParams(TempIters tempIters, ProblemShape shape, SingleCoreShape singleCoreShape);
    __aicore__ inline QuantMode_t GetQuantPre();
    __aicore__ inline void SetMNBeforeIterateK(const TempIters& tempIters, const ConvInterate& iterate);

private:
    ProblemShape convShapeLocal;
    SingleCoreShape singleCoreShapeLocal;
    uint64_t orgHixWi;
    uint64_t kernelHxkernelW;
    uint64_t dilatedKernelH;
    uint64_t dilatedKernelW;
    uint64_t alignCinKhKw;
    bool kAL1fullload;
    bool kBL1fullload;
    uint64_t fmapOneBatchSize;
    uint64_t outputOneBatchSize;
    uint64_t cinOffsetBlockInGM;
    uint64_t coutOffsetBlock;
    uint64_t maxKBL1Iter;
    uint64_t kBL1Tail;
    uint64_t cinBTailInCore;
    uint64_t cinBInCore;
    uint64_t maxNBL1Iter;
    uint64_t nBL1Tail;
    uint64_t nL0Tail;
    uint64_t nL1DivBlockSize;
    uint64_t xmA_ = 0;
    uint64_t xmB_ = 0;
    uint64_t xtA_ = 0;
    uint64_t xtB_ = 0;
    uint64_t xmtmpA_ = 0;
    uint64_t xmtmpB_ = 0;
    uint64_t nStep = 0;
    uint64_t maxKL0Iter;
    uint64_t kStep;
    uint64_t kL0Tail;
    uint64_t multiKBL1;
    uint64_t maxMAL1Iter;
    uint64_t mAL1Tail;
    uint64_t mAL0Tail;
    uint64_t cinAInCore;
    uint64_t kAL1Tail;
    uint64_t cinATailInCore;
    uint64_t maxKAL1Iter;
    uint64_t alignCinATailInCore;
    uint64_t multiKAL1;
    uint64_t channelSize_;
    uint64_t valueHoWo_;
    uint64_t currentML0;
    uint64_t currentML0Align;
    uint64_t currentNL0;
    uint64_t currentNL0Align;
    uint64_t ratioOfNToN0;
    int64_t kAL1 = 0;
    int64_t kBL1 = 0;
};

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::SetMNBeforeIterateK(const TempIters& tempIters,
                                                                                    const ConvInterate& iterate)
{
    bool isML0Tail = iterate.mAL1Iter == maxMAL1Iter;
    currentML0 = isML0Tail ? mAL0Tail : mL0;
    currentML0Align = isML0Tail ? Align(currentML0, BLOCK_L0_M) : currentML0;
    bool isNL0Tail = iterate.nBL1Iter == maxNBL1Iter;
    currentNL0 = isNL0Tail ? nL0Tail : nL0;
    currentNL0Align = isNL0Tail ? Align(currentNL0, BLOCK_L0_N) : currentNL0;

    #if defined(__DAV_C310__) // for Ascend910_95
    AscendC::LoadDataRepeatParam repeatParams = {0, 1, 0, static_cast<uint16_t>(currentML0Align / BLOCK_L0_M)};
    AscendC::SetLoadDataRepeat(repeatParams);
    #endif
    ratioOfNToN0 = (iterate.nBL1Iter == maxNBL1Iter) ? currentNL0Align / BLOCK_L0_N : nStep;
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::SetParamsCommon()
{
    orgHixWi = convShapeLocal.hin_ * convShapeLocal.win_;
    kernelHxkernelW = convShapeLocal.kh_ * convShapeLocal.kw_;
    kAL1 = kAL1Tiling * kernelHxkernelW;
    kBL1 = kBL1Tiling * kernelHxkernelW;
    dilatedKernelH = 1 + (convShapeLocal.kh_ - 1) * convShapeLocal.dilationh_;
    dilatedKernelW = 1 + (convShapeLocal.kw_ - 1) * convShapeLocal.dilationw_;
    alignCinKhKw = Align(singleCoreShapeLocal.singleCoreCi, k0) * kernelHxkernelW;
    kAL1fullload = alignCinKhKw == kAL1;
    kBL1fullload = alignCinKhKw == kBL1;

    fmapOneBatchSize = convShapeLocal.cin_ * convShapeLocal.hin_ * convShapeLocal.win_;
    outputOneBatchSize = convShapeLocal.cout_ * convShapeLocal.ho_ * convShapeLocal.wo_;
    cinOffsetBlockInGM = kAL1 / kernelHxkernelW * orgHixWi;
    coutOffsetBlock = convShapeLocal.cin_ * kernelHxkernelW;
    nL1DivBlockSize = nBL1 / BLOCK_L0_N;
    kStep = kL0 / k0;
    multiKBL1 = CeilDiv(kBL1, kL0);

    // calc BL1 tail block
    maxKBL1Iter = CeilDiv(Align(singleCoreShapeLocal.singleCoreCi, k0) * kernelHxkernelW, kBL1) - 1;
    kBL1Tail = singleCoreShapeLocal.singleCoreCi * kernelHxkernelW % kBL1;
    kBL1Tail = kBL1Tail == 0 ? kBL1 : kBL1Tail;
    cinBInCore = kBL1 / kernelHxkernelW;
    cinBTailInCore = kBL1Tail / kernelHxkernelW;

    // calc BL0 tail block
    maxNBL1Iter = CeilDiv(singleCoreShapeLocal.singleCoreN, nBL1)- 1;
    nBL1Tail = singleCoreShapeLocal.singleCoreN % nBL1;
    nBL1Tail = nBL1Tail == 0 ? nBL1 : nBL1Tail;
    nL0Tail = nBL1Tail % nL0;
    nL0Tail = nL0Tail == 0 ? nL0 : nL0Tail;

    // calc AL0 tail block
    maxMAL1Iter = CeilDiv(singleCoreShapeLocal.singleCoreM, mAL1) - 1;
    mAL1Tail = singleCoreShapeLocal.singleCoreM % mAL1;
    mAL1Tail = mAL1Tail == 0 ? mAL1 : mAL1Tail;
    mAL0Tail = mAL1Tail % mL0;
    mAL0Tail = mAL0Tail == 0 ? mL0 : mAL0Tail;

    // calc KL0 tail block
    maxKL0Iter = CeilDiv(Align(singleCoreShapeLocal.singleCoreCi, k0) * kernelHxkernelW, kL0)- 1;
    kL0Tail = (Align(singleCoreShapeLocal.singleCoreCi, k0) * kernelHxkernelW) % kL0;
    kL0Tail = kL0Tail == 0 ? kL0 : kL0Tail;
    multiKAL1 = CeilDiv(kAL1, kL0);
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::SetParams(TempIters tempIters, ProblemShape shape,
                                                                          SingleCoreShape singleCoreShape)
{
    convShapeLocal = shape;
    singleCoreShapeLocal = singleCoreShape;
    SetParamsCommon();
    
    maxKAL1Iter = CeilDiv(singleCoreShapeLocal.singleCoreCi * kernelHxkernelW, kAL1) - 1;
    kAL1Tail = (singleCoreShapeLocal.singleCoreCi * kernelHxkernelW) % kAL1;
    kAL1Tail = kAL1Tail == 0 ? kAL1 : kAL1Tail;
    cinAInCore = kAL1 / kernelHxkernelW;
    cinATailInCore = kAL1Tail / kernelHxkernelW;
    alignCinATailInCore = Align(cinATailInCore, k0);
    channelSize_ = (tempIters.kAL1Iter - 1 != maxKAL1Iter ? cinAInCore : alignCinATailInCore);
    // LoadAL0Tools SetParams初始化
    
    nStep = CeilDiv(nL0, BLOCK_L0_N);
    // LoadBL0Tools初始化

    valueHoWo_ = convShapeLocal.ho_ * convShapeLocal.wo_;
    // CopyOut SetParam初始化
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::SetLoad3dFMatrix(uint64_t padTopL1, uint64_t hiLoadL1)
{
    uint8_t padList[PAD_SIZE] = {MAX_PAD_R, MAX_PAD_R, MAX_PAD_R, MAX_PAD_R};
    padList[PAD_IDX_L] = convShapeLocal.padLeft_;
    padList[PAD_IDX_R] = convShapeLocal.padRight_;
    padList[PAD_IDX_T] = padTopL1;
    AscendC::Load3DSetFMatrixCal(hiLoadL1, convShapeLocal.win_, padList);
    AscendC::Load3DSetPaddingCal(offsetx);
}

#if defined(__DAV_C310__) // for Ascend910_95
CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::SetDn2NzIntriParams(AscendC::Dn2NzParams &intriParams,
                                                                                    uint64_t kAL1Iter,
                                                                                    uint64_t hiLoadL1)
{
    uint64_t aL1Mi = hiLoadL1 * convShapeLocal.win_;
    intriParams.dnNum = 1;
    intriParams.nValue = aL1Mi;
    intriParams.dValue = kAL1Iter == maxKAL1Iter ? cinATailInCore : cinAInCore;
    intriParams.srcDValue = orgHixWi;
    intriParams.dstNzNStride = 1;
    intriParams.dstNzC0Stride = aL1Mi;
}
#endif

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::CopyGmToA1MMode(AscendC::GlobalTensor<AType>& aGm,
                                                                                AscendC::LocalTensor<AType>& aL1,
                                                                                ConvInterate& iterate,
                                                                                const TempIters& tempIters,
                                                                                const uint64_t hiIdx,
                                                                                uint64_t padTopL1,
                                                                                uint64_t hiLoadL1)
{
    uint64_t aL1GmOffset = iterate.batchIter * fmapOneBatchSize + tempIters.kAL1Iter * cinOffsetBlockInGM +
                           hiIdx * convShapeLocal.win_;
    #if defined(__DAV_C310__) // for Ascend910_95
    AscendC::Dn2NzParams intriParams;
    SetDn2NzIntriParams(intriParams, tempIters.kAL1Iter, hiLoadL1);
    DataCopy<AType>(aL1, aGm[aL1GmOffset], intriParams);
    #endif
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::CopyA1ToA2MMode(AscendC::LocalTensor<AType>& aL1,
                                                                                AscendC::LocalTensor<AType>& aL0,
                                                                                const TempIters& tempIters,
                                                                                const ConvInterate& iterate,
                                                                                const uint64_t mStartPos,
                                                                                const bool isFirst)
{
    uint64_t currentKL0 = tempIters.kIter == maxKL0Iter ? kL0Tail : kL0;
    uint64_t posK = tempIters.kAL0Iter * kL0;
    if (isFirst) {
        uint64_t posM = (mStartPos + iterate.mAL1Iter * mAL1) % convShapeLocal.wo_;
        xmtmpA_ = ((currentML0 & MASK_16) << MSTEP_OFFSET) | ((posM & MASK_16) << POSM_OFFSET);
        xtA_ = ((static_cast<uint64_t>(convShapeLocal.stridew_) & MASK_6) << 0) |
                ((static_cast<uint64_t>(convShapeLocal.strideh_) & MASK_6) << STRIDEH_OFFSET) |
                ((static_cast<uint64_t>(convShapeLocal.kw_) & MASK_8) << KERNELW_OFFSET) |
                ((static_cast<uint64_t>(convShapeLocal.kh_) & MASK_8) << KERNELH_OFFSET) |
                ((static_cast<uint64_t>(convShapeLocal.dilationw_) & MASK_8) << DILATIONW_OFFSET) |
                ((static_cast<uint64_t>(convShapeLocal.dilationh_) & MASK_8) << DILATIONH_OFFSET) |
                ((static_cast<uint64_t>(channelSize_) & MASK_16) << CIN_OFFSET);
    }
    xmA_ = ((currentKL0 & MASK_16) << 0) | ((posK & MASK_16) << POSK_OFFSET) | xmtmpA_;
    #if defined(__DAV_C310__) // for Ascend910_95
    img2colv2_cbuf_to_ca((__ca__ AType *)aL0.GetPhyAddr(),
                            (__cbuf__ AType *)aL1.GetPhyAddr(), xmA_, xtA_);
    #endif
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::CopyGmToB1(AscendC::GlobalTensor<BType>& bGm,
                                                                           AscendC::LocalTensor<BType>& bL1,
                                                                           ConvInterate& iterate,
                                                                           const TempIters& tempIters)
{
    uint64_t bL1GmOffset = iterate.nBL1Iter * nBL1 * coutOffsetBlock + tempIters.kBL1Iter * kBL1;
    int64_t currentNBL1 = iterate.nBL1Iter == maxNBL1Iter ? nBL1Tail : nBL1;
    #if defined(__DAV_C310__) // for Ascend910_95
    AscendC::Dn2NzParams intriParams;
    intriParams.dnNum = currentNBL1;
    intriParams.nValue = kernelHxkernelW;
    intriParams.dValue = tempIters.kBL1Iter == maxKBL1Iter ? cinBTailInCore : cinBInCore;
    intriParams.srcDnMatrixStride =  coutOffsetBlock;
    intriParams.srcDValue = kernelHxkernelW;
    intriParams.dstNzC0Stride = kernelHxkernelW * nBL1;
    intriParams.dstNzNStride = nBL1;
    intriParams.dstNzMatrixStride = k0;
    DataCopy<BType>(bL1, bGm[bL1GmOffset], intriParams);
    #endif
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::CopyB1ToB2(AscendC::LocalTensor<BType>& bL1,
                                                                           AscendC::LocalTensor<BType>& bL0,
                                                                           const TempIters& tempIters,
                                                                           bool isFirst)
{
    uint64_t kBL0Iter = tempIters.kIter % multiKBL1;
    if (unlikely(isFirst)) {
        uint64_t mStartPosition = 0;
        xtB_ = (((nL1DivBlockSize & MASK_16) << 0) | ((ratioOfNToN0 & MASK_16) << DST_STRIDE_OFFSET));
        xmtmpB_ = (((mStartPosition & MASK_16) << 0) | ((ratioOfNToN0 & MASK_8) << M_STEP_OFFSET));
    }
    uint64_t kStepTemp = (tempIters.kIter != maxKL0Iter) ? kStep : (kL0Tail / k0);
    xmB_ = (((tempIters.kBL0Iter * kStepTemp) & MASK_16) << K_START_OFFSET) |
           ((kStepTemp & MASK_8) << K_STEP_OFFSET) | xmtmpB_;
    #if defined(__DAV_C310__) // for Ascend910_95
    load_cbuf_to_cb((__cb__ BType *)bL0.GetPhyAddr(),
                    (__cbuf__ BType *)bL1.GetPhyAddr(), xmB_, xtB_, false);
    #endif
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::CopyGmToA1Bias(AscendC::GlobalTensor<BiasType>& biasGM,
                                                                               AscendC::LocalTensor<BiasType>& biasL1,
                                                                               uint64_t loadNum, uint64_t gmStartAddr)
{
    uint64_t byteNum = sizeof(BiasType);
    AscendC::DataCopyParams dataCopyParams(1, loadNum * byteNum, 0, 0);
    uint8_t rightPadding = (uint8_t)(Align(loadNum * byteNum, PADDING_ALIGN_SIZE) / byteNum - loadNum);
    AscendC::DataCopyPadParams padParams(true, 0, rightPadding, 0);
    AscendC::DataCopyPad<BiasType>(biasL1, biasGM[gmStartAddr], dataCopyParams, padParams);
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::CopyA1ToC2Bias(AscendC::LocalTensor<BiasType>& biasL1,
                                                                               AscendC::LocalTensor<L0CType>& biasBt,
                                                                               ConvInterate& iterate)
{
    uint32_t offset = 0;
    offset = iterate.nBL1Iter * nBL1;
    if constexpr (AscendC::IsSameType<BiasType, half>::value) {
        copy_cbuf_to_bt((uint64_t)0, (__cbuf__ BiasType*)biasL1[offset].GetPhyAddr(),
            (bool)1, 1, currentNL0Align * sizeOfBias / BT_BLOCK_SIZE, 0, 0);
    } else {
        copy_cbuf_to_bt((uint64_t)0, (__cbuf__ BiasType*)biasL1[offset].GetPhyAddr(),
            (bool)0, 1, currentNL0Align * sizeOfBias / BT_BLOCK_SIZE, 0, 0);
    }
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::Mmad(AscendC::LocalTensor<AType>& aL0,
                                                                     AscendC::LocalTensor<BType>& bL0,
                                                                     AscendC::LocalTensor<L0CType>& cL0,
                                                                     const TempIters& tempIters,
                                                                     ConvInterate& iterate)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = currentML0Align;
    mmadParams.n = currentNL0Align;
    mmadParams.k = tempIters.kIter == maxKL0Iter ? kL0Tail : kL0;
    if (!convShapeLocal.hasbias_) {
        mmadParams.cmatrixInitVal = tempIters.kIter == 0;
        mmadParams.cmatrixSource = false;
    } else {
        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource = tempIters.kIter == 0;
    }
    #if defined(__DAV_C310__) // for Ascend910_95
    AscendC::Mmad<L0CType, AType, BType>(cL0, aL0, bL0, mmadParams);
    #endif
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline QuantMode_t ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::GetQuantPre()
{
    if constexpr (AscendC::IsSameType<AType, hifloat8_t>::value ||
                  AscendC::IsSameType<AType, fp8_e4m3fn_t>::value) {
        if constexpr (AscendC::IsSameType<CType, float>::value) {
            return QuantMode_t::VQF322F32_PRE;
        }

        if constexpr (AscendC::IsSameType<CType, half>::value) {
            return QuantMode_t::VQF322F16_PRE;
        }

        if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
            return QuantMode_t::VQF322BF16_PRE;
        }

        if constexpr (AscendC::IsSameType<CType, hifloat8_t>::value) {
            return QuantMode_t::VQF322HIF8_PRE;
        }

        if constexpr (AscendC::IsSameType<CType, fp8_e4m3fn_t>::value) {
            return QuantMode_t::VQF322FP8_PRE;
        }
    }

    if constexpr (AscendC::IsSameType<L0CType, float>::value &&
                    AscendC::IsSameType<CType, float>::value) {
        return QuantMode_t::NoQuant;
    }

    if constexpr (AscendC::IsSameType<L0CType, int32_t>::value &&
                    AscendC::IsSameType<CType, half>::value) {
        return QuantMode_t::VDEQF16;
    }

    if constexpr (AscendC::IsSameType<L0CType, int32_t>::value &&
                    AscendC::IsSameType<CType, int8_t>::value) {
        return QuantMode_t::VREQ8;
    }

    if constexpr (AscendC::IsSameType<L0CType, float>::value &&
                  AscendC::IsSameType<CType, bfloat16_t>::value) {
        return QuantMode_t::F322BF16;
    }
    
    return QuantMode_t::F322F16;
}

CONV2D_BLOCK_CLASS_LOCAL_PARAMS
__aicore__ inline void ConvImpl<CONV2D_BLOCK_FUNC_LOCAL_PARAM>::CopyCo1ToGmMMode(AscendC::LocalTensor<L0CType>& cL0,
                                                                                 AscendC::GlobalTensor<CType>& cGm,
                                                                                 ConvInterate& iterate)
{
    #if defined(__DAV_C310__) // for Ascend910_95
    AscendC::FixpipeParamsC310<AscendC::CFG_COLUMN_MAJOR.format> intriParams;
    intriParams.nSize = currentNL0;
    intriParams.mSize = currentML0;
    intriParams.srcStride = Align(currentML0, BLOCK_L0_M);
    intriParams.dstStride = valueHoWo_;
    intriParams.quantPre = GetQuantPre();
    intriParams.params.dnNum = 1;
    intriParams.params.srcNzMatrixStride = currentNL0 * currentML0 / BLOCK_L0_N;
    intriParams.params.dstDnMatrixStride = valueHoWo_;
    intriParams.params.srcNzC0Stride = 1;

    uint64_t offset = iterate.batchIter * outputOneBatchSize;
    uint64_t offsetCout = iterate.nBL1Iter * nBL1;
    uint64_t offsetMAL1 = iterate.mAL1Iter * mAL1;
    offset += offsetCout * valueHoWo_ + offsetMAL1;
    Fixpipe<CType, L0CType, AscendC::CFG_COLUMN_MAJOR>(cGm[offset], cL0, intriParams);
    #endif
}
}
}
}
#endif
