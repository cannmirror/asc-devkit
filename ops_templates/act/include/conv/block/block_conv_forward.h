/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file block_conv_forward.h
 * \brief
 */

#ifndef CONV_BLOCK_BLOCK_CONV_FORWARD_H
#define CONV_BLOCK_BLOCK_CONV_FORWARD_H

#include "../utils/conv_common_utils.h"
#include "../utils/conv_host_utils.h"
#include "../utils/conv_status_utils.h"
#include "../utils/conv_layout_utils.h"
#include "../utils/conv_status_utils.h"
#include "block_conv_impl.h"

namespace Act {
namespace Conv {
namespace Block {
template <class ProblemShape_, class DispatchPolicy_, class L1TileShape_, class L0TileShape_,
          class AConvType_, class BConvType_, class CConvType_, class BiasConvType_,
          class AGlobalTensorType_, class BGlobalTensorType_, class CGlobalTensorType_, class BiasGlobalTensorType_,
          class TileAttr_, class TileCopyParam_ = void>
class BlockConv {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>,
        "BlockConv is not implemented for this DispatchPolicy");
};

#define BLOCK_CONV_CLASS_LOCAL_PARAMS                                                                                \
    template <class ProblemShape_, class DispatchPolicy_, class L1TileShape_, class L0TileShape_,                    \
            class AConvType_, class BConvType_, class CConvType_, class BiasConvType_,                               \
            class AGlobalTensorType_, class BGlobalTensorType_, class CGlobalTensorType_, class BiasGlobalTensorType_,\
            class TileAttr_>

#define BLOCK_CONV_LOCAL_PARAMS                                                                                      \
    ProblemShape_, DispatchPolicy_, L1TileShape_, L0TileShape_, AConvType_, BConvType_, CConvType_, BiasConvType_,   \
    AGlobalTensorType_, BGlobalTensorType_, CGlobalTensorType_, BiasGlobalTensorType_, TileAttr_

BLOCK_CONV_CLASS_LOCAL_PARAMS
class BlockConv<BLOCK_CONV_LOCAL_PARAMS> {
public:

    using AType = AConvType_;
    using BType = BConvType_;
    using CType = CConvType_;
    using BiasType =  BiasConvType_;
    using TileAttr =  TileAttr_;
    using ProblemShape = ProblemShape_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using BiasGlobalTensorType = BiasGlobalTensorType_;
    using AGlobalTensorType = AGlobalTensorType_;
    using BGlobalTensorType = BGlobalTensorType_;
    using CGlobalTensorType = CGlobalTensorType_;
    using L0CType = typename GetDstType<AType>::Type;

    SingleCoreShape singleCoreShapeLocal;
    ProblemShape convShape;
    ConvDim dimInfo;
    static constexpr int64_t mL0 = GetIntegralConstant<0, L0TileShape>();
    static constexpr int64_t nL0 = GetIntegralConstant<1, L0TileShape>();
    static constexpr int64_t kL0 = GetIntegralConstant<2, L0TileShape>();
    static constexpr int64_t mAL1 = GetIntegralConstant<0, L1TileShape>();
    static constexpr int64_t nBL1 = GetIntegralConstant<1, L1TileShape>();
    static constexpr int64_t kAL1Tiling = GetIntegralConstant<2, L1TileShape>();
    static constexpr int64_t kBL1Tiling = GetIntegralConstant<3, L1TileShape>();

    static constexpr int64_t kAL1fullload = GetIntegralConstant<0, TileAttr>();
    static constexpr int64_t kBL1fullload = GetIntegralConstant<1, TileAttr>();
    static constexpr int64_t biasFullLoadFlagL = GetIntegralConstant<2, TileAttr>();
    static constexpr int64_t pingpongL1 = GetIntegralConstant<3, TileAttr>();
    /*
    依次为Fmap载入状态(0(全载L1)、1(全载Lθ)、2(切分))、
    Weight载入状态((全载L1)、1(全载Lθ)、2(切分)
    Bias载入状态(θ(全载)、1(切分))
    L1乒乓状态 (0(双开)、1(Fmap单开)、2(Weight单开)、3(双不开))
    */

    constexpr static uint8_t sizeOfFmap = sizeof(AType);
    constexpr static uint8_t sizeOfWeight = sizeof(BType);
    constexpr static uint8_t sizeOfL0c = sizeof(L0CType);
    constexpr static uint8_t sizeOfBias = sizeof(BiasType);
    ConvImpl<ProblemShape, L1TileShape, L0TileShape, AType, BType, CType, L0CType, BiasType, TileAttr> conv;
    const static uint8_t DOUBLE_BUF = 2;
    const static uint64_t C0_SIZE = 32;
    const static uint64_t BT_SIZE = 64;
    constexpr static uint32_t CONST_VALUE_2 = 2;
    constexpr static uint32_t M0 = 16;
    const static uint32_t BLOCK_L0_N = 16;
    const static uint64_t L0A_SIZE = 65536;
    const static uint64_t L0B_SIZE = 65536;
    constexpr static uint64_t k0 = C0_SIZE / sizeof(AType);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::A1, 1> queueBiasL1; // BiasL1
    AscendC::TQue<AscendC::TPosition::C2, 1> queueBiasBT; // BT
    AscendC::TQue<AscendC::QuePosition::CO1, 1> queueCL0; // CL0
    AscendC::TBuf<AscendC::TPosition::A2> al0Buf;
    AscendC::TBuf<AscendC::TPosition::B2> bl0Buf;
    AscendC::TQue<AscendC::QuePosition::A1, 2> queueAL1; // AL1
    AscendC::TQue<AscendC::QuePosition::B1, 2> queueBL1; // BL1

    BiasGlobalTensorType biasGM;
    AGlobalTensorType aGM;
    BGlobalTensorType bGM;
    CGlobalTensorType cGM;

    AscendC::LocalTensor<AType> al1;
    AscendC::LocalTensor<BType> bl1;
    AscendC::LocalTensor<BiasType> biasL1;
    AscendC::LocalTensor<AType> wholeAl0Tensor;
    AscendC::LocalTensor<AType> al0;
    AscendC::LocalTensor<BType> wholeBl0Tensor;
    AscendC::LocalTensor<BType> bl0;
    AscendC::LocalTensor<L0CType> biasBT;
    AscendC::LocalTensor<L0CType> cl0;

    uint32_t aL1SpaceSize;
    uint32_t bL1SpaceSize;
    uint32_t cl0Spacesize;
    uint32_t biasl1Spacesize;
    uint32_t biasBTSpacesize;

    // normal: <A1, 1>, <B1, 1>; preload: <A1, 2>, <B1, 2>
    uint64_t padTopL1;
    uint64_t hiLoadL1;
    uint64_t hiIdx;
    uint64_t ddr2l0LoopK;
    uint64_t multiKAL1;
    uint64_t lastLoopKAL1StartPos;
    uint64_t lastLoopKAL1StartPosTail;
    uint64_t kAL1Tail;
    uint64_t maxKAL1Iter;
    uint64_t multiKBL1;
    uint64_t kBL1Tail;
    uint64_t lastLoopKBL1StartPos;
    uint64_t lastLoopKBL1StartPosTail;
    uint64_t maxKBL1Iter;
    uint64_t mStartPos;
    int64_t kAL1 = 0;
    int64_t kBL1 = 0;
    TempIters tempIters;

public:
    __aicore__ inline void Init(ProblemShape problemShape, ConvDim dim, SingleCoreShape SingleCoreShape,
                                BiasGlobalTensorType biasGlobal_, AGlobalTensorType aGlobal_,
                                BGlobalTensorType bGlobal_);
    __aicore__ inline void calAL1SpaceSize();
    __aicore__ inline void calBL1SpaceSize();
    __aicore__ inline void calcl0Spacesize();
    __aicore__ inline void calbiasl1Spacesize();
    __aicore__ inline void CalcHiL1Pad(ConvInterate& iterate, uint64_t mStartPos);
    __aicore__ inline void FreeL1Tensor();
    __aicore__ void IterateK(ConvInterate& iterate, uint64_t mStartPos);
    __aicore__ void GetTensorC(ConvInterate& iterate, CGlobalTensorType cGlobal_);
    __aicore__ void ReduceKPreload(ConvInterate& iterate);
    __aicore__ void ReduceKPreloadFmapIter(ConvInterate& iterate, TempIters& tempIters);
    __aicore__ void ReduceKPreloadWeightIter(ConvInterate& iterate, TempIters& tempIters);
    __aicore__ void ReduceKOpenL0AL0BPingPong(TempIters& tempIters, ConvInterate& iterate,
                                              uint16_t& al0PingPongFlag, bool isFirst);
    __aicore__ inline void LoadBiasFull();
    __aicore__ inline void End();
};

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::calAL1SpaceSize() {
    uint64_t dilatedKernelH = (convShape.kh_ - 1) * convShape.dilationh_ + 1;
    uint64_t mL1Max = mAL1 < singleCoreShapeLocal.singleCoreM ?  mAL1 : singleCoreShapeLocal.singleCoreM;
    uint64_t hoL1Max = Min(mL1Max / convShape.wo_ + CONST_VALUE_2, convShape.ho_);
    uint64_t hiAL1Max = (hoL1Max - 1) * convShape.strideh_ + dilatedKernelH;
    hiAL1Max = hiAL1Max > convShape.hin_ ? convShape.hin_ : hiAL1Max;
    uint64_t cinAInCore = kAL1 / (convShape.kh_ * convShape.kw_);
    aL1SpaceSize = cinAInCore * hiAL1Max * convShape.win_;
    aL1SpaceSize = Align(aL1SpaceSize * sizeOfFmap, C0_SIZE);
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::calBL1SpaceSize() {
    bL1SpaceSize = nBL1 * kBL1;
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::calcl0Spacesize() {
    uint64_t mStep = Align(mL0, M0);
    cl0Spacesize = mStep * nL0;
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::calbiasl1Spacesize() {
    biasl1Spacesize = Align(singleCoreShapeLocal.singleCoreN * sizeOfBias, BLOCK_L0_N * sizeOfBias);
    biasBTSpacesize = nL0;
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::Init(ProblemShape problemShape, ConvDim dim,
                                                                SingleCoreShape singleCoreShape,
                                                                BiasGlobalTensorType biasGlobal_,
                                                                AGlobalTensorType aGlobal_,
                                                                BGlobalTensorType bGlobal_) {
    convShape = problemShape;
    singleCoreShapeLocal = singleCoreShape;
    dimInfo = dim;
    biasGM = biasGlobal_;
    aGM = aGlobal_;
    bGM = bGlobal_;
    kAL1 = kAL1Tiling * convShape.kh_ * convShape.kw_;
    kBL1 = kBL1Tiling * convShape.kh_ * convShape.kw_;
    conv.SetParams(tempIters, problemShape, singleCoreShape);
    calAL1SpaceSize();
    pipe.InitBuffer(queueAL1, DOUBLE_BUF, aL1SpaceSize);

    calBL1SpaceSize();
    pipe.InitBuffer(queueBL1, DOUBLE_BUF, bL1SpaceSize * sizeOfWeight);

    calcl0Spacesize();
    pipe.InitBuffer(queueCL0, 1, cl0Spacesize * sizeOfL0c);
    
    pipe.InitBuffer(al0Buf, L0A_SIZE);
    pipe.InitBuffer(bl0Buf, L0B_SIZE);
    wholeAl0Tensor = al0Buf.template Get<AType>();
    wholeBl0Tensor = bl0Buf.template Get<BType>();

    calbiasl1Spacesize();
    if (convShape.hasbias_) {
        pipe.InitBuffer(queueBiasL1, 1, Align(biasl1Spacesize, C0_SIZE));
        pipe.InitBuffer(queueBiasBT, 1, Align(biasBTSpacesize * sizeOfL0c, BT_SIZE));
    }

    ddr2l0LoopK = CeilDiv(Align(singleCoreShapeLocal.singleCoreCi, k0) * convShape.kh_ * convShape.kw_, kL0);
    multiKAL1 = CeilDiv(kAL1, kL0);
    multiKBL1 = CeilDiv(kBL1, kL0);
    lastLoopKAL1StartPos = ddr2l0LoopK - multiKAL1;
    kAL1Tail = (singleCoreShapeLocal.singleCoreCi * convShape.kh_ * convShape.kw_) % kAL1;
    kAL1Tail = kAL1Tail == 0 ? kAL1 : kAL1Tail;
    lastLoopKAL1StartPosTail = ddr2l0LoopK - CeilDiv(kAL1Tail, kL0);
    maxKAL1Iter = CeilDiv(singleCoreShapeLocal.singleCoreCi * convShape.kh_ * convShape.kw_, kAL1) - 1;

    kBL1Tail = (singleCoreShapeLocal.singleCoreCi * convShape.kh_ * convShape.kw_) % kBL1;
    kBL1Tail = kBL1Tail == 0 ? kBL1 : kBL1Tail;
    lastLoopKBL1StartPos = ddr2l0LoopK - multiKBL1;
    lastLoopKBL1StartPosTail = ddr2l0LoopK - CeilDiv(kBL1Tail, kL0);
    maxKBL1Iter = CeilDiv(singleCoreShapeLocal.singleCoreCi * convShape.kh_ * convShape.kw_, kBL1) - 1;
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::GetTensorC(ConvInterate& iterate, CGlobalTensorType cGlobal_)
{
    if ASCEND_IS_AIC {
        conv.CopyCo1ToGmMMode(cl0, cGlobal_, iterate);
        queueCL0.FreeTensor(cl0);
    }
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::IterateK(ConvInterate& iterate, uint64_t mStartPosTmp)
{
    tempIters.kAL1Iter = 0;
    tempIters.kBL1Iter = 0;
    tempIters.kIter = 0;
    conv.SetMNBeforeIterateK(tempIters, iterate);
    mStartPos = mStartPosTmp;
    if (convShape.hasbias_) {
        biasBT = queueBiasBT.template AllocTensor<L0CType>();
        conv.CopyA1ToC2Bias(biasL1, biasBT, iterate);
        queueBiasBT.EnQue(biasBT);
        biasBT = queueBiasBT.template DeQue<L0CType>();
    }
    cl0 = queueCL0.template AllocTensor<L0CType>();
    ReduceKPreload(iterate);
    FreeL1Tensor();
    queueCL0.EnQue(cl0);
    cl0 = queueCL0.template DeQue<L0CType>();
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::FreeL1Tensor()
{
    queueBL1.FreeTensor(bl1);
    queueAL1.FreeTensor(al1);
    if (convShape.hasbias_) {
        queueBiasBT.FreeTensor(biasBT);
    }
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
 __aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::CalcHiL1Pad(ConvInterate& iterate, uint64_t mStartPos)
{
    uint64_t padBottomL1 = 0;
    uint64_t currentM = mStartPos + iterate.mAL1Iter * mAL1;
    uint64_t maxMAL1Iter = CeilDiv(singleCoreShapeLocal.singleCoreM, mAL1) - 1;
    uint64_t mAL1Tail = singleCoreShapeLocal.singleCoreM % mAL1;
    mAL1Tail = mAL1Tail == 0 ? mAL1 : mAL1Tail;
    uint64_t currentML1 = iterate.mAL1Iter == maxMAL1Iter ? mAL1Tail : mAL1;
    uint64_t hoStartIdx = currentM / convShape.wo_;
    uint64_t hoEndIdx = CeilDiv(currentM + currentML1, convShape.wo_);
    uint64_t dilatedKernelH = (convShape.kh_ - 1) * convShape.dilationh_ + 1;
    hiLoadL1 = ((hoEndIdx - hoStartIdx) - 1) * convShape.strideh_ + dilatedKernelH;

    uint64_t hiStartIdxWithPad = hoStartIdx * convShape.strideh_;
    uint64_t hiEndIdxWithPad = hiStartIdxWithPad + hiLoadL1;
    hiIdx = hiStartIdxWithPad - convShape.padTop_;
    if (hiStartIdxWithPad < convShape.padTop_) {
        hiIdx = 0;
        padTopL1 = convShape.padTop_ - hiStartIdxWithPad;
        hiLoadL1 -= padTopL1;
    }

    if (hiEndIdxWithPad > convShape.hin_ + convShape.padTop_) {
        padBottomL1 = hiEndIdxWithPad - (convShape.hin_ + convShape.padTop_);
        hiLoadL1 -= padBottomL1;
    }
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::ReduceKPreload(ConvInterate& iterate)
{
    if (tempIters.kIter == 0) {
        CalcHiL1Pad(iterate, mStartPos);
        conv.SetLoad3dFMatrix(padTopL1, hiLoadL1);
    }
    al1 = queueAL1.template AllocTensor<AType>();
    conv.CopyGmToA1MMode(aGM, al1, iterate, tempIters, hiIdx, padTopL1, hiLoadL1);
    queueAL1.EnQue(al1);

    tempIters.kBL1Iter = 0;
    bl1 = queueBL1.template AllocTensor<BType>();
    conv.CopyGmToB1(bGM, bl1, iterate, tempIters);
    queueBL1.EnQue(bl1);


    // state
    uint16_t al0PingPongFlag = 0;
    bool isFirst = true;
    while (tempIters.kIter < ddr2l0LoopK) {
        ReduceKPreloadWeightIter(iterate, tempIters);
        ReduceKPreloadFmapIter(iterate, tempIters);
        ReduceKOpenL0AL0BPingPong(tempIters, iterate, al0PingPongFlag, isFirst);
        tempIters.kIter++;
        isFirst = false;
    }
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::ReduceKPreloadFmapIter(ConvInterate& iterate, TempIters& tempIters)
{
    if (tempIters.kIter % multiKAL1 == 0 && tempIters.kIter < lastLoopKAL1StartPos) {
        if (tempIters.kIter != 0) {
            queueAL1.FreeTensor(al1);
        }
        tempIters.kAL1Iter = tempIters.kIter / multiKAL1 + 1;
        if (tempIters.kAL1Iter <= maxKAL1Iter) {
            al1 = queueAL1.template AllocTensor<AType>();
            conv.CopyGmToA1MMode(aGM, al1, iterate, tempIters, hiIdx, padTopL1, hiLoadL1);
            queueAL1.EnQue(al1);
        }
    }
    if ((ddr2l0LoopK % multiKAL1 == 0 &&
        tempIters.kIter == lastLoopKAL1StartPos) ||
        (ddr2l0LoopK % multiKAL1 != 0 &&
        tempIters.kIter == lastLoopKAL1StartPosTail)) {
        queueAL1.FreeTensor(al1);
    }
    if (tempIters.kIter % multiKAL1 == 0) {
        al1 = queueAL1.template DeQue<AType>();
    }
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::ReduceKPreloadWeightIter(ConvInterate& iterate, TempIters& tempIters)
{
    if (tempIters.kIter % multiKBL1 == 0 && tempIters.kIter < lastLoopKBL1StartPos) {
        if (tempIters.kIter != 0) {
            queueBL1.FreeTensor(bl1);
        }
        tempIters.kBL1Iter = tempIters.kIter / multiKBL1 + 1;
        if (tempIters.kBL1Iter <= maxKBL1Iter) {
            bl1 = queueBL1.template AllocTensor<BType>();
            conv.CopyGmToB1(bGM, bl1, iterate, tempIters);
            queueBL1.EnQue(bl1);
        }
    }
    if ((ddr2l0LoopK % multiKBL1 == 0 &&
        tempIters.kIter == lastLoopKBL1StartPos) ||
        (ddr2l0LoopK % multiKBL1 != 0 &&
        tempIters.kIter == lastLoopKBL1StartPosTail)) {
        queueBL1.FreeTensor(bl1);
    }
    if (tempIters.kIter % multiKBL1 == 0) {
        bl1 = queueBL1.template DeQue<BType>();
    }
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::ReduceKOpenL0AL0BPingPong(TempIters& tempIters,
                                                                                     ConvInterate& iterate,
                                                                                     uint16_t& al0PingPongFlag,
                                                                                     bool isFirst)
{
    al0 =
        wholeAl0Tensor[(al0PingPongFlag) *  L0A_HALF_SIZE / sizeOfFmap];
    bl0 =
        wholeBl0Tensor[(al0PingPongFlag) *  L0B_HALF_SIZE / sizeOfWeight];
    event_t eventID = static_cast<event_t>(al0PingPongFlag);
    wait_flag(PIPE_M, PIPE_MTE1, eventID);
 
    tempIters.kAL0Iter = tempIters.kIter % multiKAL1;
    conv.CopyA1ToA2MMode(al1, al0, tempIters, iterate, mStartPos, isFirst);
    tempIters.kBL0Iter = tempIters.kIter % multiKBL1;
    conv.CopyB1ToB2(bl1, bl0, tempIters, isFirst);
    set_flag(PIPE_MTE1, PIPE_M, eventID);
    wait_flag(PIPE_MTE1, PIPE_M, eventID);
    conv.Mmad(al0, bl0, cl0, tempIters, iterate);
    set_flag(PIPE_M, PIPE_MTE1, eventID);
    al0PingPongFlag ^= 1;
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::LoadBiasFull()
{
    if ASCEND_IS_AIC {
        if (convShape.hasbias_) {
            biasL1 = queueBiasL1.template AllocTensor<BiasType>();
            conv.CopyGmToA1Bias(biasGM, biasL1, singleCoreShapeLocal.singleCoreN, 0);
            queueBiasL1.EnQue(biasL1);
            biasL1 = queueBiasL1.template DeQue<BiasType>();
        }
    }
}

BLOCK_CONV_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockConv<BLOCK_CONV_LOCAL_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        if (convShape.hasbias_) {
            queueBiasL1.FreeTensor(biasL1);
        }
        queueAL1.FreeAllEvent();
        queueBL1.FreeAllEvent();
        queueBiasL1.FreeAllEvent();
        queueBiasBT.FreeAllEvent();
        queueCL0.FreeAllEvent();
    }
}


} // namespace Block
} // namespace Conv
} // namespace Act
#endif
