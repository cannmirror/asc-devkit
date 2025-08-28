/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_bp_input_init_func.h
 * \brief
 */

#ifndef AICORE_ADV_API_DETAIL_CONV_BACKPROP_CONV3D_BACKPROP_INPUT_CONV3D_BP_INPUT_INIT_FUNC_H
#define AICORE_ADV_API_DETAIL_CONV_BACKPROP_CONV3D_BACKPROP_INPUT_CONV3D_BP_INPUT_INIT_FUNC_H

#if __CCE_AICORE__ == 220
#include "dav_v220/conv3d_bp_input_mte2_sub_func.h"
#include "dav_v220/conv3d_bp_input_sub_func.h"
#include "dav_v220/conv3d_bp_kernel_split.h"
#endif

namespace ConvBackpropInputFunc {
template <class Intf>
__aicore__ inline void InitStepMParams(Intf* self)
{
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        self->ctx.mIter_ = ConvBackpropApi::Ceil(self->ctx.splitSingleShapeM_, self->ctx.tiling_->baseM);
        self->ctx.tailM_ = self->ctx.splitSingleShapeM_ - (self->ctx.mIter_ - 1) * self->ctx.tiling_->baseM;
    } else {
        self->ctx.mIter_ = ConvBackpropApi::Ceil(self->ctx.singleShapeM_, self->ctx.tiling_->baseM);
        self->ctx.tailM_ = self->ctx.singleShapeM_ - (self->ctx.mIter_ - 1) * self->ctx.tiling_->baseM;
    }
#ifdef ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((self->ctx.mIter_ > 0),
        { KERNEL_LOG(KERNEL_ERROR, "ctx.mIter_ is %d , which should be larger than 0", self->ctx.mIter_); });
#endif
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf* self)
{
    uint64_t tmpSingleCoreK = static_cast<uint64_t>(self->ctx.singleShapeCout1_) * self->ctx.HkWkC0_;
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        tmpSingleCoreK = static_cast<uint64_t>(self->ctx.tiling_->cout1G) * self->ctx.splitHkWkC0_;
    }
    uint64_t kIter = ConvBackpropApi::Ceil(tmpSingleCoreK, self->ctx.tiling_->baseK);
    self->ctx.kIter_ = kIter;
    self->ctx.tailK_ = tmpSingleCoreK - (kIter - 1) * self->ctx.tiling_->baseK;
#ifdef ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((self->ctx.kIter_ > 0),
        { KERNEL_LOG(KERNEL_ERROR, "ctx.kIter_ is %d , which should be larger than 0", self->ctx.kIter_); });
#endif
    self->ctx.stepKaRound_ = ConvBackpropApi::Ceil(kIter, self->ctx.tiling_->stepKa);
    self->ctx.stepKbRound_ = ConvBackpropApi::Ceil(kIter, self->ctx.tiling_->stepKb);
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf* self)
{
    uint64_t singleShapeCinAlign = self->ctx.singleShapeCin1_ * self->ctx.tiling_->c0;
    self->ctx.nIter_ = ConvBackpropApi::Ceil(singleShapeCinAlign, self->ctx.tiling_->baseN);
    self->ctx.tailN_ = singleShapeCinAlign - (self->ctx.nIter_ - 1) * self->ctx.tiling_->baseN;
#ifdef ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((self->ctx.nIter_ > 0),
        { KERNEL_LOG(KERNEL_ERROR, "ctx.nIter_ is %d , which should be larger than 0", self->ctx.nIter_); });
#endif
}

template <class Intf>
__aicore__ inline void InitParams(Intf* self)
{
    self->ctx.baseMN_ = self->ctx.tiling_->baseM * self->ctx.tiling_->baseN;
    self->ctx.isFirstIter_ = true;
    self->ctx.usingCacheC1Ping_ = true;
    self->ctx.HkWk_ = self->ctx.tiling_->hk * self->ctx.tiling_->wk;
    self->ctx.HkWkC0_ = self->ctx.tiling_->hk * self->ctx.tiling_->wk * self->ctx.tiling_->c0;
    self->ctx.DkHkWkC0_ = self->ctx.tiling_->dk * self->ctx.tiling_->hk * self->ctx.tiling_->wk * self->ctx.tiling_->c0;

    self->ctx.curCin1Size_ = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN / self->ctx.tiling_->c0;
    self->ctx.isB1FullLoadFlag_ = (self->ctx.tiling_->dk == 1 && self->ctx.tiling_->bl1Pbuffer == 1
                                      && self->ctx.tiling_->baseK * self->ctx.tiling_->stepKb
                                             >= self->ctx.tiling_->singleCoreCout * self->ctx.HkWk_)
                                  && (self->ctx.curCin1Size_ >= self->ctx.tiling_->singleCoreCin1);
    self->ctx.hwI_ = static_cast<uint64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    self->ctx.hwO_ = static_cast<uint64_t>(self->ctx.tiling_->ho) * self->ctx.tiling_->wo;
    if constexpr (std::is_same<typename Intf::DstT, float>::value) {
        self->ctx.alignedCout1_ = DivCeil(self->ctx.tiling_->cout1G * self->ctx.tiling_->c0, AscendC::BLOCK_CUBE);
        self->ctx.alignedCout_ = self->ctx.alignedCout1_ * AscendC::BLOCK_CUBE;
    }
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
#ifdef ASCENDC_CPU_DEBUG
        ASCENDC_ASSERT((self->ctx.tiling_->hk >= self->ctx.tiling_->strideH),
            { KERNEL_LOG(KERNEL_ERROR, "kernelH should be GE strideH"); });
        ASCENDC_ASSERT((self->ctx.tiling_->wk >= self->ctx.tiling_->strideW),
            { KERNEL_LOG(KERNEL_ERROR, "kernelW should be GE strideW"); });
#endif
        // 泛化时需考虑不整除场景
        self->ctx.splitHk_ = self->ctx.tiling_->hk / self->ctx.tiling_->strideH;
        self->ctx.splitWk_ = self->ctx.tiling_->wk / self->ctx.tiling_->strideW;
        self->ctx.splitHkWk_ = self->ctx.splitHk_ * self->ctx.splitWk_;
        self->ctx.splitHkWkC0_ = self->ctx.splitHkWk_ * self->ctx.tiling_->c0;
        self->ctx.splitHi_ = self->ctx.tiling_->hi / self->ctx.tiling_->strideH;
        self->ctx.splitWi_ = self->ctx.tiling_->wi / self->ctx.tiling_->strideW;
        self->ctx.channelSize_ = (self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK) / self->ctx.splitHkWk_;
        self->ctx.curHoSize_ = CalFmapHForKernelSplit<Intf>(self, self->ctx.tiling_->baseM * self->ctx.tiling_->stepM);
    } else {
        self->ctx.channelSize_ = (self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK) / self->ctx.HkWk_;
        self->ctx.curHoSize_ = CalFmapH<Intf>(self, self->ctx.tiling_->baseM * self->ctx.tiling_->stepM);
    }
    self->ctx.l0aPingPongFlag_ = 0;
    self->ctx.useL0PingPong_ = (self->ctx.tiling_->al0Pbuffer - 1) & (self->ctx.tiling_->bl0Pbuffer - 1);
    InitLoadToA2Params<Intf>(self);
    if constexpr ((std::is_same<typename Intf::SrcT, bfloat16_t>::value)
                  || (std::is_same<typename Intf::SrcT, half>::value)) {
        InitLoadToB2Params<Intf>(self);
    }
}

template <class Intf>
__aicore__ inline void InitTque(Intf* self)
{
    // fp32场景下baseK可能为8的倍数，非16倍数，但是GM中K0一定是16倍数，但是实际K可能仅有8，额外为padding数据
    uint32_t bMatrixByteSize = 0;
    uint32_t aMatrixByteSize = 0;
    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        uint32_t hoSize = (self->ctx.curHoSize_ < self->ctx.tiling_->ho) ? self->ctx.curHoSize_ : self->ctx.tiling_->ho;
        // 泛化时，每个小kernel需要加载的wo大小可能不一样，可能是wo wo-1 wo-2 ...
        aMatrixByteSize = hoSize * (self->ctx.tiling_->wo - 1) * self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK
                          / self->ctx.splitHkWk_ * sizeof(typename Intf::SrcT);
    } else {
        uint32_t hoSize = self->ctx.curHoSize_;
        uint64_t hoExpand = static_cast<uint64_t>(self->ctx.tiling_->ho - 1) * self->ctx.tiling_->strideH + 1;
        if (hoExpand < static_cast<uint64_t>(self->ctx.curHoSize_)) {
            hoSize = static_cast<uint32_t>(hoExpand);
        }
        aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW * self->ctx.tiling_->stepKa
                          * self->ctx.tiling_->baseK / self->ctx.HkWk_ * sizeof(typename Intf::SrcT);
    }

    if constexpr (std::is_same<typename Intf::SrcT, float>::value) {
        bMatrixByteSize =
            self->ctx.tiling_->stepN * self->ctx.tiling_->baseN
            * AscendC::DivCeil(self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK / self->ctx.HkWkC0_, 2)
            * self->ctx.HkWk_ * AscendC::BLOCK_CUBE * sizeof(typename Intf::SrcT);
    } else {
        bMatrixByteSize = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN * self->ctx.tiling_->stepKb
                          * self->ctx.tiling_->baseK * sizeof(typename Intf::SrcT);
    }

    self->ctx.pipe_.InitBuffer(self->ctx.a1Ping_, 1, aMatrixByteSize);
    self->ctx.pipe_.InitBuffer(self->ctx.b1Ping_, 1, bMatrixByteSize);
    if (self->ctx.tiling_->al1Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.a1Pong_, 1, aMatrixByteSize);
    }
    if (self->ctx.tiling_->bl1Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.b1Pong_, 1, bMatrixByteSize);
    }

    self->ctx.pipe_.InitBuffer(self->ctx.c1Ping_, 1, self->ctx.baseMN_ * sizeof(typename Intf::L0cT));
    if (self->ctx.tiling_->cl0Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.c1Pong_, 1, self->ctx.baseMN_ * sizeof(typename Intf::L0cT));
    }
    self->ctx.pipe_.InitBuffer(self->ctx.l0aBuf_, AscendC::TOTAL_L0A_SIZE);
    self->ctx.pipe_.InitBuffer(self->ctx.l0bBuf_, AscendC::TOTAL_L0B_SIZE);
}

template <class Intf>
static __aicore__ inline void Compute(Intf* self)
{
    // 先刷新h方向的值，方便判断是否为有效计算
    UpdateLoadToA2ParamsM<Intf>(self);

    // 在跳过计算逻辑中，如果有部分无需跳过的操作逻辑。后续如果有类似逻辑，可以在此处继续增加
    // 当前已存在的操作为isFreeB1_为true的情况(B1全加载且循环至最后一块运算空间，需要释放B1空间)。此时预期的只有ping空间被使用
    if (!self->ctx.needComputeFlag_) {
        if (self->ctx.isFreeB1_ && !self->ctx.isLoadB1_) {
            self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
        }
        return;
    }

    if ASCEND_IS_AIV {
        return;
    }

    InitMmadParams<Intf>(self);
    if constexpr ((std::is_same<typename Intf::SrcT, bfloat16_t>::value)
                  || (std::is_same<typename Intf::SrcT, half>::value)) {
        if (unlikely(self->ctx.curNL0Idx_ == 0 || self->ctx.curNL0Idx_ + 1 == self->ctx.nIter_)) {
            UpdateLoadToB2ParamsN<Intf>(self);
        }
    }

    bool isFirstDk = true;
    bool a1PingPongFlag = true;
    bool b1PingPongFlag = true;
    AscendC::LocalTensor<typename Intf::SrcT> l0a;
    AscendC::LocalTensor<typename Intf::SrcT> l0b;
    AscendC::LocalTensor<typename Intf::L0cT> l0c;
    uint8_t l0aPingPongFlag = self->ctx.l0aPingPongFlag_;
    constexpr uint32_t l0aPingPongAddr = AscendC::TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = AscendC::TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);

    if (self->ctx.usingCacheC1Ping_) {
        l0c = self->ctx.c1Ping_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.c1Pong_.template AllocTensor<typename Intf::L0cT>();
    }
    self->ctx.needComputeKFlag_ = false;
    for (uint64_t curKdIdx = 0; curKdIdx < self->ctx.tiling_->dk; curKdIdx++) {
        int64_t dTmp = 0;
        if (unlikely(self->ctx.tiling_->strideD > self->ctx.tiling_->dk)) {
            dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront;
            if (ConvBackpropApi::CalcRemainder(dTmp, self->ctx.tiling_->strideD) >= self->ctx.tiling_->dk
                || dTmp / self->ctx.tiling_->strideD >= self->ctx.tiling_->dout) {
                continue;
            }
        } else {
            // 由于膨胀卷积使dk的位置发生改变，求解dout_idx时，dk_idx需乘上膨胀系数再参与计算，才能求取正确的索引
            dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - curKdIdx * self->ctx.tiling_->dilationD;
            if (dTmp < 0 || ConvBackpropApi::CalcRemainder(dTmp, self->ctx.tiling_->strideD) > 0
                || dTmp >= self->ctx.tiling_->dout * self->ctx.tiling_->strideD) {
                continue;
            }
        }
        self->ctx.needComputeKFlag_ = true;
        int64_t curDoutIdx = dTmp;
        if (self->ctx.tiling_->strideD > 1) {
            curDoutIdx = dTmp / self->ctx.tiling_->strideD;
        }

        uint32_t kaIdx = 0;
        uint32_t kbIdx = 0;
        uint64_t kaStepIdx = 0;
        uint64_t kbStepIdx = 0;
        self->ctx.load3d_.kExtension = self->ctx.tiling_->baseK;
        self->ctx.mmad_.k = self->ctx.tiling_->baseK;
        self->ctx.curLoadKal1_ = self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
        self->ctx.curLoadKbl1_ = self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;
        for (uint64_t kIdx = 0; kIdx < self->ctx.kIter_; kIdx++) {
            bool isLastKIdx = (kIdx + 1 == self->ctx.kIter_);
            if (isLastKIdx) {
                self->ctx.load3d_.kExtension = self->ctx.tailK_;
                self->ctx.mmad_.k = self->ctx.tailK_;
            }
            if (kIdx == self->ctx.kIterStepKaTail) {
                self->ctx.curLoadKal1_ = (self->ctx.stepKaTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
            }
            if (kIdx == self->ctx.kIterStepKbTail) {
                self->ctx.curLoadKbl1_ = (self->ctx.stepKbTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
            }
            /*
            通过M*K的奇偶判断load到L1A ping还是L1A pong, BL1同理
                        kL1Idx=0  kL1Idx=1 kL1Idx=2
                        ----------------------------
            mL1Idx=0    |  ping  |  pong  |  ping  |
                        ----------------------------
            mL1Idx=1    |  pong  |  ping  |  pong  |
                        ----------------------------
            mL1Idx=2    |  ping  |  pong  |  ping  |
                        ----------------------------
            */
            bool isLoadA1 = kaIdx == 0;
            if (isLoadA1 && self->ctx.tiling_->al1Pbuffer > 1) {
                // 此处默认stepM = 1
                a1PingPongFlag = ((self->ctx.curML1Idx_ * self->ctx.stepKaRound_ + kaStepIdx + 1) & 1);
            }
            ConvBackpropInputFunc::LoadToA1<Intf, typename Intf::SrcT>(
                self, kIdx, curDoutIdx, a1PingPongFlag, isLoadA1);

            bool isLoadB1 = kbIdx == 0;
            if (isLoadB1 && self->ctx.tiling_->bl1Pbuffer > 1) {
                // 此处默认stepN = 1
                b1PingPongFlag = ((self->ctx.curNL1Idx_ * self->ctx.stepKbRound_ + kbStepIdx + 1) & 1);
            }
            ConvBackpropInputFunc::LoadToB1<Intf, typename Intf::SrcT>(self, kIdx, curKdIdx, b1PingPongFlag, isLoadB1);

            l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
            if (l0aPingPongFlag) {
                l0a = l0a[l0aPingPongAddr];
            }

            self->ctx.load3d_.kStartPt = kaIdx * self->ctx.tiling_->baseK;

            if (unlikely(isLoadA1)) {
                if (a1PingPongFlag) {
                    self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();
                } else {
                    self->ctx.cacheA1BufPong_ = self->ctx.a1Pong_.template DeQue<typename Intf::SrcT>();
                }
            }

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0aPingPongFlag);
            if (a1PingPongFlag) {
                LoadToA2<Intf>(self, self->ctx.cacheA1BufPing_, l0a);
            } else {
                LoadToA2<Intf>(self, self->ctx.cacheA1BufPong_, l0a);
            }

            bool isLastStepKa = kaIdx + 1 == self->ctx.tiling_->stepKa;
            if (isLastStepKa || isLastKIdx) {
                if (a1PingPongFlag) {
                    self->ctx.a1Ping_.FreeTensor(self->ctx.cacheA1BufPing_);
                } else {
                    self->ctx.a1Pong_.FreeTensor(self->ctx.cacheA1BufPong_);
                }
            }

            l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
            if (l0aPingPongFlag) {
                l0b = l0b[l0bPingPongAddr];
            }

            if (unlikely(isLoadB1
                         && (!self->ctx.isB1FullLoadFlag_ || (self->ctx.isB1FullLoadFlag_ && self->ctx.isLoadB1_)))) {
                if (b1PingPongFlag) {
                    self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
                } else {
                    self->ctx.cacheB1BufPong_ = self->ctx.b1Pong_.template DeQue<typename Intf::SrcT>();
                }
                if (self->ctx.isLoadB1_) {
                    self->ctx.isLoadB1_ = false;
                }
            }

            if constexpr ((std::is_same<typename Intf::SrcT, bfloat16_t>::value)
                          || (std::is_same<typename Intf::SrcT, half>::value)) {
                if (unlikely(kIdx == 0 || kIdx == self->ctx.kIterStepKbTail)) {
                    UpdateLoadToB2ParamsK<Intf>(self);
                }
            }
            if (b1PingPongFlag) {
                LoadToB2<Intf>(self, self->ctx.cacheB1BufPing_, kbIdx, kIdx, b1PingPongFlag, l0b);
            } else {
                LoadToB2<Intf>(self, self->ctx.cacheB1BufPong_, kbIdx, kIdx, b1PingPongFlag, l0b);
            }

            bool isLastStepKb = kbIdx + 1 == self->ctx.tiling_->stepKb;
            if ((isLastStepKb || isLastKIdx)
                && (!self->ctx.isB1FullLoadFlag_ || (self->ctx.isB1FullLoadFlag_ && self->ctx.isFreeB1_))) {
                if (b1PingPongFlag) {
                    self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
                } else {
                    self->ctx.b1Pong_.FreeTensor(self->ctx.cacheB1BufPong_);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0aPingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0aPingPongFlag);

            MmadLocal<Intf>(self, l0a, l0b, l0c);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0aPingPongFlag);
            if (unlikely(isFirstDk && kIdx == 0)) {
                self->ctx.mmad_.cmatrixInitVal = 0;
            }

            l0aPingPongFlag ^= self->ctx.useL0PingPong_;
            if (isLastStepKa) {
                ++kaStepIdx;
                kaIdx = 0;
            } else {
                ++kaIdx;
            }
            if (isLastStepKb) {
                ++kbStepIdx;
                kbIdx = 0;
            } else {
                ++kbIdx;
            }
        }
        isFirstDk = false;
    }
    if (self->ctx.usingCacheC1Ping_) {
        self->ctx.c1Ping_.EnQue(l0c);
    } else {
        self->ctx.c1Pong_.EnQue(l0c);
    }
    self->ctx.l0aPingPongFlag_ = l0aPingPongFlag;

    if constexpr (Intf::conv3dConfig.enableKernelSplit) {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
    }
}
} // namespace ConvBackpropInputFunc
#endif
