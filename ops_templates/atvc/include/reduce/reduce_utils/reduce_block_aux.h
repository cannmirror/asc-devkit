/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file reduce_block_aux.h
 * \brief reduce schedule aux
 */

#ifndef ATVC_REDUCE_UTILS_BLOCK_AUX_H
#define ATVC_REDUCE_UTILS_BLOCK_AUX_H
#include "common/kernel_utils.h"
#include "common/const_def.h"
#include "reduce/common/patterns.h"
#include "reduce_block_aux_util.h"

namespace {
template <int32_t dim>
struct SliceView {
    uint64_t addr;
    uint64_t burstLen;
    uint64_t axisSize;
    struct {
        uint64_t repeat = 1;
        uint64_t srcStride = 0;
        uint64_t dstStride = 0;
        uint64_t idx = 0;
        bool isAxisA = false;
    } axis[dim];
};
}
namespace ATVC {
namespace KernelUtils {
namespace Reduce {
template <typename DataType, auto LoopInfo, class ReduceOp, class InDtype, class OutDtype, class PreOp = void,
    class EOp = void>
struct ReduceBlockAux {
    using Pattern = typename ReducePattern::GetPattern<LoopInfo->patternID>::T;
    using InnerPattern = typename ReducePattern::GetPattern<LoopInfo->innerPatternID>::T;
    using PromoteDataType = typename ReduceOp::PromoteDataType;
    constexpr static int32_t Dim = Pattern::Dim;
    constexpr static uint32_t InputSize = ReduceOp::InputSize;
    constexpr static uint32_t OutputSize = ReduceOp::OutputSize;
    constexpr static int32_t ELEMENT_ONE_REPEAT = Platform::GetVRegSize() / sizeof(PromoteDataType);
    constexpr static uint64_t BLOCK_SIZE_BYTE = Platform::GetUbBlockSize();

public:
    uint64_t loopAStartIndex_;
    uint64_t loopAEndIndex_;
    uint64_t loopAAxisStep_;
    uint64_t ubFactorA_;

    uint64_t loopRStartIndex_;
    uint64_t loopREndIndex_;
    uint64_t loopRAxisStep_;
    uint64_t ubFactorR_;

    AscendC::GlobalTensor<InDtype>* input_;
    AscendC::GlobalTensor<OutDtype>* output_;

    int64_t rCount_;
    int64_t bisectionPos_;
    int64_t cacheCount_;
    int64_t bisectionTail_;

    uint64_t aOutBurstLen_;
    uint64_t aOutNBurst_;

    struct {
        uint64_t start = 0;
        uint64_t stride = 1; // 拷贝步长
    } iterAddr_[Dim];

    const DataType* tiling_;
    ReduceOp* op_;

public:
    __aicore__ inline ReduceBlockAux(ReduceOp *op, AscendC::GlobalTensor<InDtype> *input,
        AscendC::GlobalTensor<OutDtype> *output, const DataType *tiling)
    {
        this->op_ = op;
        this->input_ = input;
        this->output_ = output;
        this->tiling_ = tiling;
        for (uint64_t i = 0; i < Dim; i++) {
            iterAddr_[i].stride = tiling_->shape[i];
        }
    }

    __aicore__ inline void SetLoopRange()
    {
        int32_t blockId = AscendC::GetBlockIdx();
        if constexpr (IsBlockCutA<LoopInfo>()) {
            loopAStartIndex_ = blockId * tiling_->factorACntPerCore;
            loopAEndIndex_ = loopAStartIndex_ + tiling_->factorACntPerCore;
            if (unlikely(loopAEndIndex_ > tiling_->factorATotalCnt)) {
                loopAEndIndex_ = tiling_->factorATotalCnt;
            }
            int32_t aAxisIdx = LoopInfo->loopACount - 1;
            int32_t aAxis = LoopInfo->loopAAxis[aAxisIdx];
            loopAAxisStep_ = OpsUtils::CeilDiv(tiling_->shape[aAxis], tiling_->ubFactorA);
            if constexpr (LoopInfo->loopInnerRCount > 0) {
                constexpr int32_t rAxisIdx = LoopInfo->loopInnerRCount - 1;
                constexpr int32_t rAxis = LoopInfo->loopInnerRAxis[rAxisIdx];
                loopRAxisStep_ = OpsUtils::CeilDiv(tiling_->shape[rAxis], tiling_->ubFactorR);
            }
        } else {
            loopRStartIndex_ = blockId / tiling_->groupR * tiling_->factorRTotalCnt +
                               blockId % tiling_->groupR * tiling_->factorRCntPerCore;
            loopREndIndex_ = loopRStartIndex_ + tiling_->factorRCntPerCore;
            uint64_t maxRCnt = (blockId / tiling_->groupR + 1) * tiling_->factorRTotalCnt;
            uint64_t totalCnt = tiling_->factorATotalCnt * tiling_->factorRTotalCnt;
            maxRCnt = maxRCnt > totalCnt ? totalCnt : maxRCnt;
            if (unlikely(loopRStartIndex_ > maxRCnt)) {
                loopRStartIndex_ = maxRCnt;
            }
            if (unlikely(loopREndIndex_ > maxRCnt)) {
                loopREndIndex_ = maxRCnt;
            }

            int32_t rAxisIdx = LoopInfo->loopRCount - 1;
            int32_t rAxis = LoopInfo->loopRAxis[rAxisIdx];
            loopRAxisStep_ = OpsUtils::CeilDiv(tiling_->shape[rAxis], tiling_->ubFactorR);  // 切分轴Rfactor的个数

            if constexpr (LoopInfo->loopACount > 0) {
                int32_t aAxisIdx = LoopInfo->loopACount - 1;
                int32_t aAxis = LoopInfo->loopAAxis[aAxisIdx];
                loopAAxisStep_ = OpsUtils::CeilDiv(tiling_->shape[aAxis], tiling_->ubFactorA);
            }
        }
        ubFactorA_ = tiling_->ubFactorA;
        ubFactorR_ = tiling_->ubFactorR;
    }

    template <class... Args>
    __aicore__ inline void Process(Args... args)
    {
        SetLoopRange();
        // 构造UB内轴index数组
        // 1、尾轴
        // 2、非循环轴
        // 3、核外A循环轴最内轴且UbFactorA > 1
        // 4、核外R循环轴最内轴且UbFactorR > 1
        // 5、核内R循环轴最内轴且UbFactorR > 1
        if constexpr (LoopInfo->loopRCount == 0) {
            rCount_ = tiling_->factorRCntPerCore;
        } else {
            rCount_ = loopREndIndex_ - loopRStartIndex_;
        }
        bisectionPos_ = KernelUtils::FindNearestPower2(rCount_);
        cacheCount_ = KernelUtils::CalLog2(bisectionPos_) + 1;
        bisectionTail_ = rCount_ - bisectionPos_;
        SetEvent<AscendC::HardEvent::V_MTE2>(AscendC::HardEvent::V_MTE2);
        if constexpr (LoopInfo->loopRCount == 0) {
            for (uint64_t i = loopAStartIndex_; i < loopAEndIndex_; i++) {
                CalcIterA<LoopInfo->loopACount>(i);
                IterateInnerA<0, LoopInfo->loopInnerACount>(args...);
            }
        } else {
            IterateInnerA<0, LoopInfo->loopInnerACount>(args...);
        }
    }

    template <int32_t start = 0, int32_t end = 0, class... Args>
    __aicore__ inline void IterateInnerA(Args... args)
    {
        if constexpr (start == end) {
            if constexpr (LoopInfo->reduceDichotomy) {
                int64_t tmpBufOffest = 0;
                Shape<InnerPattern::Dim> shape;
                if constexpr (LoopInfo->loopRCount == 0) {
                    LinearComputeR(tmpBufOffest, shape, args...);
                    CopyOut(tmpBufOffest, shape, args...);
                } else {
                    LinearComputeR(tmpBufOffest, shape, args...);
                    CopyOutGroup(tmpBufOffest, shape, args...);
                }
                SetEvent<AscendC::HardEvent::MTE3_V>(AscendC::HardEvent::MTE3_V);
            }
        } else {
            constexpr int32_t axis = LoopInfo->loopInnerAAxis[start];
            uint64_t shape = tiling_->shape[axis];
            if constexpr (start + 1 == end) {
                uint64_t loopSize = shape / this->ubFactorA_;
                uint64_t tail = shape - loopSize * this->ubFactorA_;
                this->iterAddr_[axis].start = 0;
                this->iterAddr_[axis].stride = this->ubFactorA_;

                for (uint64_t i = 0; i < loopSize; i++) {
                    IterateInnerA<start + 1, end>(args...);
                    this->iterAddr_[axis].start += this->ubFactorA_;
                }

                if (tail) {
                    this->iterAddr_[axis].stride = shape - this->iterAddr_[axis].start;
                    IterateInnerA<start + 1, end>(args...);
                }
            } else {
                for (uint64_t i = 0; i < shape; i++) {
                    this->iterAddr_[axis].start = i;
                    IterateInnerA<start + 1, end>(args...);
                }
            }
        }
    }

    template <bool isPadding, bool IsTail, class V, class U>
    __aicore__ inline void PrePareReduce(int64_t index, V& view, U& shape, AscendC::LocalTensor<InDtype>& ubTensor,
                                      AscendC::LocalTensor<PromoteDataType>& computeTensor)
    {
        if constexpr (IsTail) {
            index = index + bisectionPos_;
        }
        if constexpr (LoopInfo->loopRCount > 0) {
            CalcIterR<LoopInfo->loopRCount>(index + loopRStartIndex_);
        } else {
            CalcInnerIterR<LoopInfo->loopInnerRCount>(index);
        }
        CalcCopyInParam(view);
        if (index == 0) {
            CalcInnerShape(view, shape);
        }
        if constexpr (AscendC::IsSameType<PromoteDataType, InDtype>::value) {
            CopyIn<isPadding>(view, shape, ubTensor);
            SetEvent<AscendC::HardEvent::MTE2_V>(AscendC::HardEvent::MTE2_V);
            computeTensor = ubTensor;
        } else {
            // AllocComputeTensorAux 的index 外部不需要感知
            op_->ReduceOp::template AllocTensorAux<false>(computeTensor);
            CopyIn<isPadding>(view, shape, ubTensor);
            SetEvent<AscendC::HardEvent::MTE2_V>(AscendC::HardEvent::MTE2_V);
            AscendC::Cast(computeTensor, ubTensor, AscendC::RoundMode::CAST_NONE, shape.value[0] * shape.value[1]);
            op_->FreeTensorAux(ubTensor);
        }
    }

    template <class V, class... Args>
    __aicore__ inline void LinearComputeR(int64_t& tmpBufOffest, V& shape, Args... args)
    {
        SliceView<MAX_DIM> view;
        for (int64_t i = 0; i < bisectionTail_; i++) {
            AscendC::LocalTensor<InDtype> tensorLeft;
            op_->ReduceOp::template AllocTensorAux<true>(tensorLeft);
            AscendC::LocalTensor<PromoteDataType> computeLeft;
            PrePareReduce<(!InnerPattern::TailA), false>(i, view, shape, tensorLeft, computeLeft);
            
            AscendC::LocalTensor<InDtype> tensorRight;
            op_->ReduceOp::template AllocTensorAux<true>(tensorRight);
            AscendC::LocalTensor<PromoteDataType> computeRight;
            PrePareReduce<(!InnerPattern::TailA), true>(i, view, shape, tensorRight, computeRight);
            ComputeMerge(shape, computeLeft, computeRight, args...);

            // fp32 tensorLeft和computeLeft是同一个tensor，fp16 computeLeft在free时不用index
            op_->ReduceOp::template FreeTensorAux(computeRight);
            op_->compute_.template UpdateCache<Pattern, V>(i, shape, op_->tempBuf_, op_->computeRes_);
        }

        for (int64_t i = bisectionTail_; i < bisectionPos_; i++) {
            AscendC::LocalTensor<InDtype> tensor;
            op_->ReduceOp::template AllocTensorAux<true>(tensor);
            AscendC::LocalTensor<PromoteDataType> computeLeft;
            PrePareReduce<(!InnerPattern::TailA && Pattern::Dim > 2), false>(i, view, shape, tensor, computeLeft);
            Compute(shape, computeLeft, args...);
            op_->ReduceOp::template FreeTensorAux(computeLeft);
            op_->compute_.template UpdateCache<Pattern>(i, shape, op_->tempBuf_, op_->computeRes_);
        }
        int64_t dimA = Pattern::TailA ? shape.value[1] : shape.value[0];
        int64_t cacheStride = OpsUtils::CeilDiv(dimA, static_cast<int64_t>(ELEMENT_ONE_REPEAT)) * ELEMENT_ONE_REPEAT;
        tmpBufOffest = (cacheCount_ - 1) * cacheStride;
    }

    template <int32_t LoopInnerRIdx>
    __aicore__ inline void CalcInnerIterR(uint64_t basicBlockIdx)
    {
        if constexpr (LoopInnerRIdx != 0) {
            constexpr auto axis = LoopInfo->loopInnerRAxis[LoopInnerRIdx - 1];
            if constexpr (LoopInnerRIdx == LoopInfo->loopInnerRCount) {
                // 最内层循环
                auto cur = basicBlockIdx % this->loopRAxisStep_;
                this->iterAddr_[axis].start = cur * this->ubFactorR_;
                this->iterAddr_[axis].stride = tiling_->shape[axis] - this->iterAddr_[axis].start;
                if (likely(this->iterAddr_[axis].stride >= this->ubFactorR_)) {
                    this->iterAddr_[axis].stride = this->ubFactorR_;
                }
                CalcInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / this->loopRAxisStep_);
            } else {
                this->iterAddr_[axis].start = basicBlockIdx % tiling_->shape[axis];
                this->iterAddr_[axis].stride = 1;
                CalcInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / tiling_->shape[axis]);
            }
        }
    }

    template <int32_t LoopAIdx>
    __aicore__ inline void CalcIterA(uint64_t step)
    {
        if constexpr (LoopAIdx != 0) {
            constexpr auto axis = LoopInfo->loopAAxis[LoopAIdx - 1];
            if constexpr (LoopAIdx == LoopInfo->loopACount) {
                // 切分轴
                auto cur = step % this->loopAAxisStep_;
                this->iterAddr_[axis].start = cur * this->ubFactorA_;
                this->iterAddr_[axis].stride = tiling_->shape[axis] - this->iterAddr_[axis].start;
                if (likely(this->iterAddr_[axis].stride >= this->ubFactorA_)) {
                    this->iterAddr_[axis].stride = this->ubFactorA_;
                }

                if constexpr (LoopAIdx > 0) {
                    CalcIterA<LoopAIdx - 1>(step / this->loopAAxisStep_);
                }
            } else {
                this->iterAddr_[axis].start = step % tiling_->shape[axis];
                this->iterAddr_[axis].stride = 1;
                CalcIterA<LoopAIdx - 1>(step / tiling_->shape[axis]);
            }
        }
    }

    template <int32_t LoopRIdx>
    __aicore__ inline void CalcIterR(uint64_t step)
    {
        uint64_t temp = step;
        if constexpr (LoopRIdx != 0) {
            for (auto idx = LoopInfo->loopRCount - 1; idx > -1; --idx) {
                if (idx == LoopInfo->loopRCount - 1) {
                    constexpr auto axis = LoopInfo->loopRAxis[LoopInfo->loopRCount - 1];
                    auto cur = temp % this->loopRAxisStep_;
                    this->iterAddr_[axis].start = cur * this->ubFactorR_;
                    this->iterAddr_[axis].stride = tiling_->shape[axis] - this->iterAddr_[axis].start;
                    if (likely(this->iterAddr_[axis].stride >= this->ubFactorR_)) {
                        this->iterAddr_[axis].stride = this->ubFactorR_;
                    }
                    temp = temp / this->loopRAxisStep_;
                } else {
                    auto axis = LoopInfo->loopRAxis[idx];
                    if (IsLoopSpliteAAxis<LoopInfo>(axis)) {
                        auto cur = temp % this->loopAAxisStep_;
                        this->iterAddr_[axis].start = cur * this->ubFactorA_;
                        this->iterAddr_[axis].stride = tiling_->shape[axis] - this->iterAddr_[axis].start;
                        if (likely(this->iterAddr_[axis].stride >= this->ubFactorA_)) {
                            this->iterAddr_[axis].stride = this->ubFactorA_;
                        }
                        temp = temp / this->loopAAxisStep_;
                    } else {
                        this->iterAddr_[axis].start = temp % tiling_->shape[axis];
                        this->iterAddr_[axis].stride = 1;
                        temp = temp / tiling_->shape[axis];
                    }
                }
            }
        }
    }

    template <class V>
    __aicore__ inline void CalcCopyInParam(V& view)
    {
        uint64_t addrOffset = 0;
        for (int32_t i = 0; i < Dim; i++) {
            addrOffset += iterAddr_[i].start * tiling_->stride[i];
        }
        constexpr static auto burstLenAxis = Dim - 1;  // 获取第一个循环轴
        view.addr = addrOffset;                        // 搬运地址
        view.burstLen = GetBurstLen<LoopInfo, burstLenAxis>(iterAddr_, tiling_);
        view.axisSize = 0;
        if constexpr (burstLenAxis > 0) {
            int32_t axis = burstLenAxis;
            for (int32_t i = 0; i < Dim; i++) {
                view.axisSize = i + 1;
                view.axis[i].repeat =
                    GetRepeatStride<LoopInfo>(axis - 1, iterAddr_, tiling_, view.axis[i].srcStride);
                view.axis[i].idx = GetRepeatStrideAxis<LoopInfo>(axis - 1, iterAddr_);
                view.axis[i].isAxisA = IsAxisA<Pattern::FirstA>(view.axis[i].idx);
                if (view.axis[i].idx <= 0) {
                    break;
                }
                axis = view.axis[i].idx;
            }
        }
    }

    template <class V, class U>
    __aicore__ inline void CalcInnerShape(V& view, U& shape)
    {
        shape.oriBurstLen = view.burstLen;
        if constexpr (!InnerPattern::TailA) {
            int64_t value = OpsUtils::CeilAlign(view.burstLen, BLOCK_SIZE_BYTE / sizeof(InDtype));
            if (IsLoopSpliteRAxis<LoopInfo>(Dim - 1)) {
                value = OpsUtils::CeilAlign(this->ubFactorR_, BLOCK_SIZE_BYTE / sizeof(InDtype));
            }
            for (uint64_t i = 0; i < view.axisSize; i++) {
                if (!view.axis[i].isAxisA) {
                    view.axis[i].dstStride = value;
                    if (IsLoopSpliteRAxis<LoopInfo>(view.axis[i].idx)) {
                        value = value * this->ubFactorR_;
                    } else {
                        value = value * view.axis[i].repeat;
                    }
                }
            }
            shape.value[InnerPattern::Dim - 1] = value;
            for (uint64_t i = 0; i < view.axisSize; i++) {
                if (view.axis[i].isAxisA) {
                    view.axis[i].dstStride = value;
                    value = value * view.axis[i].repeat;
                }
            }
            shape.value[InnerPattern::Dim - 2] = value / shape.value[InnerPattern::Dim - 1];
        } else {
            int64_t value = OpsUtils::CeilAlign(view.burstLen, BLOCK_SIZE_BYTE / sizeof(InDtype));
            aOutBurstLen_ = view.burstLen;
            if (IsLoopSpliteRAxis<LoopInfo>(Dim - 1)) {
                value = OpsUtils::CeilAlign(this->ubFactorA_, BLOCK_SIZE_BYTE / sizeof(InDtype));
                aOutBurstLen_ = this->ubFactorA_;
            }
            aOutNBurst_ = 1;
            for (uint64_t i = 0; i < view.axisSize; i++) {
                if (view.axis[i].isAxisA) {
                    view.axis[i].dstStride = value;
                    value = value * view.axis[i].repeat;
                    aOutNBurst_ = aOutNBurst_ * view.axis[i].repeat;
                }
            }
            shape.value[InnerPattern::Dim - 1] = value;
            for (uint64_t i = 0; i < view.axisSize; i++) {
                if (!view.axis[i].isAxisA) {
                    view.axis[i].dstStride = value;
                    value = value * view.axis[i].repeat;
                }
            }
            shape.value[InnerPattern::Dim - 2] = value / shape.value[InnerPattern::Dim - 1];
        }
    }

    template <bool isPadding, class U, class V>
    __aicore__ inline void CopyIn(U& view, V& shape, const AscendC::LocalTensor<InDtype>& ubTensor)
    {
        // 计算在UB的 View
        op_->ReduceOp::template CopyInAux<isPadding>(this->input_[0], view, shape, ubTensor);
    }

    template <class V, class... Args>
    __aicore__ inline void Compute(V& shape, const AscendC::LocalTensor<PromoteDataType>& ubTensor, Args... args)
    {
        op_->ReduceOp::template ComputeAux<(!InnerPattern::TailA && Pattern::Dim <= 2), InnerPattern>(
            shape, ubTensor, args...);
    }

    template <class V, class... Args>
    __aicore__ inline void ComputeMerge(V& shape, const AscendC::LocalTensor<PromoteDataType>& ubTensorLeft,
                                        const AscendC::LocalTensor<PromoteDataType>& ubTensorRight, Args... args)
    {
        // Ub间Reduce
        op_->compute_.ReduceBetweenUB(ubTensorLeft, ubTensorRight, shape.value[0] * shape.value[1]);
        op_->ReduceOp::template FreeTensorAux(ubTensorLeft);
        op_->ReduceOp::template ComputeAux<false, InnerPattern>(shape, ubTensorRight, args...);
        op_->ReduceOp::template FreeTensorAux(ubTensorRight);
    }

    template <class V>
    __aicore__ inline void CopyOut(int64_t tmpBufOffest, V& shape)
    {
        constexpr int32_t axis = Pattern::FirstA ? 0 : 1;
        uint64_t addrOffset = 0;
        for (int32_t i = axis; i < Dim; i += CONST2) {
            addrOffset += this->iterAddr_[i].start * tiling_->dstStride[i];
        }
        SliceView<CONST2> view;
        view.addr = addrOffset;
        if constexpr (Pattern::TailA) {
            view.burstLen = aOutBurstLen_;
            view.axis[0].repeat = aOutNBurst_;
        } else {
            view.burstLen = shape.value[0];
            view.axis[0].repeat = 1;
        }
        op_->ReduceOp::template CopyOutAux(this->output_[0], view, tmpBufOffest);
    }

    template <class V>
    __aicore__ inline void CopyOutGroup(int64_t tmpBufOffest, V& shape)
    {
        // CopyOut As RA Pattern
        SliceView<CONST2> view;
        int32_t blockId = AscendC::GetBlockIdx();

        int32_t innerA = GetInnerA<LoopInfo, Pattern::TailA, Pattern::Dim>(iterAddr_);
        if constexpr (Pattern::TailA) {
            view.burstLen = aOutBurstLen_;
            view.axis[0].repeat = aOutNBurst_;
            view.axis[0].srcStride = OpsUtils::CeilAlign(aOutBurstLen_, BLOCK_SIZE_BYTE / sizeof(InDtype));
        } else {
            view.burstLen = shape.value[0];
            view.axis[0].repeat = 1;
            view.axis[0].srcStride = shape.value[0];
        }
        int32_t axis = Pattern::FirstA ? 0 : 1;
        if constexpr (LoopInfo->loopACount > 0) {
            axis = LoopInfo->loopAAxis[LoopInfo->loopACount - 1];
        }

        uint64_t addrOffset = 0;
        if constexpr (LoopInfo->loopInnerACount > 0) {
            for (int32_t i = axis; i < Dim; i += CONST2) {
                addrOffset += this->iterAddr_[i].start * tiling_->dstStride[i];
            }
        }

        uint64_t axisStep = LoopInfo->loopACount > 0 ? this->loopAAxisStep_ : 1;
        view.addr = (blockId % tiling_->groupR) * tiling_->outSize +                            // group offset
                    (blockId / (tiling_->groupR * axisStep)) * tiling_->shape[axis] * innerA +  // AAxis offset
                    (blockId / tiling_->groupR % axisStep) * this->ubFactorA_ * innerA +        // main offset
                    addrOffset;                                                                 // innerA offset

        op_->ReduceOp::template CopyOutAuxGroup(this->output_[0], view, tmpBufOffest);
    }
};
}  // namespace Reduce
}  // namespace KernelUtils
}  // namespace ATVC
#endif // ATVC_REDUCE_UTILS_BLOCK_AUX_H
