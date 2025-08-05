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
 * \file reduce_op_template.h
 * \brief
 */
#ifndef ATVC_REDUCE_OP_TEMPLATE_H
#define ATVC_REDUCE_OP_TEMPLATE_H

#include <type_traits>
#include "common/const_def.h"
#include "kernel_operator.h"
#include "reduce/common/patterns.h"
#include "reduce/reduce_utils/reduce_block_aux_util.h"
#include "reduce/reduce_utils/reduce_block_aux.h"
#include "reduce/reduce_utils/reduce_util.h"
#include "reduce/reduce_utils/reduce_buf_pool.h"
#include "reduce/common/reduce_common.h"

namespace ATVC {
namespace Kernel {
template <class ReduceCompute, const auto& SelectReducePolicy,
            class PreCompute = void, class PostCompute = void>
class ReduceOpTemplate {
public:
    constexpr static ReduceSchLoopInfo SchLoopInfo = KernelUtils::Reduce::GetSchLoopInfo<SelectReducePolicy>();
    using Pattern = typename ReducePattern::GetPattern<SchLoopInfo.patternID>::T;
    using DataType = typename ReduceCompute::DataType;
    using PromoteDataType = typename ReduceCompute::PrompteDtype;
    constexpr static int32_t ELEMENT_ONE_REPEAT_ORI = Platform::GetVRegSize() / sizeof(DataType);
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE = Platform::GetVRegSize() / sizeof(PromoteDataType);
    constexpr static int32_t VL_LENGTH_B = Platform::GetVRegSize();
    constexpr static uint64_t BLOCK_SIZE_BYTE = Platform::GetUbBlockSize();
    constexpr static uint64_t CACHE_BUF_SIZE = 16 * 1024;
    constexpr static uint64_t RES_BUF_SIZE = 16 * 1024;
    constexpr static uint32_t TBufSize = KernelUtils::GetCopyInCount<DataType>();
    constexpr static uint32_t PromoteBufSize = KernelUtils::GetComputeCount<DataType>();
    AscendC::LocalTensor<PromoteDataType> tempBuf_;
    AscendC::LocalTensor<PromoteDataType> computeRes_;
    // 算子开发者传入的计算对象
    ReduceCompute compute_;

public:
    __aicore__ inline ReduceOpTemplate() {}
    
    // 按照输入、输出、ReduceParam、其他标量的顺序传入
    // 内部根据ReduceParam进行数据调度并调用ReduceOpTemplate完成计算后搬出到GM
    template<typename ...Args>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR param) {
        param_ = reinterpret_cast<__gm__ ReduceParam*>(param);
        // 完成一些编译期的检查，比如PreCompute和PostCompute的In、Out个数是否与args的个数匹配
        tiling_ = &param_->tilingData;

        Init((GM_ADDR)(param_->workspaceAddr), x, y);
        Process();
    }

public:
    template <class... Args>
    __aicore__ inline void Init(GM_ADDR workspace, Args... args)
    {
        pipe_ = GetTPipePtr();
        basicBlockLen_ = tiling_->basicBlock;
        bufPool_.template Init<DataType, PromoteDataType>(pipe_, TBufSize, PromoteBufSize, tiling_->basicBlock);

        InitArgsInput<0>(args...);
        InitArgsWorkspace(workspace);
        pipe_->InitBuffer(tempResQue_, RES_BUF_SIZE);
        computeRes_ = tempResQue_.Get<PromoteDataType>();
        pipe_->InitBuffer(tempBufQue_, CACHE_BUF_SIZE);

        tempBuf_ = tempBufQue_.template Get<PromoteDataType>();

        pipe_->InitBuffer(tempUbQue_, BLOCK_SIZE_BYTE);
        tempUb_ = tempUbQue_.template Get<PromoteDataType>();
    }

    template <bool IsInput, bool needDup=true, typename T>
    __aicore__ inline void AllocTensorAux(AscendC::LocalTensor<T>& tensor)
    {
        bufPool_.AllocTensor<IsInput, needDup>(tensor);
    }

    template <typename T>
    __aicore__ inline void FreeTensorAux(const AscendC::LocalTensor<T>& tensor)
    {
        bufPool_.FreeTensor(tensor);
    }

    template <class... Args>
    __aicore__ inline void Process(Args... args)
    {
        if constexpr (SelectReducePolicy.patternID == ATVC::AR_PATTERN::A) {
            CopyInput2Output();
            return;
        }
        if constexpr (SchLoopInfo.loopRCount == 0) {
            using SchTypeA = KernelUtils::Reduce::ReduceBlockAux<
                __gm__ ATVC::ReduceTilingData, &SchLoopInfo,
                std::remove_reference_t<decltype(*this)>, DataType, DataType,
                PreCompute, PostCompute>;

            SchTypeA op(this, input_, output_, tiling_);
            op.Process(args...);
        } else {
            // 完成第一阶段的Reduce
            using SchTypeR = KernelUtils::Reduce::ReduceBlockAux<
                __gm__ ATVC::ReduceTilingData, &SchLoopInfo,
                std::remove_reference_t<decltype(*this)>, DataType,
                PromoteDataType, PreCompute, void>;
            SchTypeR op(this, input_, &workspace_, tiling_);
            op.Process(args...);
            bufPool_.ResetEvent();

            // 全核同步
            AscendC::SyncAll();

            // 完成第二阶段的Reduce
            bufPool_.template ResetInputSize<PromoteDataType>(3);
            constexpr static ReduceSchLoopInfo groupSchLoopInfo = KernelUtils::Reduce::GetGroupSchLoopInfo();
            ATVC::ReduceTilingData groupTiling;
            SetGroupTiling(groupTiling);
            using SchTypeA = KernelUtils::Reduce::ReduceBlockAux<
                ATVC::ReduceTilingData, &groupSchLoopInfo,
                std::remove_reference_t<decltype(*this)>, PromoteDataType,
                DataType, void, PostCompute>;
            SchTypeA groupOp(this, &workspace_, output_, &groupTiling);
            groupOp.Process(args...);
        }
    }

    __aicore__ inline void SetGroupTiling(ATVC::ReduceTilingData& groupTiling)
    {
        groupTiling.ubFactorA = ELEMENT_ONE_REPEAT_COMPUTE;
        groupTiling.ubFactorR = tiling_->groupR;
        groupTiling.shape[0] = tiling_->groupR;
        groupTiling.shape[1] = tiling_->outSize;
        groupTiling.stride[0] = tiling_->outSize;
        groupTiling.stride[1] = 1;
        groupTiling.dstStride[0] = tiling_->outSize;
        groupTiling.dstStride[1] = 1;
        groupTiling.groupR = 1;
        groupTiling.outSize = tiling_->outSize;
        groupTiling.factorRCntPerCore = 1;
        groupTiling.factorRTotalCnt = 1;
        groupTiling.factorATotalCnt = OpsUtils::CeilDiv(groupTiling.shape[1], groupTiling.ubFactorA);
        groupTiling.factorACntPerCore = OpsUtils::CeilDiv(groupTiling.factorATotalCnt,
                                                          static_cast<uint64_t>(64));  // 按照64核计算，需要tiling传
    }

    template <bool isPadding, class T, class U, class V>
    __aicore__ inline void CopyInAux(const AscendC::GlobalTensor<T> &src,
                                     U &view, V &shape,
                                     const AscendC::LocalTensor<T> &ubTensor)
    {
        T paddingValue = compute_.template GetPaddingValue<T>();
        uint8_t padCnt = ((view.axis[0].dstStride - view.burstLen) * sizeof(T)) % BLOCK_SIZE_BYTE / sizeof(T);
        int32_t dupliCnt = view.axis[0].dstStride * view.axis[0].repeat;
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, padCnt, paddingValue};
        if constexpr (!isPadding) {
            padParams = {false, 0, 0, 0};
            shape.oriBurstLen = view.burstLen;
        }
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = view.axis[0].repeat;
        copyInParams.blockLen = view.burstLen * sizeof(T);                              // unit Byte
        copyInParams.srcStride = (view.axis[0].srcStride - view.burstLen) * sizeof(T);  // unit Byte
        copyInParams.dstStride =
            (view.axis[0].dstStride - view.burstLen) * sizeof(T) / BLOCK_SIZE_BYTE;  // unit block(32byte)
        bufPool_.SyncTensor(ubTensor);

        const int32_t repeats[CONST6] = {static_cast<int32_t>(view.axis[CONST1].repeat),
            static_cast<int32_t>(view.axis[CONST2].repeat), static_cast<int32_t>(view.axis[CONST3].repeat),
            static_cast<int32_t>(view.axis[CONST4].repeat), static_cast<int32_t>(view.axis[CONST5].repeat),
            static_cast<int32_t>(view.axis[CONST6].repeat)};
        const int32_t dstStrides[CONST6] = {static_cast<int32_t>(view.axis[CONST1].dstStride),
            static_cast<int32_t>(view.axis[CONST2].dstStride), static_cast<int32_t>(view.axis[CONST3].dstStride),
            static_cast<int32_t>(view.axis[CONST4].dstStride), static_cast<int32_t>(view.axis[CONST5].dstStride),
            static_cast<int32_t>(view.axis[CONST6].dstStride)};
        const int32_t srcStrides[CONST6] = {static_cast<int32_t>(view.axis[CONST1].srcStride),
            static_cast<int32_t>(view.axis[CONST2].srcStride), static_cast<int32_t>(view.axis[CONST3].srcStride),
            static_cast<int32_t>(view.axis[CONST4].srcStride), static_cast<int32_t>(view.axis[CONST5].srcStride),
            static_cast<int32_t>(view.axis[CONST6].srcStride)};

        int32_t total = 1;
        for (int32_t i = 0; i < CONST6; ++i)
            total *= repeats[i];

        for (int32_t idx = 0; idx < total; ++idx) {
            int32_t tmp = idx;
            int32_t dstOffset = 0;
            int32_t srcOffset = 0;
            for (int32_t axis = 0; axis < CONST6; ++axis) {
                int32_t coord = tmp % repeats[axis];
                tmp /= repeats[axis];
                dstOffset += coord * dstStrides[axis];
                srcOffset += coord * srcStrides[axis];
            }
            AscendC::DataCopyPad(ubTensor[dstOffset], src[view.addr + srcOffset], copyInParams, padParams);
        }
    }

    __aicore__ inline void CopyInput2Output()
    {
        uint32_t shapeSize = 1;
        for (uint8_t i = 0; i < MAX_DIM; i++) {
            if (tiling_->shape[i] <= 1) {
                break;
            }
            shapeSize = shapeSize * tiling_->shape[i];
        }
        shapeSize = (shapeSize * sizeof(DataType) + UB_ALIGN_31) / UB_ALIGN_32 * UB_ALIGN_32 / sizeof(DataType);
        AscendC::LocalTensor<DataType> tmpLoc;
        AllocTensorAux<true, false>(tmpLoc);
        uint32_t locSize = tmpLoc.GetSize();
        uint32_t loopCnt = shapeSize / locSize *locSize;
        uint32_t tailCnt = shapeSize - loopCnt;
        for (uint32_t i = 0; i < loopCnt; i += locSize) {
            AscendC::DataCopy(tmpLoc, input_[0][i], locSize);
            event_t copyId = static_cast<event_t>(GetTPipePtr()->FetchEventID<AscendC::HardEvent::MTE2_MTE3>());
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyId);
            AscendC::DataCopy(output_[0][i], tmpLoc, locSize);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(copyId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(copyId);
        }
        if (tailCnt > 0) {
            AscendC::DataCopy(tmpLoc, input_[0][loopCnt], tailCnt);
            event_t copyId = static_cast<event_t>(GetTPipePtr()->FetchEventID<AscendC::HardEvent::MTE2_MTE3>());
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(copyId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(copyId);
            AscendC::DataCopy(output_[0][loopCnt], tmpLoc, tailCnt);
        }
    }

    template <bool needMask, class Pattern, class V, class T, class... Args>
    __aicore__ inline void ComputeAux(V& view, const AscendC::LocalTensor<T>& ubTensor, Args... args)
    {
        compute_.template Compute<needMask, Pattern>(view, computeRes_, ubTensor, args...);
    }

    template <class T, class U>
    __aicore__ inline void CopyOutAux(const AscendC::GlobalTensor<T>& dst, U& view, int64_t tmpBufOffest)
    {
        AscendC::DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
        copyOutParams.blockCount = view.axis[0].repeat;
        copyOutParams.blockLen = view.burstLen * sizeof(T);

        if constexpr (AscendC::IsSameType<PromoteDataType, DataType>::value) {
            SetEvent<AscendC::HardEvent::V_MTE3>(AscendC::HardEvent::V_MTE3);
            AscendC::DataCopyPad(dst[view.addr], tempBuf_[tmpBufOffest], copyOutParams);
        } else {
            AscendC::LocalTensor<DataType> outputLocal = tempBuf_[tmpBufOffest].template ReinterpretCast<DataType>();
            AscendC::Cast(outputLocal, tempBuf_[tmpBufOffest], AscendC::RoundMode::CAST_RINT,
                    view.axis[0].repeat *
                        OpsUtils::CeilAlign(view.burstLen, static_cast<uint64_t>(ELEMENT_ONE_REPEAT_COMPUTE)));
            SetEvent<AscendC::HardEvent::V_MTE3>(AscendC::HardEvent::V_MTE3);
            AscendC::DataCopyPad(dst[view.addr], outputLocal, copyOutParams);
        }
    }

    template <class T, class U>
    __aicore__ inline void CopyOutAuxGroup(const AscendC::GlobalTensor<T>& dst, U& view, int64_t tmpBufOffest)
    {
        SetEvent<AscendC::HardEvent::V_MTE3>(AscendC::HardEvent::V_MTE3);
        AscendC::DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
        copyOutParams.blockCount = view.axis[0].repeat;
        copyOutParams.blockLen = view.burstLen * sizeof(T);
        copyOutParams.srcStride = (view.axis[0].srcStride - view.burstLen) * sizeof(T) / BLOCK_SIZE_BYTE;
        AscendC::DataCopyPad(dst[view.addr], tempBuf_[tmpBufOffest], copyOutParams);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

protected:
    template <class... Args>
    __aicore__ inline void InitArgsWorkspace(GM_ADDR workspace, Args... args)
    {
        workspace_.SetGlobalBuffer((__gm__ PromoteDataType*)workspace);
    }

    template <int32_t start, class... Args>
    __aicore__ inline void InitArgsOutput(GM_ADDR y, Args... args)
    {
        output_[start].SetGlobalBuffer((__gm__ DataType*)y);
        if constexpr (start + 1 < OutputSize) {
            InitArgsOutput<start + 1>(args...);
        }
    }

    template <int32_t start, class... Args>
    __aicore__ inline void InitArgsInput(GM_ADDR x, Args... args)
    {
        input_[start].SetGlobalBuffer((__gm__ DataType*)x);
        if constexpr (start + 1 < InputSize) {
            InitArgsInput<start + 1>(args...);
        } else {
            InitArgsOutput<0>(args...);
        }
    }

private:
    __gm__ ReduceParam* param_;   // CalcReduceTiling API计算出的运行态参数
    AscendC::TPipe* pipe_;
    AscendC::TBuf<> oriVecQue_;
    AscendC::TBuf<> tempResQue_;
    AscendC::TBuf<> tempBufQue_;
    AscendC::TBuf<> tempUbQue_;

    constexpr static int32_t Dim = Pattern::Dim;
    constexpr static uint32_t InputSize = 1;
    constexpr static uint32_t OutputSize = 1;

    AscendC::GlobalTensor<DataType> output_[OutputSize];
    AscendC::GlobalTensor<DataType> input_[InputSize];
    AscendC::GlobalTensor<PromoteDataType> workspace_;

    AscendC::LocalTensor<PromoteDataType> tempUb_;
    const __gm__ ATVC::ReduceTilingData* tiling_;

    int64_t basicBlockLen_;
    int64_t oriBasicBlockLen_;

    KernelUtils::ReduceBufPool bufPool_;
};
}
}
#endif // ATVC_REDUCE_OP_TEMPLATE_H
