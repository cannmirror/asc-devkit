
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

#ifndef ATVC_BROADCAST_OP_TEMPLATE_H
#define ATVC_BROADCAST_OP_TEMPLATE_H
#include "common/const_def.h"
#include "broadcast/broadcast_utils/broadcast_buf_pool.h"
namespace ATVC {
struct BroadcastDataView {
    uint32_t dimASize;
    uint32_t dimBSize;
    uint32_t inShape[ATVC::MAX_DIM];
    uint32_t outShape[ATVC::MAX_DIM];
    uint32_t copyInSize;        // 单核拷入数据量
    uint32_t A11;               // 实际参与计算的A11
    uint32_t A12;               // 实际参与计算的A12
    uint32_t B1;                // 实际参与计算的B1
    uint32_t dimAOffset;        // 输入输出数据在A维度的偏移量
    uint32_t dimBOffset;        // 输入输出数据在B维度的偏移量
    uint32_t copyOutBaseOffset; // 核间数据拷出基址
};

namespace Kernel {
template <class BroadcastCompute, const auto& SelectBroadcastPolicy>
class BroadcastOpTemplate {
public:
    using DataType = typename BroadcastCompute::DataType;
    __aicore__ inline BroadcastOpTemplate() {}
    /*
    BroadcastOpTemplate对外运行接口，主要完成资源初始化、数据搬入、计算调度、数据搬出操作
    @param src: 输入数据的gm指针
    @param dst: 输出数据的gm指针
    @broadcastParam: broadcast的动态参数，包含tiling data, workspace等
    */
    __aicore__ inline void Run(GM_ADDR src, GM_ADDR dst, GM_ADDR broadcastParam)
    {
        this->Init(src, dst, broadcastParam);
        this->Process();
    }

private:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR broadcastParam)
    {
        param_ = reinterpret_cast<__gm__ BroadcastParam*>(broadcastParam);
        tilingData_ = &param_->tilingData;
        uint32_t srcDataSize = tilingData_->basicBlock;
        uint32_t dstDataSize = tilingData_->basicBlock;
        srcGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(src), srcDataSize);
        dstGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType*>(dst), dstDataSize);
        bufPool_.template Init<DataType>(GetTPipePtr(),
                                    ATVC::CONST2, // doublebuff需要的输入个数
                                    ATVC::CONST2, // 计算结果的个数，一般与inputNum保持一致
                                    tilingData_->A2 * tilingData_->A12 * DATA_SIZE, // 输入Tensor大小
                                    tilingData_->A2 * tilingData_->B2 * DATA_SIZE); // 输出Tensor大小
    }

    __aicore__ inline void CopyOutBatch(BroadcastDataView &view,
                                        uint32_t dimACount,
                                        AscendC::LocalTensor<DataType> &output)
    {
        uint32_t dimBCount = 0;          
        SyncDataQueue<AscendC::HardEvent::V_MTE3>(); 
        for (int i = 0; i < view.B1; i++) {
            uint32_t copyOutOffset;
            if (SelectBroadcastPolicy.patternID == AB_PATTERN::ABA) {
                copyOutOffset = dimBCount * view.dimASize + dimACount * tilingData_->A2;
            } else {
                copyOutOffset = dimACount * tilingData_->A2 * view.dimBSize + dimBCount;
            }
            CopyOut(output, copyOutOffset + view.copyOutBaseOffset, view);
            dimBCount += tilingData_->B2;
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

    __aicore__ inline void Process()
    {
        BroadcastDataView view;
        CalcView(view);
        uint32_t inputOffset;
        uint32_t dimACount = 0;
        AscendC::LocalTensor<DataType> input;
        for (int i = 0; i < view.A11; i++) {
            inputOffset = 0;
            bufPool_.AllocTensor<true>(input);
            uint32_t copyInOffset = i * view.A12 * tilingData_->A2;
            if (tilingData_->A0 != 1) {
                copyInOffset += view.dimAOffset;
            }
            if (copyInOffset >= view.dimASize) {
                return;
            }
            if (copyInOffset + view.copyInSize > view.dimASize) {
                // 剩下的数据不够一次完整计算， 根据实际数据重新计算
                view.copyInSize = view.dimASize - copyInOffset;
                view.A12 = OpsUtils::CeilDiv<uint32_t>(view.copyInSize, tilingData_->A2);
            }
            CopyIn(input, copyInOffset, view);
            bufPool_.SetVecSync(input);
            bufPool_.WaitVecSync(input);
            for (int j = 0; j < view.A12; j ++) {
                AscendC::LocalTensor<DataType> output;
                bufPool_.AllocTensor<false>(output);
                SyncDataQueue<AscendC::HardEvent::MTE2_V>();
                compute_.template Compute<SelectBroadcastPolicy.patternID>(input, inputOffset, output,
                    OpsUtils::CeilAlign<uint32_t>(tilingData_->A2, UB_ALIGN_COUNT),
                    OpsUtils::CeilAlign<uint32_t>(tilingData_->B2, UB_ALIGN_COUNT));
                bufPool_.SetCopyOutSync(output);
                bufPool_.WaitCopyOutSync(output);
                CopyOutBatch(view, dimACount, output);
                bufPool_.FreeTensor<false>(output);
                dimACount++;
                inputOffset += tilingData_->A2;
                SyncDataQueue<AscendC::HardEvent::MTE3_V>();
            }
            SyncDataQueue<AscendC::HardEvent::MTE3_MTE2>();
            bufPool_.FreeTensor<true>(input);            
        }
        bufPool_.ResetEvent();
    }

    __aicore__ inline uint32_t CalcCopyOutBaseOffset(BroadcastDataView &view)
    {
        uint32_t copyOutBaseOffset = 0;
        // 计算拷出偏移基址
        if (SelectBroadcastPolicy.patternID == AB_PATTERN::ABA) {
            if (tilingData_->A0 != 1) { // 核间A切分， 取部分A
                copyOutBaseOffset += view.dimAOffset;
            }
            if (tilingData_->B0 != 1) { // 核间B切分，取部分B
                copyOutBaseOffset += view.dimBOffset * view.dimASize;
            }
        } else {
            if (tilingData_->A0 != 1) { // 核间A切分， 取部分A
                copyOutBaseOffset += view.dimAOffset * view.dimBSize;
            }
            if (tilingData_->B0 != 1) { // 核间B切分，取部分B
                copyOutBaseOffset += view.dimBOffset;
            }
        }    
        return copyOutBaseOffset;    
    }

    __aicore__ inline void CalcView(BroadcastDataView &view)
    {
        if (SelectBroadcastPolicy.patternID == AB_PATTERN::ABA) {
            view.dimASize = tilingData_->dstShape[1];
            view.dimBSize = tilingData_->dstShape[0];
            view.inShape[0] = 1;
            view.inShape[1] = tilingData_->A2;
            view.outShape[0] = tilingData_->B2;
            view.outShape[1] = tilingData_->A2;
        } else {
            view.dimASize = tilingData_->dstShape[0];
            view.dimBSize = tilingData_->dstShape[1];
            view.inShape[0] = tilingData_->A2;
            view.inShape[1] = 1;
            view.outShape[0] = tilingData_->A2;
            view.outShape[1] = tilingData_->B2;
        }        
        view.A11 = tilingData_->A11;
        view.A12 = tilingData_->A12;
        view.B1 = tilingData_->B1;
        uint32_t blockId = AscendC::GetBlockIdx();
        uint32_t dimAIdx = blockId / tilingData_->B0;
        uint32_t dimBIdx = blockId % tilingData_->factorBTotalCnt;
        view.dimAOffset = dimAIdx * tilingData_->factorACntPerCore;
        view.dimBOffset = dimBIdx * tilingData_->factorBCntPerCore;
        // 计算一次计算的输入数据大小
        view.copyInSize = view.A12 * tilingData_->A2; // 一次拷贝A12份数据， for循环计算A12次
        if (view.dimAOffset + tilingData_->factorACntPerCore > view.dimASize) {
            // 剩下的A维度的数据不够每个核分到的A数目，重新计算实际的A维度切分
            uint32_t realShape = view.dimASize - view.dimAOffset;
            uint32_t A1 =  OpsUtils::CeilDiv<uint32_t>(realShape, tilingData_->A2);
            if (A1 < view.A12) {
                view.A11 = 1;
                view.A12 = A1;
            } else {
                view.A11 = OpsUtils::CeilDiv<uint32_t>(A1, view.A12);
            }
        }
        if (view.dimBOffset + tilingData_->factorBCntPerCore > view.dimBSize) {
            uint32_t realShape = view.dimBSize - view.dimBOffset;
            view.B1 =  OpsUtils::CeilDiv<uint32_t>(realShape, tilingData_->B2);
        }
        view.copyOutBaseOffset = CalcCopyOutBaseOffset(view);
    }

    __aicore__ inline void CopyIn(AscendC::LocalTensor<DataType> &input, uint32_t copyInOffset, BroadcastDataView &view)
    {
        AscendC::DataCopyPadExtParams<DataType> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = view.copyInSize * DATA_SIZE;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        AscendC::DataCopyPad(input, srcGlobal_[copyInOffset], copyInParams, padParams);
    }

    __aicore__ inline void CopyOutNonAligned(AscendC::LocalTensor<DataType> &output,
                                             uint32_t copyOutOffset, BroadcastDataView &view)
    {
        uint32_t blockId = AscendC::GetBlockIdx();
        uint32_t dstDataSize = view.outShape[0] * view.outShape[1];
        uint64_t dstShape = tilingData_->dstShape[1];
        AscendC::DataCopyExtParams copyOutParams;
        copyOutParams.blockLen = view.outShape[1] * DATA_SIZE;
        copyOutParams.blockCount = dstDataSize * DATA_SIZE / copyOutParams.blockLen;
        copyOutParams.srcStride = 0;
        if (view.outShape[1] + copyOutOffset % dstShape > dstShape) {
            // 列非对齐， 按实际数据拷贝
            copyOutParams.srcStride = OpsUtils::CeilAlign<uint32_t>(view.outShape[1], UB_ALIGN_COUNT) * DATA_SIZE;
            copyOutParams.blockLen = (dstShape - copyOutOffset % dstShape) * DATA_SIZE;
            copyOutParams.srcStride = (copyOutParams.srcStride - copyOutParams.blockLen) / ATVC::UB_ALIGN_32;
        }
        if (view.outShape[0] + copyOutOffset / dstShape > tilingData_->dstShape[0]) {
            // 行非对齐， 按实际数据拷贝
            copyOutParams.blockCount = (tilingData_->dstShape[0] - copyOutOffset / dstShape);
        }
        copyOutParams.dstStride = dstShape * DATA_SIZE - copyOutParams.blockLen;
        AscendC::DataCopyPad(dstGlobal_[copyOutOffset], output, copyOutParams);
    }

    __aicore__ inline void CopyOut(AscendC::LocalTensor<DataType> &output, uint32_t copyOutOffset, BroadcastDataView &view)
    {
        CopyOutNonAligned(output, copyOutOffset, view);
    }

    AscendC::GlobalTensor<DataType> srcGlobal_;
    AscendC::GlobalTensor<DataType> dstGlobal_;
    BroadcastCompute compute_;
    const __gm__ BroadcastParam *param_;
    const __gm__ BroadcastOpTilingData *tilingData_;
    KernelUtils::BroadcastBufPool bufPool_;
    constexpr static uint32_t DATA_SIZE = sizeof(DataType);
    constexpr static uint32_t UB_ALIGN_COUNT = ATVC::UB_ALIGN_32 / DATA_SIZE;
};
} // namespace Kernel
} // namespace ATVC
#endif // ATVC_BROADCAST_OP_TEMPLATE_H