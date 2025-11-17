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
 * \file broadcast_buf_pool.h
 * \brief class BroadcastBufPool
 */

#ifndef ATVC_BROADCAST_BUF_POOL_H
#define ATVC_BROADCAST_BUF_POOL_H

#include "kernel_operator.h"

namespace ATVC {
namespace KernelUtils {
struct BrcPoolManagerUnit {
    int32_t idx = -1;
    int32_t eleSize = 0;
    int32_t bufNum = 0;
    int32_t offset = 0;
};

class BroadcastBufPool {
constexpr static int32_t MAX_INPUT_SIZE = 10;

public:
    __aicore__ inline BroadcastBufPool(){};

    template <class DataType>
    __aicore__ inline void Init(AscendC::TPipe* pipeIn,
                                int32_t inputNum,           // doublebuff需要的输入个数
                                int32_t computeNum,         // 计算结果的个数，一般与inputNum保持一致
                                int32_t inBlockLen,         // 一次计算的输入基本块大小
                                int32_t outBlockLen) {      // 一次计算的输出大小
        /*
         _______________________________________________________________________________________________________
        |   inputTensor 0   |   inputTensor 1   |        outputTensor 0         |        outputTensor 0         |
        |___________________|___________________|_______________________________|_______________________________|
        */
        pipe_ = pipeIn;
        int32_t eleSize = sizeof(DataType);
        inputNum_ = inBlockLen / eleSize;
        outputNum_ = outBlockLen / eleSize;
        int32_t poolSize = inBlockLen * inputNum + outBlockLen * computeNum;
        inputUnit_.bufNum = inputNum;
        inputUnit_.eleSize = eleSize;
        inputUnit_.offset = 0;
        computeUnit_.bufNum = computeNum;
        computeUnit_.eleSize = eleSize;
        computeUnit_.offset = inBlockLen * inputNum;
        // Init buffer
        pipe_->InitBuffer(qQue_, poolSize);
    }

    template <bool IsInput, typename T>
    __aicore__ inline const void AllocTensor(AscendC::LocalTensor<T>& tensor) {
        if constexpr (IsInput) {
            int32_t idx = GetInputTensorId();
            tensor = qQue_.GetWithOffset<T>(inputNum_, inputUnit_.offset + idx * inputNum_ * sizeof(T));
        } else {
            int32_t idx = GetComputeTensorId();
            tensor = qQue_.GetWithOffset<T>(outputNum_, computeUnit_.offset + idx * outputNum_ * sizeof(T));
        }
    }

    template <bool IsInput, typename T>
    __aicore__ inline const void FreeTensor(AscendC::LocalTensor<T>& tensor) {
        if constexpr (!IsInput) {
            uint32_t idx = GetOutputTensorIdx(tensor);
            isBusyOut_[idx] = false; // 恢复isBusy_状态
        }
    }

    template <typename T>
    __aicore__ inline const void SetVecSync(AscendC::LocalTensor<T>& tensor) {
        uint32_t idx = GetInputTensorIdx(tensor);
        event_t eventId = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        vecEventId_[idx] = eventId;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
    }

    template <typename T>
    __aicore__ inline const void WaitVecSync(AscendC::LocalTensor<T>& tensor) {
        uint32_t idx = GetInputTensorIdx(tensor);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vecEventId_[idx]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(vecEventId_[idx]);
    }

    template <typename T>
    __aicore__ inline const void SetCopyOutSync(AscendC::LocalTensor<T>& tensor) {
        uint32_t idx = GetOutputTensorIdx(tensor);
        event_t eventId = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>());
        outEventId_[idx] = eventId;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
    }

    template <typename T>
    __aicore__ inline const void WaitCopyOutSync(AscendC::LocalTensor<T>& tensor)
    {
        uint32_t idx = GetOutputTensorIdx(tensor);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outEventId_[idx]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(outEventId_[idx]);
    }

    template <typename T>
    __aicore__ inline uint32_t GetInputTensorIdx(AscendC::LocalTensor<T>& tensor)
    {
        uint64_t start = (uint64_t)qQue_.GetWithOffset<T>(inputNum_, 0).GetPhyAddr();
        uint64_t offset = (uint64_t)tensor.GetPhyAddr();
        uint32_t idx = (offset - start) / sizeof(T) / inputNum_;
        return idx;
    }

    template <typename T>
    __aicore__ inline uint32_t GetOutputTensorIdx(AscendC::LocalTensor<T>& tensor)
    {
        uint64_t start = (uint64_t)qQue_.GetWithOffset<T>(outputNum_, inputNum_).GetPhyAddr();
        uint64_t offset = (uint64_t)tensor.GetPhyAddr();
        uint32_t idx = (offset - start) / sizeof(T) / outputNum_;
        return idx;
    }

    __aicore__ inline const void ResetEvent()
    {
        pipe_->Reset();
    }

private:
    __aicore__ inline int32_t GetComputeTensorId()
    {
        uint32_t loopCnt = 0;
        do {
            computeUnit_.idx = (computeUnit_.idx + 1) % computeUnit_.bufNum;
            if (!isBusyOut_[computeUnit_.idx]) {
                break;
            }
            ++loopCnt;
        } while (loopCnt < ATVC::CONST10); // 10: 最多找10次，实际上 每个循环计算和拷贝间有流水同步，这里基本循环1次即可

        isBusyOut_[computeUnit_.idx] = true; // 标识isBusy_状态为busy
        return computeUnit_.idx;
    }

    __aicore__ inline int32_t GetInputTensorId()
    {
        inputUnit_.idx = (inputUnit_.idx + 1) % inputUnit_.bufNum;
        return inputUnit_.idx;
    }

    BrcPoolManagerUnit inputUnit_;
    BrcPoolManagerUnit computeUnit_;
    event_t vecEventId_[MAX_INPUT_SIZE];
    event_t outEventId_[MAX_INPUT_SIZE];
    bool isBusyOut_[MAX_INPUT_SIZE] = {false};
    AscendC::TBuf<> qQue_;
    AscendC::TPipe* pipe_;
    int32_t inputNum_;
    int32_t outputNum_;
};
} // namespace KernelUtils
} // namespace ATVC
#endif // ATVC_BROADCAST_BUF_POOL_H
