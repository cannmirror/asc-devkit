/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file set_atomic_add_multi_aic.h
 * \brief 多个AIC串行执行AtomicAdd操作
 */

#include "acl/acl.h"
#include "data_utils.h"
#include "kernel_operator.h"

class KernelSetAtomicAddMultiAIC {
public:
    __aicore__ inline KernelSetAtomicAddMultiAIC() {}

    __aicore__ inline void Init(
        __gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* src2Gm, __gm__ uint8_t* src3Gm,
        __gm__ uint8_t* src4Gm, __gm__ uint8_t* dstGm, AscendC::TPipe* pipeIn)
    {
        pipe = pipeIn;
        blockIdx = AscendC::GetBlockIdx();
        src0Global.SetGlobalBuffer((__gm__ float*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ float*)src1Gm);
        src2Global.SetGlobalBuffer((__gm__ float*)src2Gm);
        src3Global.SetGlobalBuffer((__gm__ float*)src3Gm);
        src4Global.SetGlobalBuffer((__gm__ int32_t*)src4Gm);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);

        pipe->InitBuffer(vecIn, 1, 8 * sizeof(int32_t));
        pipe->InitBuffer(inQueueSrc0, 1, 8 * sizeof(float));
        pipe->InitBuffer(inQueueSrc1, 1, 8 * sizeof(float));
        pipe->InitBuffer(inQueueSrc2, 1, 8 * sizeof(float));
        pipe->InitBuffer(inQueueSrc3, 1, 8 * sizeof(float));
        pipe->InitBuffer(inQueueSrc4, 1, 256 * sizeof(int32_t));
        pipe->InitBuffer(outQueueDst, 1, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // 预期的4个核执行的顺序是核0-->核2-->核3-->核1
        // 核0负责在原子累加之前，以src0Local初始化GM；核2、3、1负责按顺序做原子累加
        __gm__ int32_t* addr = const_cast<__gm__ int32_t*>(src4Global.GetPhyAddr());

        if (blockIdx == 0) {
            AscendC::LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
            AscendC::DataCopy(src0Local, src0Global, 8);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

            // 在原子累加之前，以src0Local初始化GM
            AscendC::DataCopy(dstGlobal, src0Local, 8);

            // 核0通知核2，核0的src0Local初始化已完成
            inQueueSrc0.FreeTensor(src0Local);

            // 在通知下一个核之前，等待当前核的任务完成
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WriteGmByPassDCache<int32_t>(addr + blockIdx * 32, 1);
        } else if (blockIdx == 1) {
            int32_t preblockIdx = 3;
            AscendC::LocalTensor<float> src1Local = inQueueSrc1.AllocTensor<float>();

            // 核1等待核3，直到核3数据已完成累加
            while (true) {
                int32_t value = AscendC::ReadGmByPassDCache<int32_t>(addr + preblockIdx * 32);
                if (value == 1) {
                    AscendC::WriteGmByPassDCache<int32_t>(addr + preblockIdx * 32, 0);
                    break;
                }
            }
            // 开始当前核的任务之前，等待上一个核的通知
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);

            AscendC::DataCopy(src1Local, src1Global, 8);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

            AscendC::DisableDmaAtomic();
            AscendC::SetAtomicAdd<float>();
            AscendC::DataCopy(dstGlobal, src1Local, 8);
            AscendC::DisableDmaAtomic();

            inQueueSrc1.FreeTensor(src1Local);
        } else if (blockIdx == 2) {
            int32_t preblockIdx = 0;
            AscendC::LocalTensor<float> src2Local = inQueueSrc2.AllocTensor<float>();

            // 核2等待核0，直到核0数据已完成累加
            while (true) {
                int32_t value = AscendC::ReadGmByPassDCache<int32_t>(addr + preblockIdx * 32);
                if (value == 1) {
                    AscendC::WriteGmByPassDCache<int32_t>(addr + preblockIdx * 32, 0);
                    break;
                }
            }
            // 开始当前核的任务之前，等待上一个核的通知
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);

            AscendC::DataCopy(src2Local, src2Global, 8);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

            AscendC::DisableDmaAtomic();
            AscendC::SetAtomicAdd<float>();
            AscendC::DataCopy(dstGlobal, src2Local, 8);
            AscendC::DisableDmaAtomic();

            inQueueSrc2.FreeTensor(src2Local);

            // 在通知下一个核之前，等待当前核的任务完成
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WriteGmByPassDCache<int32_t>(addr + blockIdx * 32, 1);
        } else if (blockIdx == 3) {
            int32_t preblockIdx = 2;
            AscendC::LocalTensor<float> src3Local = inQueueSrc3.AllocTensor<float>();

            // 核3等待核2，直到核2数据已完成累加
            while (true) {
                int32_t value = AscendC::ReadGmByPassDCache<int32_t>(addr + preblockIdx * 32);
                if (value == 1) {
                    AscendC::WriteGmByPassDCache<int32_t>(addr + preblockIdx * 32, 0);
                    break;
                }
            }

            // 开始当前核的任务之前，等待上一个核的通知
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);

            AscendC::DataCopy(src3Local, src3Global, 8);
            // AscendC::DumpTensor(src3Local, 0, 8);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

            AscendC::DisableDmaAtomic();
            AscendC::SetAtomicAdd<float>();
            AscendC::DataCopy(dstGlobal, src3Local, 8);
            AscendC::DisableDmaAtomic();

            // 在通知下一个核之前，等待当前核的任务完成
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
            AscendC::WriteGmByPassDCache<int32_t>(addr + blockIdx * 32, 1);

            inQueueSrc3.FreeTensor(src3Local);
        }
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueSrc0;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueSrc2;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueSrc3;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueSrc4;
    AscendC::TQue<AscendC::TPosition::A1, 1> vecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<float> src0Global, src1Global, src2Global, src3Global, dstGlobal;
    AscendC::GlobalTensor<int32_t> src4Global;
    AscendC::LocalTensor<float> src0Local, src1Local, src2Local, src3Local;
    AscendC::LocalTensor<int32_t> src4Local;
    int32_t blockIdx = -1;
};
