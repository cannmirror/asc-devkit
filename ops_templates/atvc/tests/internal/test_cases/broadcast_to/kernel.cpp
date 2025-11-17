/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "broadcast/broadcast_device.h"

using OpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int>, ATVC::OpOutputs<int>>;
using OpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;

template<typename Traits, const auto& Policy>
__global__ __aicore__ void BroadcastCustom(GM_ADDR x, GM_ADDR y, GM_ADDR broadcastParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0); // 使用了多核控制指令，设置算子执行时只启动Vector核
    // 将计算模板类模板定义作为模板参数传入， SelectPolicy由Host层的策略分派API给出
    auto op = ATVC::Kernel::BroadcastOpTemplate<ATVC::BroadcastCompute<Traits>, Policy>();
    op.Run(x, y, broadcastParam);
}

void BroadcastCustomInt0(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    BroadcastCustom<OpTraitsInt, ATVC::BROADCAST_POLICY0><<<blockDim, nullptr, stream>>>(x, y, param);
}

void BroadcastCustomInt1(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    BroadcastCustom<OpTraitsInt, ATVC::BROADCAST_POLICY1><<<blockDim, nullptr, stream>>>(x, y, param);
}

void BroadcastCustomFloat0(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    BroadcastCustom<OpTraitsFloat, ATVC::BROADCAST_POLICY0><<<blockDim, nullptr, stream>>>(x, y, param);
}

void BroadcastCustomFloat1(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    BroadcastCustom<OpTraitsFloat, ATVC::BROADCAST_POLICY1><<<blockDim, nullptr, stream>>>(x, y, param);
}
