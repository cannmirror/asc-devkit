/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "reduce/reduce_device.h"

using OpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int>, ATVC::OpOutputs<int>>;
using OpTraitsFloat = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;

template<typename Traits, const auto& Policy>
__global__ __aicore__ void ReduceCustom(GM_ADDR x, GM_ADDR y, GM_ADDR reduceParam)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0); // 使用了多核控制指令，设置算子执行时只启动Vector核
    // 将计算模板类模板定义作为模板参数传入， SelectPolicy由Host层的策略分派API给出
    auto op = ATVC::Kernel::ReduceOpTemplate<ATVC::ReduceSumCompute<Traits>, Policy>();
    op.Run(x, y, reduceParam);
}

void ReduceCustomInt0(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY0><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt1(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY1><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt2(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY2><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt3(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY3><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt4(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY4><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt5(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY5><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt6(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY6><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt7(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY7><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt8(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY8><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt9(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY9><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt10(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY10><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt11(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY11><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt12(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY12><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt13(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY13><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt14(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY14><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt15(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY15><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt16(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY16><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt17(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY17><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt18(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY18><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt19(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY19><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt20(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY20><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt21(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY21><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomInt22(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsInt, ATVC::REDUCE_POLICY22><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat0(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY0><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat1(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY1><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat2(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY2><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat3(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY3><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat4(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY4><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat5(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY5><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat6(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY6><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat7(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY7><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat8(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY8><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat9(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY9><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat10(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY10><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat11(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY11><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat12(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY12><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat13(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY13><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat14(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY14><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat15(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY15><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat16(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY16><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat17(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY17><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat18(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY18><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat19(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY19><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat20(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY20><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat21(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY21><<<blockDim, nullptr, stream>>>(x, y, param);
}

void ReduceCustomFloat22(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param) {
    ReduceCustom<OpTraitsFloat, ATVC::REDUCE_POLICY22><<<blockDim, nullptr, stream>>>(x, y, param);
}
