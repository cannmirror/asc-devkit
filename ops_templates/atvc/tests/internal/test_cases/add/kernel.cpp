/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "elewise/elewise_device.h"

using ADD_FLOAT = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;
using ADD_INT = ATVC::OpTraits<ATVC::OpInputs<int, int>, ATVC::OpOutputs<int>>;

template<typename Traits>
struct AddComputeFunc {
    /*
    函数说明： c = a + b
    参数说明：
        a                   : 参与运算的输入
        b                   : 参与运算的输入
        c                   : 参与运算的输出
    */
    template<typename T> 
    __aicore__ inline void operator()(AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, AscendC::LocalTensor<T> c) {
        AscendC::Add(c, a, b, c.GetSize()); // 开发调用AscendC Api自行实现计算逻辑, 通过c.GetSize()获取单次计算的元素数量
    }
};

template<class Traits>
/*
 * 该函数为Add算子核函数入口
 * a        Device上的gm地址，指向Add算子第一个输入
 * b        Device上的gm地址，指向Add算子第二个输入
 * c        Device上的gm地址，指向Add算子第一个输出
 * param    Device上的gm地址，指向运行态ATVC::EleWiseParam数据
*/
__global__ __aicore__ void AddCustom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto op = ATVC::Kernel::EleWiseOpTemplate<AddComputeFunc<Traits>>();  // 将AddComputeFunc仿函数作为模板参数传入，实例化EleWiseOpTemplate模板类
    op.Run(a, b, c, param); // 按照输入、输出、param的顺序传入Run函数，实现GM->GM的数据计算
}


void AddCustomInt(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* param) {
    AddCustom<ADD_INT><<<blockDim, nullptr, stream>>>(x, y, z, param);
}

void AddCustomFloat(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* param) {
    AddCustom<ADD_FLOAT><<<blockDim, nullptr, stream>>>(x, y, z, param);
}