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

#include <vector>
#include <cstdint>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include "acl/acl.h"
#include "elewise/elewise_host.h"
#include "elewise/elewise_device.h" 

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0)

namespace {
static constexpr float REL_TOL = 1e-3f;
static constexpr float ABS_TOL = 1e-5f;

// 判断两个浮点数是否足够接近
bool IsClose(float a, float b)
{
    const float eps = 1e-40f; // 防止分母为零
    float diff = std::abs(a - b);
    return (diff <= ABS_TOL) || (diff <= REL_TOL * std::max(std::abs(a), std::abs(b) + eps));
}

// 描述算子的输入输出以及临时计算资源
using SinhOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>, ATVC::OpTemps<float, float>>;

// 传入编译态参数ATVC::OpTraits
template<class Traits>
// 开发自定义函数名/类名
struct SinhComputeFunc {
    // DataType模板参数，根据实际数据类型个数填写
    template<typename T, typename U>
    // 重载operator公有接口，提供给`ATVC::Kernel::EleWiseOpTemplate`调用
    __aicore__ inline void operator()(AscendC::LocalTensor<T> x,
                                      AscendC::LocalTensor<T> y,
                                      AscendC::LocalTensor<U> tempBuffer1,
                                      AscendC::LocalTensor<U> tempBuffer2)
    {
        // 开发调用AscendC Api自行实现计算仿函数
        uint32_t tiledCnt = y.GetSize(); // 进行单次基块计算的元素个数
        AscendC::Muls(tempBuffer1, x, static_cast<T>(-1), tiledCnt); // tempBuffer1 = -1 * x
        AscendC::Exp(tempBuffer1, tempBuffer1, tiledCnt); // tempbuffer1 = exp(-x)
        AscendC::Exp(tempBuffer2, x, tiledCnt);  // tempbuffer2 = exp(x)
        AscendC::Sub(y, tempBuffer2, tempBuffer1, tiledCnt); // y = exp(x) - exp(-x)
        AscendC::Muls(y, y, static_cast<T>(0.5), tiledCnt); // y = (e^(x) - e^(-x)) / 2 
    }
};

void InitializeData(int32_t eleNum, std::vector<float> &inputX, std::vector<float> &golden)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (int i = 0; i < eleNum; ++i) {
        inputX[i] = dis(gen);
        golden[i] = std::sinh(inputX[i]);
    }
}

bool VerifyResults(const std::vector<float> &golden, const std::vector<float> &output)
{
    for (int32_t i = 0; i < golden.size(); i++) {
        if (!IsClose(golden[i], output[i])) {
            printf("Accuracy verification failed! The expected value of element "
                   "in index [%d] is %f, but actual value is %f.\n",
                i,
                golden[i],
                output[i]);
            return false;
        }
    }
    return true;
}

void InitializeACL(aclrtContext &context, aclrtStream &stream, int32_t deviceId)
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));
}

void CleanACL(aclrtStream &stream, int32_t deviceId)
{
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}
}

/*
 * 该函数为SinhCustom算子核函数入口
 * x        Device上的gm地址，指向Add算子第一个输入
 * y        Device上的gm地址，指向Add算子第一个输出
 * param    Device上的gm地址，指向运行态ATVC::EleWiseParam数据
*/
template<class OpTraits>
__global__ __aicore__ void SinhCustom(GM_ADDR x, GM_ADDR y, GM_ADDR param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 控制算子执行时只启动Vector核
    auto op = ATVC::Kernel::EleWiseOpTemplate<SinhComputeFunc<OpTraits>>();
    op.Run(x, y, param);          // 按照输入、输出、param的顺序传入Run函数中；OpTraits内部的ATVC::OpTemps将由EleWiseOpTemplate内部申请资源，开发无需关注
}

int main()
{
    // init data
    int32_t eleNum = 8 * 2048;
    size_t inputByteSize = static_cast<size_t>(eleNum) * sizeof(float);
    size_t outputByteSize = static_cast<size_t>(eleNum) * sizeof(float);

    std::vector<float> inputX(eleNum);
    std::vector<float> golden(eleNum);
    InitializeData(eleNum, inputX, golden);

    ATVC::EleWiseParam param;

    // 计算输入为8*2048个float元素的sinh算子的运行态参数param
    if (!ATVC::Host::CalcEleWiseTiling<SinhOpTraits>(eleNum, param)) {
        printf("Elewise tiling error.");
        return -1;
    };
    // 初始化Acl资源与数据
    aclrtContext context;
    aclrtStream stream = nullptr;
    int32_t deviceId = 0;
    InitializeACL(context, stream, deviceId);

    uint8_t *yHost;
    uint8_t *xDevice;
    uint8_t *yDevice;
    uint8_t *paramDevice;
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, inputX.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMallocHost((void **)(&yHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // 将tiling计算的运行时参数EleWiseParam param传到Device侧
    auto elementParamSize = sizeof(param);
    CHECK_ACL(aclrtMalloc((void**)&paramDevice, elementParamSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(paramDevice, elementParamSize,
                          reinterpret_cast<uint8_t *>(&param), elementParamSize,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    // 调用自定义的Kernel API, <<<>>>的BlockNum参数可通过param的TilingData获取
    SinhCustom<SinhOpTraits><<<param.tilingData.blockNum, nullptr, stream>>>(xDevice, yDevice, paramDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(yHost, outputByteSize, yDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> outputY(reinterpret_cast<float*>(yHost), reinterpret_cast<float*>(yHost) + eleNum);

    // 释放资源
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(paramDevice));
    CHECK_ACL(aclrtFreeHost(yHost));

    CleanACL(stream, deviceId);

    if (!VerifyResults(golden, outputY)) {
        return -1;
    }
    printf("Accuracy verification passed.\n");
    return 0;
}