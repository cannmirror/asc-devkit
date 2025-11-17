/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
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
#include <string>
#include "data_utils.h"
#include "acl/acl.h"
#include "broadcast/broadcast_host.h"

void BroadcastCustomInt0(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void BroadcastCustomInt1(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void BroadcastCustomFloat0(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void BroadcastCustomFloat1(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);

template<class OpTraits>
void BroadcastOpAdapter(uint8_t* x, uint8_t* y, ATVC::BroadcastParam &param, ATVC::BroadcastPolicy &policy, aclrtStream& stream, bool enableProf)
{
    using Inputs = typename OpTraits::In::types;
    using T = typename ATVC::TypeListGet<Inputs, 0>::Type;
    // 申请临时空间workspace，并将其与BroadcastTilingData一同传到Device侧
    uint8_t *paramDevice;
    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, param.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspaceDevice);
    auto broadcastParamSize = sizeof(param);
    CHECK_ACL(aclrtMalloc((void**)&paramDevice, broadcastParamSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(paramDevice, broadcastParamSize, reinterpret_cast<uint8_t*>(&param), broadcastParamSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 将tiling api计算出的BroadcastPolicy转化为编译态参数并实例化相应的核函数
    int32_t loopCnt = 1;
    if (enableProf) {
        loopCnt = 20; // 循环20次
    }
    for (int32_t i = 0; i < loopCnt; i++) {
        if constexpr (std::is_same<T, int>::value) {
            if (policy == ATVC::BROADCAST_POLICY0) {
                BroadcastCustomInt0(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::BROADCAST_POLICY1) {
                BroadcastCustomInt1(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else {
                printf("[ERROR] Cannot find any matched policy.");
            }
        } else {
            if (policy == ATVC::BROADCAST_POLICY0) {
                BroadcastCustomFloat0(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::BROADCAST_POLICY1) {
                BroadcastCustomFloat1(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else {
                printf("[ERROR] Cannot find any matched policy.");
            }
        }
    }
    
    // 流同步后释放申请的param内存
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtFree(paramDevice));
}

std::vector<int64_t> GetInputArgs(char* args)
{
    std::string ss(args);
    std::vector<int64_t> vec;
    auto lastPos = ss.find_first_not_of(',', 0);
    auto pos = ss.find_first_of(',', lastPos);
    while(pos != std::string::npos || lastPos != std::string::npos) {
        vec.push_back(std::stoll(ss.substr(lastPos, pos - lastPos)));
        lastPos = ss.find_first_not_of(',', pos);
        pos = ss.find_first_of(',', lastPos);
    }
    return vec;
}

size_t GetShapeSize(std::vector<int64_t> vec) {
    size_t shapeSize = 1;
    for (const auto &s : vec) {
        shapeSize *= s;
    }
    return shapeSize;
}

int32_t main(int32_t argc, char* argv[])
{
    std::vector<int64_t> shape = GetInputArgs(argv[1]);         // 第1个输入，表示输入shape
    std::vector<int64_t> outputShape = GetInputArgs(argv[2]);   // 第2个输入，表示输出shape
    bool intDtype = std::string(argv[3]) == "0";                // 第3个输入，表示算子数据类型
    bool enableProf = std::string(argv[4]) == "1";              // 第4个输入，表示是否使能Profling
    size_t xByteSize = 4 * GetShapeSize(shape);
    size_t yByteSize = 4 * GetShapeSize(outputShape);
    using OpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
    using OpTraitsInt = ATVC::OpTraits<ATVC::OpInputs<int>, ATVC::OpOutputs<int>>;

    uint8_t *xHost;
    uint8_t *yHost;

    // 初始化Acl资源
    CHECK_ACL(aclInit({}));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xDevice;
    uint8_t *yDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&xHost), xByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), yByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/input_x.bin", xByteSize, xHost, xByteSize);
    CHECK_ACL(aclrtMalloc((void **)&yDevice, yByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(xDevice, xByteSize, xHost, xByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ATVC::BroadcastParam param;    // Broadcast运行态参数，包含TilingData以及临时空间的相关信息
    ATVC::BroadcastPolicy policy;  // Broadcast运行态参数，负责映射最适合的Broadcast模板实现
    // Host侧调用Tiling API完成相关运行态参数的运算
    if (!ATVC::Host::CalcBroadcastTiling<OpTraitsInt>(shape, outputShape, &policy, &param)) {
        printf("Broadcast tiling error.");
        return false;
    };
    if (intDtype) {
        BroadcastOpAdapter<OpTraitsInt>(xDevice, yDevice, param, policy, stream, enableProf);
    } else {
        BroadcastOpAdapter<OpTraits>(xDevice, yDevice, param, policy, stream, enableProf);
    }

    CHECK_ACL(aclrtMemcpy(yHost, yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_y.bin", yHost, yByteSize);

    // 释放Acl资源
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}