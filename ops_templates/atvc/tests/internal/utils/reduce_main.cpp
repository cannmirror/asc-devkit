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
#include <string>
#include "data_utils.h"
#include "acl/acl.h"
#include "reduce/reduce_host.h"

void ReduceCustomInt0(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt1(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt2(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt3(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt4(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt5(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt6(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt7(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt8(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt9(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt10(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt11(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt12(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt13(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt14(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt15(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt16(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt17(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt18(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt19(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt20(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt21(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomInt22(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat0(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat1(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat2(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat3(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat4(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat5(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat6(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat7(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat8(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat9(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat10(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat11(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat12(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat13(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat14(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat15(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat16(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat17(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat18(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat19(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat20(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat21(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);
void ReduceCustomFloat22(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, uint8_t* param);

template<class OpTraits>
void ReduceOpAdapter(uint8_t* x, uint8_t* y, ATVC::ReduceParam &param, ATVC::ReducePolicy &policy, aclrtStream& stream, bool enableProf)
{
    using Inputs = typename OpTraits::In::types;
    using T = typename ATVC::TypeListGet<Inputs, 0>::Type;
    // 申请临时空间workspace，并将其与ReduceTilingData一同传到Device侧
    uint8_t *paramDevice;
    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, param.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    param.workspaceAddr = reinterpret_cast<uint64_t>(workspaceDevice);
    auto reduceParamSize = sizeof(param);
    CHECK_ACL(aclrtMalloc((void**)&paramDevice, reduceParamSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(paramDevice, reduceParamSize, reinterpret_cast<uint8_t*>(&param), reduceParamSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 将tiling api计算出的ReducePolicy转化为编译态参数并实例化相应的核函数
    int32_t loopCnt = 1;
    if (enableProf) {
        loopCnt = 20; // 循环20次
    }
    for (int32_t i = 0; i < loopCnt; i++) {
        if constexpr (std::is_same<T, int>::value) {
            if (policy == ATVC::REDUCE_POLICY0) {
                ReduceCustomInt0(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY1) {
                ReduceCustomInt1(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY2) {
                ReduceCustomInt2(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY3) {
                ReduceCustomInt3(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY4) {
                ReduceCustomInt4(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY5) {
                ReduceCustomInt5(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY6) {
                ReduceCustomInt6(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY7) {
                ReduceCustomInt7(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY8) {
                ReduceCustomInt8(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY9) {
                ReduceCustomInt9(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY10) {
                ReduceCustomInt10(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY11) {
                ReduceCustomInt11(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY12) {
                ReduceCustomInt12(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY13) {
                ReduceCustomInt13(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY14) {
                ReduceCustomInt14(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY15) {
                ReduceCustomInt15(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY16) {
                ReduceCustomInt16(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY17) {
                ReduceCustomInt17(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY18) {
                ReduceCustomInt18(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY19) {
                ReduceCustomInt19(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY20) {
                ReduceCustomInt20(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY21) {
                ReduceCustomInt21(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY22) {
                ReduceCustomInt22(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else {
                printf("[ERROR] Cannot find any matched policy.");
            }
        } else {
            if (policy == ATVC::REDUCE_POLICY0) {
                ReduceCustomFloat0(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY1) {
                ReduceCustomFloat1(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY2) {
                ReduceCustomFloat2(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY3) {
                ReduceCustomFloat3(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY4) {
                ReduceCustomFloat4(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY5) {
                ReduceCustomFloat5(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY6) {
                ReduceCustomFloat6(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY7) {
                ReduceCustomFloat7(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY8) {
                ReduceCustomFloat8(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY9) {
                ReduceCustomFloat9(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY10) {
                ReduceCustomFloat10(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY11) {
                ReduceCustomFloat11(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY12) {
                ReduceCustomFloat12(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY13) {
                ReduceCustomFloat13(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY14) {
                ReduceCustomFloat14(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY15) {
                ReduceCustomFloat15(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY16) {
                ReduceCustomFloat16(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY17) {
                ReduceCustomFloat17(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY18) {
                ReduceCustomFloat18(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY19) {
                ReduceCustomFloat19(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY20) {
                ReduceCustomFloat20(param.tilingData.coreNum, stream, x, y, paramDevice);
            } else if (policy == ATVC::REDUCE_POLICY21) {
                ReduceCustomFloat21(param.tilingData.coreNum, stream, x, y, paramDevice);
            }else if (policy == ATVC::REDUCE_POLICY22) {
                ReduceCustomFloat22(param.tilingData.coreNum, stream, x, y, paramDevice);
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
    std::vector<int64_t> dim = GetInputArgs(argv[3]);           // 第3个输入，表示reduceDim
    bool intDtype = std::string(argv[4]) == "0";                // 第4个输入，表示算子数据类型
    bool enableProf = std::string(argv[5]) == "1";              // 第5个输入，表示是否使能Profling
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

    ATVC::ReduceParam param;    // Reduce运行态参数，包含TilingData以及临时空间的相关信息
    ATVC::ReducePolicy policy;  // Reduce运行态参数，负责映射最适合的Reduce模板实现
    // Host侧调用Tiling API完成相关运行态参数的运算
    if (!ATVC::Host::CalcReduceTiling<OpTraitsInt>(shape, dim, &policy, &param)) {
        printf("Reduce tiling error.");
        return false;
    };
    if (intDtype) {
        ReduceOpAdapter<OpTraitsInt>(xDevice, yDevice, param, policy, stream, enableProf);
    } else {
        ReduceOpAdapter<OpTraits>(xDevice, yDevice, param, policy, stream, enableProf);
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