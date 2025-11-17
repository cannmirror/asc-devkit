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
* \file main.cpp
* \brief
*/

#include <iostream>
#include <cstdint>
#include <sstream>
#include <acl/acl.h>

#include "tiling/platform/platform_ascendc.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/matmul/block/block_mmad_builder.h"
#include "include/matmul/kernel/kernel_matmul.h"
#include "include/matmul/device/device_matmul.h"
#include "include/utils/host_utils.h"
#include "include/utils/layout_utils.h"
#include "include/utils/status_utils.h"
#include "../utils.h"

using namespace Act;
using namespace Act::Gemm;

bool isBias = false;

// 定义L1和L0的TileShape
using L1TileShape = AscendC::Shape<_128, _256, _256>;
using L0TileShape = AscendC::Shape<_128, _256, _64>;

// 定义矩阵的类型和布局 支持half和bfloat16_t类型, 需要根据实际情况修改
using AType = half;
using BType = half;
using CType = half;

using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

// 定义scheduler类型
using BlockScheduler = MisplaceCoreScheduler;

// 定义MMAD类型
using BlockMmad = Block::BlockMmadBuilder<
        AType, LayoutA, BType, LayoutB, CType, LayoutC, CType, LayoutC,
        L1TileShape, L0TileShape, BlockScheduler, MatmulMultiBlockWithLayout<>>;

// 定义BlockEpilogue类型
using BlockEpilogue = Block::BlockEpilogueEmpty;

// 定义shape的形状，tuple保存 m n k batch
using ProblemShape = MatmulShape;

// 定义Kernel类型
using MatmulKernel = Kernel::KernelMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
using Arguments = typename MatmulKernel::Arguments;

// 定义deviceMatmul
using DeviceMatmul = Device::DeviceMatmul<MatmulKernel>;

void MatmulOp(uint8_t* x1, uint8_t* x2, uint8_t* y, uint8_t* bias, int64_t m, int64_t n, int64_t k, int64_t b,
    void* stream = nullptr)
{
    // Init args
    uint8_t *workspaceDevice;
    MatmulShape shape {m, n, k, b};
    Arguments args = {
        shape,                // problem shape
        {x1, x2, y, bias},    // mmad args
        {}                    // epilogue args
    };

    // Instantiate matmul with specfied kernel
    DeviceMatmul mm;

    // Query workspace size
    size_t workspaceSize = DeviceMatmul::GetWorkspaceSize(args);

    // Allocate workspace on device
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACT_CHECK(mm.CanImplement(args));

    // Initialize kernel with arguments and workspace pointer
    mm.InitParams(args, workspaceDevice);

    // Launch kernel
    mm();

    CHECK_ACL(aclrtFree(workspaceDevice));
}

void TestAclInit(aclrtContext &context, aclrtStream &stream, int64_t &deviceId)
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));
}

void TestAclDeInit(aclrtContext &context, aclrtStream &stream, int64_t &deviceId)
{
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
}

void TestMatmul(int64_t m, int64_t n, int64_t k, int64_t b)
{
    size_t x1FileSize = b * m * k * sizeof(half);
    size_t x2FileSize = b * k * n * sizeof(half);
    size_t yFileSize = b * m * n * sizeof(half);
    size_t biasFileSize = 1 * n * sizeof(half);

    aclrtContext context;
    aclrtStream stream = nullptr;
    int64_t deviceId = 0;
    TestAclInit(context, stream, deviceId);

    uint8_t *x1Host;
    uint8_t *x1Device;
    CHECK_ACL(aclrtMallocHost((void **)(&x1Host), x1FileSize));
    CHECK_ACL(aclrtMalloc((void **)&x1Device, x1FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("../input/x1_gm.bin", x1FileSize, x1Host, x1FileSize);
    CHECK_ACL(aclrtMemcpy(x1Device, x1FileSize, x1Host, x1FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *x2Host;
    uint8_t *x2Device;
    CHECK_ACL(aclrtMallocHost((void **)(&x2Host), x2FileSize));
    CHECK_ACL(aclrtMalloc((void **)&x2Device, x2FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("../input/x2_gm.bin", x2FileSize, x2Host, x2FileSize);
    CHECK_ACL(aclrtMemcpy(x2Device, x2FileSize, x2Host, x2FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *biasHost = nullptr;
    uint8_t *biasDevice = nullptr;
    if (isBias) {
        CHECK_ACL(aclrtMallocHost((void **)(&biasHost), biasFileSize));
        CHECK_ACL(aclrtMalloc((void **)&biasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ReadFile("../input/bias_gm.bin", biasFileSize, biasHost, biasFileSize);
        CHECK_ACL(aclrtMemcpy(biasDevice, biasFileSize, biasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    uint8_t *yHost = nullptr;
    uint8_t *yDevice = nullptr;
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), yFileSize));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, yFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    MatmulOp(x1Device, x2Device, yDevice, biasDevice, m, n, k, b, stream);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(yHost, yFileSize, yDevice, yFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("../output/output.bin", yHost, yFileSize);

    if (isBias) {
        CHECK_ACL(aclrtFree(biasDevice));
        CHECK_ACL(aclrtFreeHost(biasHost));
    }
    CHECK_ACL(aclrtFree(x1Device));
    CHECK_ACL(aclrtFreeHost(x1Host));
    CHECK_ACL(aclrtFree(x2Device));
    CHECK_ACL(aclrtFreeHost(x2Host));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFreeHost(yHost));
    TestAclDeInit(context, stream, deviceId);
}

int32_t main(int32_t argc, const char *args[])
{
    int64_t problem[4] = {1, 1, 1, 1};

    for (int32_t i = 1; i < argc && i < 5; ++i) { // 4
        std::stringstream ss(args[i]);
        ss >> problem[i - 1];
    }

    TestMatmul(
        problem[0],
        problem[1],
        problem[2],
        problem[3] // b
    );

    return 0;
}
