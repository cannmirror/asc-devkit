/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include <unistd.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "asc_info_manager.h"
#include "asc_host_code_generate.h"
#include "asc_host_binary_generator.h"

class TEST_ASC_HOST_BINARY_GENERATE : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_HOST_BINARY_GENERATE, asc_get_host_binary_succeed)
{
    std::string golden = R"(#include <stdio.h>
#include <stdint.h>
#include <vector>

namespace AscPluginGenerator {
constexpr unsigned int ascendcExceptionDumpHead = 2U;
typedef struct rtArgsSizeInfoAsc {
    void *infoAddr;
    uint32_t atomicIndex;
} rtArgsSizeInfoAsc_t;
} // namespace AscPluginGenerator

extern "C" {
int32_t rtSetExceptionExtInfo(const AscPluginGenerator::rtArgsSizeInfoAsc_t * const sizeInfo);
namespace Adx {
void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex);
} // namespace Adx
}

namespace AscPluginGenerator {
__attribute__ ((visibility("hidden"))) uint32_t ascendc_set_exception_dump_info(uint32_t dumpSize)
{
    uint32_t atomicIndex = 0U;
    constexpr uint32_t addrNum = 1U;
    void *exceptionDumpAddr = Adx::AdumpGetSizeInfoAddr(addrNum + ascendcExceptionDumpHead, atomicIndex);
    if (exceptionDumpAddr == nullptr) {
        ::printf("[ERROR] [AscPlugin] Get exceptionDumpAddr is nullptr.\n");
        return 1;
    }
    uint64_t *sizeInfoAddr = reinterpret_cast<uint64_t *>(exceptionDumpAddr);
    *sizeInfoAddr = static_cast<uint64_t>(atomicIndex);
    sizeInfoAddr++;
    *sizeInfoAddr = static_cast<uint64_t>(1);
    sizeInfoAddr++;
    *sizeInfoAddr = dumpSize * 75;
    constexpr uint64_t workspaceOffset = (4ULL << 56ULL);
    *sizeInfoAddr |= workspaceOffset;
    const rtArgsSizeInfoAsc sizeInfo = {exceptionDumpAddr, atomicIndex};
    int32_t ret = rtSetExceptionExtInfo(&sizeInfo);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] rtSetExceptionExtInfo failed, ret = %d.\n", ret);
        return 1;
    }
    return 0;
}
} // namespace AscPluginGenerator

extern "C" {
int32_t AscendDevBinaryRegister(const void *fileBuf, size_t fileSize, void **handle);
int32_t AscendKernelLaunchWithFlagV2(const char *stubFunc, const uint32_t blockDim, void **args,
    uint32_t size, const void *stream);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
void AscendProfRegister();
}

namespace {
char ascendcErrMsg[4096] = {0};
void *g_kernel_handle = nullptr;

typedef void (*KernelFuncRegister)(void*);

class AscPluginRegFuncRegister {
public:
    inline static AscPluginRegFuncRegister& GetInstance()
    {
        static AscPluginRegFuncRegister instance;
        return instance;
    }

public:
    std::vector<KernelFuncRegister> regFuncCallbackList;
private:
    AscPluginRegFuncRegister() = default;
    ~AscPluginRegFuncRegister() = default;
    AscPluginRegFuncRegister(const AscPluginRegFuncRegister&) = delete;
    AscPluginRegFuncRegister& operator=(const AscPluginRegFuncRegister&) = delete;
    AscPluginRegFuncRegister(AscPluginRegFuncRegister&&) = delete;
    AscPluginRegFuncRegister& operator=(AscPluginRegFuncRegister&&) = delete;
};

void RegisterKernels(void)
{
    int32_t ret;
    ret = AscendDevBinaryRegister(fatbinDataPtr, fatbinDataLength, &g_kernel_handle);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] Kernel binary register failure! ret %d \n", ret);
    }
    AscendProfRegister();
}

class KernelHandleGradUnregister {
private:
    KernelHandleGradUnregister() = default;
    ~KernelHandleGradUnregister() {
        if (g_kernel_handle) {
            UnregisterAscendBinary(g_kernel_handle);
            g_kernel_handle = nullptr;
        }
    }
    KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
    KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;
public:
    static KernelHandleGradUnregister& GetInstance() {
        static KernelHandleGradUnregister instance;
        return instance;
    }
};

class AscendCOperatorRegister {
public:
static AscendCOperatorRegister& GetInstance() {
    static AscendCOperatorRegister instance;
    return instance;
}
private:
AscendCOperatorRegister() {
    RegisterKernels();
    const auto& inst = AscPluginRegFuncRegister::GetInstance();
    for (auto func : inst.regFuncCallbackList) {
        func(g_kernel_handle);
    }
}
~AscendCOperatorRegister() = default;
AscendCOperatorRegister(const AscendCOperatorRegister&) = delete;
AscendCOperatorRegister& operator=(const AscendCOperatorRegister&) = delete;
};

} // namespace

namespace AscPluginGenerator {
__attribute__ ((visibility("hidden"))) int32_t BindKernelRegisterFunc(KernelFuncRegister func)
{
    auto& inst = AscPluginRegFuncRegister::GetInstance();
    inst.regFuncCallbackList.emplace_back(func);
    return 0;
}

__attribute__ ((visibility("hidden"))) void GetHandleUnregisterInst() {
    auto& regMng = KernelHandleGradUnregister::GetInstance();
}

__attribute__ ((visibility("hidden"))) uint32_t LaunchAndProfiling(
    const char *stubFunc, uint32_t blockDim, void *stream, void **args, uint32_t size, uint32_t ktype)
{
    const auto& reg = AscendCOperatorRegister::GetInstance();
    uint64_t startTime;
    const char *name = stubFunc;
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle == nullptr) {
        ::printf("[ERROR] [AscPlugin] %s\n", ascendcErrMsg);
        return 1;
    }
    int32_t retLaunch = AscendKernelLaunchWithFlagV2(stubFunc, blockDim, args, size, stream);
    if (retLaunch != 0) {
        ::printf("[ERROR] [AscPlugin] AscendKernelLaunchWithFlagV2 ret %u\n", retLaunch);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, ktype, startTime);
    }
    return retLaunch;
}

} // namespace AscPluginGenerator
)";
    auto HostBinaryResult = AscPlugin::GetBinaryRegisterCode();
    int32_t deviceResult = 0;
    if (HostBinaryResult != "") {
        deviceResult = 1;
    }
    EXPECT_EQ(deviceResult, 1);
    EXPECT_EQ(HostBinaryResult, golden);
}