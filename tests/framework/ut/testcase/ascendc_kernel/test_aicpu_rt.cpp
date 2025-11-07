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
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "securec.h"
#include "aicpu_api/aicpu_api.h"
#include "aicpu_dump_utils.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdint>
#include <mockcpp/mockcpp.hpp>

class TEST_AICPU_RT : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

extern "C" {
    aclrtBinHandle AicpuLoadBinaryFromBuffer(const unsigned long *aicpuFileBuf, size_t fileSize);
    aclrtFuncHandle AicpuRegFunctionByName(const aclrtBinHandle binHandle, const char *funcName);
    void AicpuLaunchKernel(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, void *arg,
        size_t argSize);
    void AicpuDumpPrintBuffer(const void *dumpBuffer, const size_t bufSize);
    int AicpuGetDumpConfig(void **addr, size_t *size);
    void StartAscendProf(const char *name, uint64_t *startTime);
    void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
    int AscendProfRegister();
    bool GetAscendProfStatus();
    int32_t ElfGetSymbolOffset(uint8_t* elf, size_t elfSize, const char* symbolName, size_t* offset, size_t* size);
}

int AicpuGetDumpConfigStub(void **addr, size_t *size)
{
    char buffer[16] = {0};
    *addr = buffer;
    *size = 1048576;
    return 0;
}

TEST_F(TEST_AICPU_RT, AicpuLoadBinaryFromBufferReturnNullptr)
{
    const size_t fileSize = 1024;
    unsigned long* aicpuFileBuf = new unsigned long[fileSize / sizeof(unsigned long)];
    memset(aicpuFileBuf, 0, fileSize);
    MOCKER(ElfGetSymbolOffset).stubs().will(returnValue(1));
    MOCKER(AicpuGetDumpConfig).stubs().will(invoke(AicpuGetDumpConfigStub));
    EXPECT_EQ(AicpuLoadBinaryFromBuffer(aicpuFileBuf, fileSize), nullptr);
    delete[] aicpuFileBuf;
}

TEST_F(TEST_AICPU_RT, AicpuLoadBinaryFromBufferTest)
{
    const size_t fileSize = 1024;
    unsigned long* aicpuFileBuf = new unsigned long[fileSize / sizeof(unsigned long)];
    memset(aicpuFileBuf, 0, fileSize);
    MOCKER(ElfGetSymbolOffset).stubs().will(returnValue(0));
    MOCKER(AicpuGetDumpConfig).stubs().will(invoke(AicpuGetDumpConfigStub));
    EXPECT_EQ(AicpuLoadBinaryFromBuffer(aicpuFileBuf, fileSize), nullptr);
    delete[] aicpuFileBuf;
}

TEST_F(TEST_AICPU_RT, AicpuRegFunctionByNameTest)
{
    aclrtBinHandle binHandle;
    const char *funcName = "testFunc";
    EXPECT_NO_THROW(AicpuRegFunctionByName(binHandle, funcName));
}

TEST_F(TEST_AICPU_RT, AicpuLaunchKernelTest)
{
    aclrtFuncHandle funcHandle;
    uint32_t blockDim = 1;
    aclrtStream stream;
    char argBuffer[256];
    void *arg = &argBuffer;
    size_t argSize = sizeof(argBuffer);
    MOCKER(GetAscendProfStatus).stubs().will(returnValue(true));
    EXPECT_NO_THROW(AicpuLaunchKernel(funcHandle, blockDim, stream, arg, argSize));
}

int aclrtGetDeviceStub(int32_t *devicdId)
{
    *devicdId = 0;
    return 0;
}

int aclrtMallocStub(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    *devPtr = malloc(size);
    return 0;
}

TEST_F(TEST_AICPU_RT, AicpuGetDumpConfigTest)
{
    char buffer[16] = {0};
    void *addr = buffer;
    size_t size = 1048576;
    MOCKER(aclrtGetDevice).stubs().will(invoke(aclrtGetDeviceStub));
    MOCKER(aclrtMalloc).stubs().will(invoke(aclrtMallocStub));
    EXPECT_EQ(AicpuGetDumpConfig(&addr, &size), 0);
}

TEST_F(TEST_AICPU_RT, AicpuDumpPrintBufferReturn)
{
    char buffer[16] = {0};
    void *addr = buffer;
    size_t size = 0;
    EXPECT_NO_THROW(AicpuDumpPrintBuffer(addr, size));
}

TEST_F(TEST_AICPU_RT, AicpuDumpPrintBufferTest)
{
    char buffer[16] = {0};
    void *addr = buffer;
    size_t size = 1048576;
    EXPECT_NO_THROW(AicpuDumpPrintBuffer(addr, size));
}
