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
 * \file aicpu_rt.cpp
 * \brief
 */

#ifndef __AICPU_DEVICE__
#include "acl/acl.h"
#include "securec.h"
#include "aicpu_rt.h"
#include "ascendc_runtime.h"
#include "ascendc_tool_log.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <chrono>

#define CHECK_ACL_PTR(x)                                                                    \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            ASCENDLOGE("[Check]acl Error: %d.", __ret);                                    \
            return nullptr;                                                                 \
        }                                                                                   \
    } while (0)

#define CHECK_ACL_ERR(x)                                                                    \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            ASCENDLOGE("[Check]acl Error: %d.", __ret);;                                   \
            return __ret;                                                                   \
        }                                                                                   \
    } while (0)

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            ASCENDLOGE("[Check]acl Error: %d.", __ret);                                    \
            return;                                                                         \
        }                                                                                   \
    } while (0)

constexpr int32_t REFRESH_RATE = 0.1;
constexpr size_t DUMP_SIZE = 1048576;
constexpr uint32_t MAX_NAME_LEN = 256;

extern "C" {
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
int32_t ElfGetSymbolOffset(uint8_t* elf, size_t elfSize, const char* symbolName, size_t* offset, size_t* size);

void AicpuDumpPrintBuffer(const void *dumpBuffer, const size_t bufSize)
{
    if (dumpBuffer == nullptr || bufSize == 0) {
        ASCENDLOGE("AicpuDumpPrintBuffer: empty buffer.");
        return;
    }
    void *bufHost = malloc(bufSize);
    memset_s(bufHost, bufSize, 0, bufSize);
    CHECK_ACL(aclrtMemcpy(bufHost, bufSize, dumpBuffer, bufSize, ACL_MEMCPY_DEVICE_TO_HOST));
    static thread_local size_t lastOffSet = 8;
    size_t curOffSet = *reinterpret_cast<size_t*>(bufHost);
    if (curOffSet > lastOffSet) {
        printf("%s", reinterpret_cast<char*>(bufHost) + lastOffSet);
        lastOffSet = curOffSet;
    }
    free(bufHost);
}

AicpuDumpThreadRes::AicpuDumpThreadRes(const void* dumpAddr, const size_t dumpSize, const int32_t deviceId)
    : thread_([dumpAddr, dumpSize, deviceId, args = &mutex_]()
    {
        CHECK_ACL(aclrtSetDevice(deviceId));
        while (!args->stop.load(std::memory_order_relaxed)) {
            AicpuDumpPrintBuffer(dumpAddr, dumpSize);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 100 means 100 ms
        }
        CHECK_ACL(aclrtResetDevice(deviceId));
    }) {}

AicpuDumpThreadRes::~AicpuDumpThreadRes()
{
    mutex_.stop.store(true, std::memory_order_relaxed);
    thread_.join();
}

int AicpuGetDumpConfig(void **addr, size_t *size)
{
    static void *dumpAddr[16] = {nullptr};
    static std::mutex dumpMutex;
    int32_t deviceId = -1;
    CHECK_ACL_ERR(aclrtGetDevice(&deviceId));

    if (dumpAddr[deviceId] != nullptr) {
        *addr = dumpAddr[deviceId];
        *size = DUMP_SIZE;
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(dumpMutex);
        if (dumpAddr[deviceId] == nullptr) {
            void *device = nullptr;
            CHECK_ACL_ERR(aclrtMalloc(reinterpret_cast<void**>(&device), 1048576, ACL_MEM_MALLOC_HUGE_FIRST));
            dumpAddr[deviceId] = device;
            uint64_t bufferOffSet = 8;
            CHECK_ACL_ERR(aclrtMemcpy(device, 8, &bufferOffSet, 8, ACL_MEMCPY_HOST_TO_DEVICE));
            *addr = dumpAddr[deviceId];
            *size = DUMP_SIZE;
            static AicpuDumpThreadRes dumpThread(*addr, *size, deviceId);
        }
    }
    return 0;
}

size_t* AicpuSetDumpConfig(const unsigned long *aicpuFileBuf, size_t fileSize) {
    void *dumpAddr = nullptr;
    size_t dumpSize = 0;
    AicpuGetDumpConfig(&dumpAddr, &dumpSize);
    size_t *kernelBuf = reinterpret_cast<size_t*>(malloc(fileSize));
    memcpy_s(kernelBuf, fileSize, aicpuFileBuf, fileSize);
    size_t startIndex = 0, symbolSize = 0;
    int32_t ret = ElfGetSymbolOffset(reinterpret_cast<uint8_t*>(kernelBuf), fileSize, "g_aicpuDumpConfig", &startIndex,
        &symbolSize);
    if (ret != 0) {
        if (ret == 1) {
            free(kernelBuf);
            ASCENDLOGE("elf is not legal, please check log!");
            return nullptr;
        } else if (ret == 2) {  // 2 means no symbol: g_aicpuDumpConfig
            ASCENDLOGI("dump switch is off.");
        }
    } else {
        startIndex /= sizeof(size_t);
        kernelBuf[startIndex] = static_cast<size_t>(reinterpret_cast<uintptr_t>(dumpAddr));
        kernelBuf[startIndex + 1] = dumpSize;
    }
    return kernelBuf;
}

aclrtBinHandle AicpuLoadBinaryFromBuffer(const unsigned long *aicpuFileBuf, size_t fileSize)
{
    void *dumpAddr = nullptr;
    size_t dumpSize = 0;
    AicpuGetDumpConfig(&dumpAddr, &dumpSize);
    size_t *kernelBuf = reinterpret_cast<size_t*>(malloc(fileSize));
    memcpy_s(kernelBuf, fileSize, aicpuFileBuf, fileSize);
    size_t startIndex = 0, symbolSize = 0;
    int32_t ret = ElfGetSymbolOffset(reinterpret_cast<uint8_t*>(kernelBuf), fileSize, "g_aicpuDumpConfig", &startIndex,
        &symbolSize);
    if (ret != 0) {
        if (ret == 1) {
            free(kernelBuf);
            ASCENDLOGE("elf is not legal, please check log!");
            return nullptr;
        } else if (ret == 2) {  // 2 means no symbol: g_aicpuDumpConfig
            ASCENDLOGI("dump switch is off.");
        }
    } else {
        startIndex /= sizeof(size_t);
        kernelBuf[startIndex] = static_cast<size_t>(reinterpret_cast<uintptr_t>(dumpAddr));
        kernelBuf[startIndex + 1] = dumpSize;
    }
    aclrtBinaryLoadOption bopt;
    bopt.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    bopt.value.cpuKernelMode = 2;  // 2 means without op.ini and op.lf
    aclrtBinaryLoadOptions bcfg = {0};
    bcfg.options = &bopt;
    bcfg.numOpt = 1;
    aclrtBinHandle bhdl = nullptr;
    CHECK_ACL_PTR(aclrtBinaryLoadFromData(reinterpret_cast<const char*>(kernelBuf), fileSize, &bcfg, &bhdl));
    free(kernelBuf);
    return bhdl;
}

aclrtFuncHandle AicpuRegFunctionByName(const aclrtBinHandle binHandle, const char *funcName)
{
    aclrtFuncHandle fhdl;
    CHECK_ACL_PTR(aclrtRegisterCpuFunc(binHandle, funcName, funcName, &fhdl));
    return fhdl;
}

void AicpuLaunchKernel(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, void *arg, size_t argSize)
{
    uint64_t startTime = 0;
    char funcName[MAX_NAME_LEN];
    funcName[0] = 0;
    if (GetAscendProfStatus()) {
        CHECK_ACL(aclrtGetFunctionName(funcHandle, MAX_NAME_LEN, funcName));
        StartAscendProf(funcName, &startTime);
    }
    aclrtArgsHandle ahdl = nullptr;
    CHECK_ACL(aclrtKernelArgsInit(funcHandle, &ahdl));
    if (argSize != 0 && arg != nullptr) {
        aclrtParamHandle phdl = nullptr;
        CHECK_ACL(aclrtKernelArgsAppend(ahdl, arg, argSize, &phdl));
    }
    CHECK_ACL(aclrtKernelArgsFinalize(ahdl));
    CHECK_ACL(aclrtLaunchKernelWithConfig(funcHandle, blockDim, stream, nullptr, ahdl, nullptr));
    if (GetAscendProfStatus()) {
        ReportAscendProf(funcName, blockDim, 3, startTime);   // 3 is means WRITE_BACK
    }
}

}
#endif
