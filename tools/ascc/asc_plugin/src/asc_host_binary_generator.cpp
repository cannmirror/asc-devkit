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
 * \file asc_host_binary_generator.cpp
 * \brief
 */

#include "asc_host_binary_generator.h"

#include "asc_info_manager.h"
#include "asc_log.h"

namespace AscPlugin {
AscHostBinaryGenerator::AscHostBinaryGenerator()
{
    std::string buffer;
    constexpr size_t codeBuffLen = 16 * 1024;
    buffer.reserve(codeBuffLen);
    codeStream_.str(std::move(buffer));
}

static const char *BINARY_REGISTER_CODE = R"(#include <stdio.h>
#include <stdint.h>
extern "C" {
int32_t AscendDevBinaryLazyRegister(const char* binBuf, size_t binSize, void** handle);
int32_t AscendGetFuncFromBinary(void* const binHandle, const char* kernelName, void** funcHandle);
int32_t AscendLaunchKernelWithHostArgs(void* funcHandle,
    uint32_t blockDim, void* stream, void* hostArgs, size_t argsSize, uint32_t ubufDynamicSize);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
void AscendProfRegister();
using rtFuncHandle = void*;
uint32_t AscendCGetProfkTypeImpl(const rtFuncHandle funcHandle);
}

namespace {
class AscRegister {
public:
static AscRegister& GetInstance() {
    static AscRegister instance;
    return instance;
}
void* binHandle = nullptr;
private:
AscRegister() {
    uint32_t ret = AscendDevBinaryLazyRegister(fatbinDataPtr, fatbinDataLength, &binHandle);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] Kernel binary register failure! ret %d \n", ret);
    }
    AscendProfRegister();
}
~AscRegister() = default;
AscRegister(const AscRegister&) = delete;
AscRegister& operator=(const AscRegister&) = delete;
};

} // namespace

namespace AscPluginGenerator {
__attribute__ ((visibility("hidden"))) int32_t BindKernelRegisterFunc(void (*)(void*)) { return 0; }
__attribute__ ((visibility("hidden"))) uint32_t LaunchAndProfiling(const char *kernelName, uint32_t blockDim,
    void *stream, void **args, uint32_t size, uint32_t ktype, const uint32_t ubufDynamicSize)
{
    static auto& reg = AscRegister::GetInstance();
    void* funcHandle = nullptr;
    uint32_t ret = AscendGetFuncFromBinary(reg.binHandle, kernelName, &funcHandle);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] Get kernel function failure! ret %d \n", ret);
        return 1;
    }
    uint64_t startTime;
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        ktype = AscendCGetProfkTypeImpl(funcHandle);
        StartAscendProf(kernelName, &startTime);
    }
    ret = AscendLaunchKernelWithHostArgs(funcHandle, blockDim, stream, (void*)args, size, ubufDynamicSize);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] Launch kernel failure! ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(kernelName, blockDim, ktype, startTime);
    }
    return ret;
}
} // namespace AscPluginGenerator
)";

std::string AscHostBinaryGenerator::GenCode()
{
    codeStream_ << BINARY_REGISTER_CODE;
    ASC_LOGD("registerBinaryCode is [\n%s\n]", codeStream_.str().c_str());
    return codeStream_.str();
}
}  // namespace AscPlugin