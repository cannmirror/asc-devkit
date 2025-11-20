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
 * \file asc_dev_funcRegistry_generate.cpp
 * \brief
 */

#include "asc_dev_funcRegistry_generate.h"
#include "asc_log.h"

#include <sstream>

namespace AscPlugin {
constexpr size_t FUNCREG_SIZE_CODE_BUFFER_LEN = 16 * 1024;
static const char *FUNC_REGISTER_CODE = R"(#include <stdio.h>
#include <cstdint>
extern "C" {
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
int32_t AscendFunctionRegister(void *handle, const char *stubFunc);
uint32_t GetAscendCoreSyncAddr(void **addr);
}
namespace Adx {
void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
    void *stream, const char *opType);
} // namespace Adx
namespace AscPluginGenerator {
typedef void (*KernelFuncRegister)(void*);
int32_t BindKernelRegisterFunc(KernelFuncRegister func);
uint32_t LaunchAndProfiling(
    const char *stubFunc, uint32_t blockDim, void *stream, void **args, uint32_t size, uint32_t ktype);
void GetHandleUnregisterInst();
uint32_t ascendc_set_exception_dump_info(uint32_t dumpSize);
} // namespace AscPluginGenerator

#define ASC_PLUGIN_LAUNCH_LOGE(kernelName, stream, blockDim, fmt, ...)                      \
    ::printf("[ERROR] [AscPlugin] Kernel: [%s] Stream: [%p] BlockDim: [%u] " fmt "\n",      \
        kernelName,                                                                         \
        stream,                                                                             \
        blockDim,                                                                           \
        ##__VA_ARGS__)

)";

std::string FunctionRegistryImpl()
{
    std::stringstream codeSource;
    std::string buffer;
    buffer.reserve(FUNCREG_SIZE_CODE_BUFFER_LEN);
    codeSource.str(std::move(buffer));
    codeSource << FUNC_REGISTER_CODE;
    codeSource << "static void AscFunctionRegister(void* g_kernel_handle)\n{\n";
    codeSource << "    int32_t retRegister = 0;\n";
    codeSource << "    const char *kernelFuncMangling = nullptr;\n";
    for (const auto& GlobalSymbolInfo : AscPlugin::InfoManager::GetInstance().GetGlobalSymbolInfo()) {
        codeSource << "    kernelFuncMangling = \"";
        codeSource << GlobalSymbolInfo.first.substr(DEVICE_STUB_PREFIX_LEN);
        codeSource << "\";\n";
        codeSource << "    retRegister = AscendFunctionRegister(g_kernel_handle, kernelFuncMangling);\n";
        codeSource << "    if (retRegister != 0) {\n";
        codeSource << "        ::printf(\"[ERROR] [AscPlugin] Kernel [%s] : function register failure! ret %d\\n\", kernelFuncMangling, "
                      "retRegister);\n";
        codeSource << "    }\n";
    }
    codeSource << "}\n";
    codeSource << "static const int32_t g_regiter_regfunc_ret = "
                  "AscPluginGenerator::BindKernelRegisterFunc(AscFunctionRegister);";
    ASC_LOGD("call device stub registry function is: %s", codeSource.str().c_str());
    return codeSource.str();
}
} // namespace AscPlugin
