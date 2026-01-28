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
 * \file asc_dev_func_registry_generate.cpp
 * \brief
 */

#include "asc_dev_func_registry_generate.h"
#include "asc_log.h"

#include <sstream>

namespace AscPlugin {
constexpr size_t FUNCREG_SIZE_CODE_BUFFER_LEN = 16 * 1024;
constexpr const char* FUNC_REGISTER_CODE = R"(#include <stdio.h>
#include <cstdint>
extern "C" {
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
}
namespace Adx {
void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
    void *stream, const char *opType);
} // namespace Adx
namespace AscPluginGenerator {
int32_t BindKernelRegisterFunc(void (*)(void*));
uint32_t LaunchAndProfiling(const char *kernelName, uint32_t blockDim, void *stream, void **args, uint32_t size,
                            uint32_t ktype, const uint32_t ubufDynamicSize);
uint32_t ascendc_set_exception_dump_info(uint32_t dumpSize);
} // namespace AscPluginGenerator
static const int32_t g_ascend_plugin_register = AscPluginGenerator::BindKernelRegisterFunc(nullptr);

#define ASC_PLUGIN_LAUNCH_LOGE(kernelName, stream, blockDim, fmt, ...)                      \
    ::printf("[ERROR] [AscPlugin] Kernel: [%s] Stream: [%p] BlockDim: [%u] " fmt "\n",      \
        kernelName,                                                                         \
        stream,                                                                             \
        blockDim,                                                                           \
        ##__VA_ARGS__)
)";

constexpr const char *KERNEL_BINARY_VERSION_SECTION = R"(
static const struct BinaryMetaVersion __ascendc_binary_version__ __attribute__ ((used, section (".ascend.meta"))) =
    {0, 4, 1};
)";

std::string FunctionRegistryImpl()
{
    std::stringstream codeSource;
    std::string buffer;
    buffer.reserve(FUNCREG_SIZE_CODE_BUFFER_LEN);
    codeSource.str(std::move(buffer));
    codeSource << FUNC_REGISTER_CODE;
    if (InfoManager::GetInstance().HasKernelFunc()) {
        codeSource << KERNEL_BINARY_VERSION_SECTION;
    }
    ASC_LOGD("call device stub registry function is: %s", codeSource.str().c_str());
    return codeSource.str();
}
} // namespace AscPlugin
