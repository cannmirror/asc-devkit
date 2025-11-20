/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "asc_dev_funcRegistry_generate.h"

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "asc_log.h"

#include <sstream>
class TEST_ASC_DEV_FUNCREGISTRY_GENERATE : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_DEV_FUNCREGISTRY_GENERATE, asc_FunctionRegistryImpl)
{
    AscPlugin::InfoManager::GetInstance().kernelFuncSymbolToFuncInfo_.insert({"__device_stub__test_mangled",
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "test.cpp", 2, 2, AscPlugin::KfcScene::Close}});
    std::string res = AscPlugin::FunctionRegistryImpl();
    std::string expectRes = R"(#include <stdio.h>
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

static void AscFunctionRegister(void* g_kernel_handle)
{
    int32_t retRegister = 0;
    const char *kernelFuncMangling = nullptr;
    kernelFuncMangling = "test_mangled";
    retRegister = AscendFunctionRegister(g_kernel_handle, kernelFuncMangling);
    if (retRegister != 0) {
        ::printf("[ERROR] [AscPlugin] Kernel [%s] : function register failure! ret %d\n", kernelFuncMangling, retRegister);
    }
}
static const int32_t g_regiter_regfunc_ret = AscPluginGenerator::BindKernelRegisterFunc(AscFunctionRegister);)";
    EXPECT_EQ(res, expectRes);
}