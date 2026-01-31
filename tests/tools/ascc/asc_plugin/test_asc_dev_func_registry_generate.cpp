/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "asc_dev_func_registry_generate.h"

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "asc_log.h"

#include <sstream>
class TestAscDevFuncRegistryGenerate : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestAscDevFuncRegistryGenerate, asc_FunctionRegistryImpl)
{
    AscPlugin::InfoManager::GetInstance().kernelFuncSymbolToFuncInfo_.insert({"__device_stub__test_mangled",
        {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "test.cpp", 2, 2, AscPlugin::KfcScene::Close}});
    std::string res = AscPlugin::FunctionRegistryImpl();
    std::string expectRes = R"(#include <stdio.h>
#include <cstdint>
namespace AscPluginGenerator {
int32_t BindKernelRegisterFunc(void (*)(void*));
uint32_t LaunchAndProfiling(const char *kernelName, uint32_t blockDim, void *stream, void **args, uint32_t size,
                            uint32_t ktype, const uint32_t ubufDynamicSize);
} // namespace AscPluginGenerator
static const int32_t g_ascend_plugin_register = AscPluginGenerator::BindKernelRegisterFunc(nullptr);
#define ASC_PLUGIN_LAUNCH_LOGE(kernelName, stream, blockDim, fmt, ...)                      \
    ::printf("[ERROR] [AscPlugin] Kernel: [%s] Stream: [%p] BlockDim: [%u] " fmt "\n",      \
        kernelName,                                                                         \
        stream,                                                                             \
        blockDim,                                                                           \
        ##__VA_ARGS__)
)";
    EXPECT_EQ(res, expectRes);
}