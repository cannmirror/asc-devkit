/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "asc_compile_options.h"


class TEST_ASC_COMPILE_OPTIONS : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_COMPILE_OPTIONS, asc_CompileOptionManager)
{
    using namespace AscPlugin;
    CompileOptionManager mng = CompileOptionManager();
    auto deviceCubeExtraCompileOptions = mng.GetDeviceCompileOptions(CoreType::CUBE);
    auto deviceVecExtraCompileOptions = mng.GetDeviceCompileOptions(CoreType::VEC);
    auto hostExtraCompileOptions = mng.GetHostCompileOptions();
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT"
    };
    EXPECT_EQ(deviceCubeExtraCompileOptions, expDeviceCubeExtraCompileOptions);
    EXPECT_EQ(deviceVecExtraCompileOptions, expDeviceVecExtraCompileOptions);
    EXPECT_EQ(hostExtraCompileOptions, expHostExtraCompileOptions);
}

TEST_F(TEST_ASC_COMPILE_OPTIONS, asc_GetDeviceCompileOptionsWithSoc_91095_onttoone_onetotwo)
{
    using namespace AscPlugin;
    CompileOptionManager mng = CompileOptionManager();
    KernelTypeResult kernelTypeRes = {true, true, true, true};
    MOCKER(AscPlugin::CheckHasMixKernelFunc).stubs().will(returnValue(kernelTypeRes));
    auto cubeOpts = mng.GetDeviceCompileOptionsWithSoc<ShortSocVersion::ASCEND950>(CoreType::CUBE);
    auto vecOpts = mng.GetDeviceCompileOptionsWithSoc<ShortSocVersion::ASCEND950>(CoreType::VEC);
    EXPECT_TRUE(cubeOpts.empty());
    EXPECT_TRUE(vecOpts.empty());
}

TEST_F(TEST_ASC_COMPILE_OPTIONS, asc_GetDeviceCompileOptionsWithSoc_91095)
{
    using namespace AscPlugin;
    CompileOptionManager mng = CompileOptionManager();
    mng.l2CacheOn_ = true;
    KernelTypeResult kernelTypeRes = {true, false, false, false};
    MOCKER(AscPlugin::CheckHasMixKernelFunc).stubs().will(returnValue(kernelTypeRes));
    auto cubeOpts = mng.GetDeviceCompileOptionsWithSoc<ShortSocVersion::ASCEND950>(CoreType::CUBE);
    auto vecOpts = mng.GetDeviceCompileOptionsWithSoc<ShortSocVersion::ASCEND950>(CoreType::VEC);
    std::vector<std::string> expCubeOpts = { "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false", "-DL2_CACHE_HINT", "-D__MIX_CORE_MACRO__=1", "-D__MIX_CORE_AIC_RATION__=1" };
    std::vector<std::string> expVecOpts = { "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000", "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false", "-DL2_CACHE_HINT", "-D__MIX_CORE_MACRO__=1", "-D__MIX_CORE_AIC_RATION__=1" };
    EXPECT_EQ(cubeOpts, expCubeOpts);
    EXPECT_EQ(vecOpts, expVecOpts);
}