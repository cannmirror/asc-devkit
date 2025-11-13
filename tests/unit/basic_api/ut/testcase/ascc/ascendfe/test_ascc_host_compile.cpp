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
#include <unistd.h>
#include <mockcpp/mockcpp.hpp>
#include <string>
#include <vector>
#define private public
#include "ascc_host_compile.h"

class TEST_ASCC_HOST_COMPILE : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCC_HOST_COMPILE, ascc_HostCompileHostOnly)
{
    Ascc::AsccHostCompile asccHostCompile;
    MOCKER(&Ascc::AsccMatchGlobalInfo::HasKernelCall).stubs().will(returnValue(false));
    EXPECT_NO_THROW(asccHostCompile.HostCompile());
}

TEST_F(TEST_ASCC_HOST_COMPILE, ascc_HostCompileDeviceNoCpp)
{
    Ascc::AsccHostCompile asccHostCompile;
    MOCKER(&Ascc::AsccMatchGlobalInfo::HasKernelCall).stubs().will(returnValue(true));
    MOCKER(&Ascc::CheckAndGetFullPath, std::string(const std::string&)).stubs().will(returnValue(std::string("/tmp")));
    EXPECT_NO_THROW(asccHostCompile.HostCompile());
}

TEST_F(TEST_ASCC_HOST_COMPILE, ascc_HostCompileDeviceNoCpp_host_pretask)
{
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.preTaskType_ = Ascc::PreTaskType::HOST;
    Ascc::AsccHostCompile asccHostCompile;
    MOCKER(&Ascc::AsccMatchGlobalInfo::HasKernelCall).stubs().will(returnValue(true));
    MOCKER(&Ascc::CheckAndGetFullPath, std::string(const std::string&)).stubs().will(returnValue(std::string("/tmp")));
    EXPECT_NO_THROW(asccHostCompile.HostCompile());

    manager.preTaskType_ = Ascc::PreTaskType::NONE;
}


TEST_F(TEST_ASCC_HOST_COMPILE, ascc_ProcessKernelCallLines)
{
    Ascc::AsccHostCompile asccHostCompile;
    std::vector<std::pair<uint32_t, std::string>> kernelCallLines = {{0, "0"}};
    std::vector<std::string> lines = {"0"};
    asccHostCompile.ProcessKernelCallLines(kernelCallLines, lines);
}

TEST_F(TEST_ASCC_HOST_COMPILE, ascc_ShieldDeviceCode_PathNotExist)
{
    Ascc::AsccHostCompile asccHostCompile;
    std::string inputFile =
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    std::string outputFile =
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/shield_device_code_output.txt";
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    std::vector<std::pair<uint32_t, std::string>> kernelCallLines;
    std::vector<std::pair<uint32_t, std::string>> kernelDefLines;
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true)).then(returnValue(false));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true)).then(returnValue(false));
    asccHostCompile.ShieldDeviceCode(inputFile, outputFile, ranges, kernelCallLines, kernelDefLines);
    system("rm -f ./llt/atc/tikcpp/ut/testcase/ascc/hello_world/shield_device_code_output.txt");
}

TEST_F(TEST_ASCC_HOST_COMPILE, ascc_ShieldDeviceCode)
{
    Ascc::AsccHostCompile asccHostCompile;
    std::string inputFile =
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    std::string outputFile =
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/shield_device_code_output.txt";
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    std::vector<std::pair<uint32_t, std::string>> kernelCallLines;
    std::vector<std::pair<uint32_t, std::string>> kernelDefLines;
    Ascc::PathStatus res = Ascc::PathCheck(inputFile.c_str(), true);
    asccHostCompile.ShieldDeviceCode(inputFile, outputFile, ranges, kernelCallLines, kernelDefLines);
    system("rm -f ./llt/atc/tikcpp/ut/testcase/ascc/hello_world/shield_device_code_output.txt");
}

TEST_F(TEST_ASCC_HOST_COMPILE, ascc_ProcessKernelFuncLines)
{
    Ascc::AsccHostCompile asccHostCompile;
    std::vector<std::pair<uint32_t, uint32_t>> ranges = {{0, 0}};
    std::vector<std::string> lines;
    asccHostCompile.ProcessKernelFuncLines(ranges, lines);
}