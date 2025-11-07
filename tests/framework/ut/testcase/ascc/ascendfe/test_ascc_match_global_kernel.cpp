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
#include <set>
#define private public
#include <ascc_utils.h>
#include <ascc_match_global_kernel.h>
#include <ascc_info_callexpr.h>

class TEST_ASCC_MATCH_GLOBAL_KERNEL : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

namespace Ascc {
}

TEST_F(TEST_ASCC_MATCH_GLOBAL_KERNEL, ascc_IsFuncParamExist)
{
    Ascc::AsccMatchGlobalKernel kernel;
    std::string paramList = "(<{[]}>);";
    bool result = kernel.IsFuncParamExist(paramList);
    EXPECT_EQ(result, false);
    paramList = "(>;";
    result = kernel.IsFuncParamExist(paramList);
    EXPECT_EQ(result, false);
    paramList = "(int a);";
    result = kernel.IsFuncParamExist(paramList);
    EXPECT_EQ(result, true);
}

TEST_F(TEST_ASCC_MATCH_GLOBAL_KERNEL, ascc_MatchAndReplaceGlobalKernel)
{
    Ascc::AsccMatchGlobalKernel kernel;
    std::vector<std::pair<uint32_t, std::string>> callLines;
    std::vector<std::string> ctx1{"add_custom<float>", "<<<blockDim, nullptr, stream>>>(x);"};
    size_t start = 0;
    size_t cloumn = 1;
    Ascc::AsccStatus result = kernel.MatchAndReplaceGlobalKernel(callLines, ctx1, start, cloumn);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);

    std::vector<std::string> ctx2{"add_custom<float>(x);"};
    cloumn = 0;
    result = kernel.MatchAndReplaceGlobalKernel(callLines, ctx2, start, cloumn);
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);

    std::vector<std::string> ctx3{"add_custom<float><<<((blockDim))>>>(x);"};
    result = kernel.MatchAndReplaceGlobalKernel(callLines, ctx3, start, cloumn);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);

    std::vector<std::string> ctx4{"add_custom<float><<<blockDim>>>();"};
    cloumn = 0;
    result = kernel.MatchAndReplaceGlobalKernel(callLines, ctx4, start, cloumn);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_MATCH_GLOBAL_KERNEL, ascc_MatchAndGenerateGlobalKernel)
{
    MOCKER(&Ascc::AsccMangle::GetOriginToFixedMangledNames).stubs().will(returnValue(
        std::unordered_map<std::string, std::string>{{"hello_world", "hello_world"}}));
    Ascc::AsccInfoCallExpr expr1;
    expr1.manglingName = "add_custom";
    Ascc::AsccInfoCallExpr expr2;
    expr2.manglingName = "hello_world";
    MOCKER(&Ascc::AsccMatchGlobalInfo::GetGlobalKernelCallExpr).stubs().will(returnValue(
        std::unordered_map<std::string, Ascc::AsccInfoCallExpr>{
            {"add_custom:0", expr1},
            {"hello_world:0", expr2},
        }));
    MOCKER(&Ascc::AsccMatchGlobalKernel::MatchAndReplaceGlobalKernel)
        .stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccMatchGlobalKernel::ReadFileToVector)
        .stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    Ascc::AsccMatchGlobalKernel kernel;
    kernel.MatchAndGenerateGlobalKernel();
}

TEST_F(TEST_ASCC_MATCH_GLOBAL_KERNEL, ascc_ReadFileToVector_FileNotExist)
{
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(false));
    Ascc::AsccMatchGlobalKernel kernel;
    std::string fileName;
    std::vector<std::string> results;
    Ascc::AsccStatus result = kernel.ReadFileToVector(fileName, results);
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);
}