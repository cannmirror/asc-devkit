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
#include <string>
#define private public
#include "ascc_types.h"
#include "ascc_utils.h"
#include "ascc_ast_analyzer.h"
#include "ascc_ast_info_collector.h"
#include "ascc_ast_device_analyzer.h"
#include "ascc_ast_device_consumer.h"
#include "ascc_argument_manager.h"

class TEST_ASCC_AST : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCC_AST, ascc_AstAnalyzer_Success)
{
    system("touch test.cpp");
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    argManager.inputFile_ = "test.cpp";
    std::vector<std::string> astFiles = {"test.cpp"};
    Ascc::AsccAstAnalyzer analyzer(argManager.inputFile_);
    std::string stubPath = "/tmp";
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const std::string&)).stubs().will(returnValue(stubPath));
    EXPECT_EQ(analyzer.Process(), Ascc::AsccStatus::SUCCESS);
    system("rm -rf test.cpp");
}

TEST_F(TEST_ASCC_AST, ascc_AstAnalyzer_Failure)
{
    system("touch test.cpp");
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    argManager.inputFile_ = "test.cpp";
    std::vector<std::string> astFiles = {"test.cpp"};
    Ascc::AsccAstAnalyzer analyzer(argManager.inputFile_);
    std::string stubPath = "/tmp";
    std::vector<const char*> stubOptions = {"ABCDEFG", "1234567"};
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const std::string&)).stubs().will(returnValue(stubPath));
    MOCKER(Ascc::ConvertStringVecToCStringVec).stubs().will(returnValue(stubOptions.data()));
    EXPECT_EQ(analyzer.Process(), Ascc::AsccStatus::FAILURE);
    system("rm -rf test.cpp");
}

TEST_F(TEST_ASCC_AST, ascc_AstAnalyzerNpuError)
{
    system("touch test.cpp");
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    std::string src = "test.cpp";
    argManager.inputFile_ = src;
    std::vector<std::string> astFiles = {src};
    std::vector<std::string> options = {"-DABC"};
    Ascc::AsccAstAnalyzer analyzer(src);
    std::string stubPath = "/tmp";
    argManager.npuArch_ = Ascc::ShortSoCVersion::INVALID_TYPE;
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const std::string&)).stubs().will(returnValue(stubPath));
    EXPECT_ANY_THROW(analyzer.InitCompileArgs(argManager.inputFile_));
    argManager.npuArch_ = Ascc::ShortSoCVersion::ASCEND910B;
    system("rm -rf test.cpp");
}