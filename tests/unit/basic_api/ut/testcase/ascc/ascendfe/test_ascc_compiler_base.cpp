/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <iostream>
#include <string>
#include <cstdio>
#include <sys/types.h>
#include <sys/wait.h>
#include <cerrno>
#include <sstream>
#include <memory>
#include <cstring>
#include <vector>

#define private public
#define protected public
#include "ascc_dump_flags.h"
#include "ascc_utils.h"
#include "ascc_compile_base.h"

using namespace Ascc;
using namespace mockcpp;

namespace Ascc {


class TestCompileBase : public AsccCompileBase {
public:
    explicit TestCompileBase(const CompileArgs& args) : AsccCompileBase(args) {}

    // 实现纯虚函数 Compile
    AsccStatus Compile() override {
        return AsccStatus::SUCCESS;
    }

    // 实现纯虚函数 MergeOption
    void MergeOption() override {
    }

};
}

// 测试用例类
class AsccCompileBaseTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        args_.file = "test.c";
        args_.outputPath = "test.o";
    }

    void TearDown()
    {
        GlobalMockObject::verify();
    }

    CompileArgs args_;
};

TEST_F(AsccCompileBaseTest, GetDependencyCmdTest) {
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.dependencyOptions_ = "";
    manager.mfmtRequested_ = true;
    manager.mfFile_ = "";
    manager.mtFile_ = "";
    manager.inputFile_ = "add_custom.cpp";
    manager.outputFile_ = "";
    args_.definitions = {"DEF1", "DEF2"};
    args_.incPaths = {"include1", "include2"};
    args_.options = {"-O2", "-Wall"};
    args_.incFiles = {"header1.h", "header2.h"};
    args_.linkFiles = {"lib1", "lib2"};
    args_.linkPath = {"libpath1", "libpath2"};

    TestCompileBase compileBase(args_);
    EXPECT_EQ(compileBase.GetDependencyCmd(), " -MF add_custom.d -MT add_custom.o");   // 没-o时，用的是.cpp替换成.o / .d
}

TEST_F(AsccCompileBaseTest, GetDependencyCmdTest2) {
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.dependencyOptions_ = "";
    manager.mfmtRequested_ = true;
    manager.mfFile_ = "";
    manager.mtFile_ = "";
    manager.inputFile_ = "add_custom.cpp";
    manager.outputFile_ = "out.ggg";
    args_.definitions = {"DEF1", "DEF2"};
    args_.incPaths = {"include1", "include2"};
    args_.options = {"-O2", "-Wall"};
    args_.incFiles = {"header1.h", "header2.h"};
    args_.linkFiles = {"lib1", "lib2"};
    args_.linkPath = {"libpath1", "libpath2"};

    TestCompileBase compileBase(args_);
    EXPECT_EQ(compileBase.GetDependencyCmd(), " -MF out.d -MT out.ggg");   // 没-o时，用的是-o作为MT，.d作为-MF
}


// 测试 GetCmd 函数
TEST_F(AsccCompileBaseTest, GetCmdTest) {
    // 测试基本命令生成
    args_.definitions = {"DEF1", "DEF2"};
    args_.incPaths = {"include1", "include2"};
    args_.options = {"-O2", "-Wall"};
    args_.incFiles = {"header1.h", "header2.h"};
    args_.linkFiles = {"lib1", "lib2"};
    args_.linkPath = {"libpath1", "libpath2"};

    TestCompileBase compileBase(args_);
    std::string cmd = args_.GetCmd("ascc");

    // 预期命令
    std::string expectedCmd = "ascc -O2 -Wall -c -DDEF1 -DDEF2 -Iinclude1 -Iinclude2 -llib1 -llib2 -Llibpath1 "
                              "-Llibpath2 -include header1.h -include header2.h -o test.o -- test.c ";

    EXPECT_EQ(cmd, expectedCmd);
}


// 测试 MergeCommonOption
TEST_F(AsccCompileBaseTest, MergeCommonOptionTest) {
    CompileArgs commonArgs;
    commonArgs.definitions = {"COMMON_DEF"};
    commonArgs.incPaths = {"common_include"};
    commonArgs.options = {"-common_option"};
    commonArgs.incFiles = {"common_header.h"};
    commonArgs.linkFiles = {"common_lib"};
    commonArgs.linkPath = {"common_libpath"};
    commonArgs.outputPath = "common_output.o";
    commonArgs.file = "common_file.c";
    commonArgs.customOption = "-common_custom";

    TestCompileBase compileBase(commonArgs);
    compileBase.MergeCommonOption(commonArgs);

    // 验证合并后的结果
    EXPECT_EQ(compileBase.args_.definitions.size(), 2);
    EXPECT_EQ(compileBase.args_.incPaths.size(), 2);
    EXPECT_EQ(compileBase.args_.options.size(), 2);
    EXPECT_EQ(compileBase.args_.incFiles.size(), 2);
    EXPECT_EQ(compileBase.args_.linkFiles.size(), 2);
    EXPECT_EQ(compileBase.args_.linkPath.size(), 2);
    EXPECT_EQ(compileBase.args_.outputPath, "common_output.o");
    EXPECT_EQ(compileBase.args_.file, "common_file.c");
    EXPECT_EQ(compileBase.args_.customOption, "-common_custom");
}

TEST_F(AsccCompileBaseTest, MergeCommonOptionErrorTest) {
    CompileArgs commonArgs;
    commonArgs.definitions = {"COMMON_DEF"};
    commonArgs.incPaths = {"common_include"};
    commonArgs.options = {"-common_option"};
    commonArgs.incFiles = {"common_header.h"};
    commonArgs.linkFiles = {"common_lib"};
    commonArgs.linkPath = {"common_libpath"};
    commonArgs.outputPath = "";
    commonArgs.file = "";
    commonArgs.customOption = "";

    TestCompileBase compileBase(commonArgs);
    compileBase.MergeCommonOption(commonArgs);

    // 验证合并后的结果
    EXPECT_EQ(compileBase.args_.definitions.size(), 2);
    EXPECT_EQ(compileBase.args_.incPaths.size(), 2);
    EXPECT_EQ(compileBase.args_.options.size(), 2);
    EXPECT_EQ(compileBase.args_.incFiles.size(), 2);
    EXPECT_EQ(compileBase.args_.linkFiles.size(), 2);
    EXPECT_EQ(compileBase.args_.linkPath.size(), 2);
    EXPECT_EQ(compileBase.args_.outputPath, "");
    EXPECT_EQ(compileBase.args_.file, "");
    EXPECT_EQ(compileBase.args_.customOption, "");
}

// 测试 AddOption 和 AddDefinition
TEST_F(AsccCompileBaseTest, AddOptionAndDefinitionTest) {
    TestCompileBase compileBase(args_);
    compileBase.AddOption("-O3");
    compileBase.AddDefinition("TEST_DEF");

    EXPECT_EQ(compileBase.args_.options.size(), 1);
    EXPECT_EQ(compileBase.args_.definitions.size(), 1);
    EXPECT_EQ(compileBase.args_.options[0], "-O3");
    EXPECT_EQ(compileBase.args_.definitions[0], "TEST_DEF");
}

// 测试 AddIncPath 和 AddIncFile
TEST_F(AsccCompileBaseTest, AddIncPathAndIncFileTest) {
    TestCompileBase compileBase(args_);
    compileBase.AddIncPath("test_include");
    compileBase.AddIncFile("test_header.h");

    EXPECT_EQ(compileBase.args_.incPaths.size(), 1);
    EXPECT_EQ(compileBase.args_.incFiles.size(), 1);
    EXPECT_EQ(compileBase.args_.incPaths[0], "test_include");
    EXPECT_EQ(compileBase.args_.incFiles[0], "test_header.h");
}

// 测试 SetCustomOption
TEST_F(AsccCompileBaseTest, SetCustomOptionTest) {
    TestCompileBase compileBase(args_);
    compileBase.SetCustomOption("-custom_option");

    EXPECT_EQ(compileBase.args_.customOption, "-custom_option");
}