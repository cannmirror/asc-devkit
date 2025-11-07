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
#include "ascc_dump_flags.h"
#include "ascc_utils.h"
#include "ascc_compile_base.h"
#include "ascc_compile_host.h"

using namespace Ascc;
using namespace mockcpp;

namespace Ascc {
class TestCompileHost : public AsccCompileHost {
public:
    explicit TestCompileHost(const CompileArgs& args) : AsccCompileHost(args) {}
};
}

// 测试用例类
class AsccCompileHostTest : public ::testing::Test {
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

// 测试 AsccCompileHost的MergeOption
TEST_F(AsccCompileHostTest, MergeOptionTest) {
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

    TestCompileHost compileBase(commonArgs);
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    argManager.optimizeLevel_ = "O1";
    compileBase.MergeOption();
    auto testArgs = compileBase.args_.options;
    // when -O1 is passed, device is O2
    auto find_res = std::find(testArgs.begin(), testArgs.end(), "-O1") != testArgs.end();
    EXPECT_EQ(find_res, false);
    find_res = std::find(testArgs.begin(), testArgs.end(), "-O2") != testArgs.end();
    EXPECT_EQ(find_res, true);

    argManager.optimizeLevel_ = "O3";
    compileBase.MergeOption();
    testArgs = compileBase.args_.options;
    find_res = std::find(testArgs.begin(), testArgs.end(), "-O3") != testArgs.end();
    EXPECT_EQ(find_res, true);
}

