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
#include <fstream>
#include <cstdlib>
#include "ascc_utils.h"
#include "ascc_global_env_manager.h"
#include "ascc_argument_manager.h"
#include "ascc_dump_flags.h"
#include "ascc_device_stub.h"

class TEST_ASCC_ARGUMENT : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCC_ARGUMENT, ascc_ArchHandleTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"-arch", "Ascend910B5", "test.cpp"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_SanitizerHandleTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"-arch", "Ascend910B1", "test.cpp", "--sanitizer"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_DebugHandleTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"-arch", "Ascend910B1", "test.cpp", "--debug"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_OptimizeHandleTest)
{
    system("touch test.cpp");
    // optimize level only supports 0, 1, 2, 3
    std::vector<std::string> args = {"-arch", "Ascend910B1", "test.cpp", "--optimize=6"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);

    args = {"-arch", "Ascend910B1", "test.cpp", "--optimize=3"};
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);

    // optimize level only supports 0, 1, 2, 3
    args = {"-arch", "Ascend910B1", "test.cpp", "-O6"};
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);

    args = {"-arch", "Ascend910B1", "test.cpp", "-O3"};
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_filetype_Test)
{
    system("touch test.caaaa");
    system("touch test.asc");
    std::vector<std::string> args = {"test.asc"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
    args = {"test.caaaa"};
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);
    system("rm -rf test.asc");
    system("rm -rf test.caaaa");
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_DependencyTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"-MD", "-MT", "test.o", "-MF", "test.d", "test.cpp", "-Wl,rpath,a"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
    system("rm -rf test.cpp");
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_SaveTempsEmptyTest)
{
    system("touch test.cpp");
    system("touch ./tests");
    std::vector<std::string> args = {"-arch", "Ascend910B1", "--save-temps", "test.cpp"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_OutputHandleTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"-arch", "Ascend910B1","-o", "test.cpp"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_ArgumentTest)
{
    system("touch test.cpp");
    system("mkdir ./tmpfiles");
    system("touch ./add.o");
    system("touch ./add.so");
    std::vector<std::string> args = {"-arch", "Ascend910B1", "--npu-architecture=Ascend910B1",
        "--save-temps=./tmpfiles", "-o", "./test.cpp", "--output-file=./test.cpp",
        "-I./tmpfiles/a", "--include-path=./tmpfiles/a", "--library-path=./tmpfiles/a",
        "--library-path=./tmpfiles/a,./tmpfiles/b,./tmpfiles/c", "-ltprt", "-lparser_common",
        "--library", "data_flow_base", "-L./tmpfiles/a", "-L./tmpfiles/b", "--library=cann_kb",
        "--library=cce_aicore,datatransfer",
        "-shared", "-o", "add.so", "-c", "-o", "./add.o", "test.cpp","-DDEBUG"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
    system("rm -rf ./tmpfiles test.cpp nnn nn nnn add.so add.o");
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_ErrorArgumentTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"--save-temps", "./test_path", "-args"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_Error2ArgumentTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"-I"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_Error3ArgumentTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"test.cpp", "tests.cpp"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_Error4ArgumentTest)
{
    std::vector<std::string> args = {"tests.cpp"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_SaveTempsHandleTest)
{
    system("touch test.cpp");
    std::vector<std::string> args = {"-I./tmpfiles/a", "--include-path=./tmpfiles/a", "-ltprt",
        "--library-path=./tmpfiles/a", "-L./tmpfiles/b", "--save-temps="};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_SaveTempsHandle3Test)
{
    system("rm -rf test");
    system("touch test.cpp");
    std::vector<std::string> args = {"--save-temps=test", "test.cpp"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_ARGUMENT, ascc_ArgumentHelpTest)
{
    std::vector<std::string> args = {"--help"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    EXPECT_EQ(argManager.ArgumentParse(args), Ascc::AsccStatus::SUCCESS);
}
