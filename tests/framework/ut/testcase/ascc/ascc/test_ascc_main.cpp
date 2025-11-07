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
#include <mockcpp/mockcpp.hpp>
#define private public
#include "ascc_option.h"
#include "ascc_task_manager.h"

using namespace testing;
using namespace Ascc;

std::stringstream buffer;
std::streambuf* oriBuff = nullptr;
std::stringstream errBuffer;
std::streambuf* oriErrBuff = nullptr;
class TEST_ASCC_MAIN : public testing::Test {
protected:
    void SetUp()
    {
        MOCKER(PrinterExit).stubs();
        oriBuff = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
        oriErrBuff = std::cerr.rdbuf();
        std::cerr.rdbuf(errBuffer.rdbuf());
    }

    void TearDown()
    {
        if (oriBuff != nullptr) {
            std::cout.rdbuf(oriBuff);
            std::cout<< buffer.str()<<std::endl;
            buffer.str("");
        }

        if (oriErrBuff != nullptr) {
            std::cerr.rdbuf(oriErrBuff);
            std::cerr<< errBuffer.str()<<std::endl;
            errBuffer.str("");
        }

        GlobalMockObject::verify();
    }
};

void SetUpCommonMocker()
{
    ClearUpOptionValue();   // 单例状态刷新
    char* ascHomePath = "";
    std::string pathA = "a";
    MOCKER(getenv).stubs().will(returnValue(ascHomePath));
    MOCKER(Ascc::FindExecPath).stubs().will(returnValue(pathA));
}

void MockerProcessFileProcess()
{
    MOCKER(Ascc::ProcessFiles).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
}

extern int EntryAsccMain(int argc, char *argv[]);


// ==============================================================================
// 异常分支
// 验证走进fileSize = 0 异常分支
TEST_F(TEST_ASCC_MAIN, ascc_main_filesize_failure)
{
    SetUpCommonMocker();
    int argc = 2;
    char *argv[] = {(char *)"ascc", (char *)"-arch=Ascend910B1"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}


// 验证走进InitArgInfo异常分支
TEST_F(TEST_ASCC_MAIN, ascc_main_InitArgInfo_failure)
{
    SetUpCommonMocker();
    MOCKER(Ascc::InitArgInfo).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    int argc = 3;
    char *argv[] = {(char *)"ascc", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}

// 验证走进InitPathInfo异常分支
TEST_F(TEST_ASCC_MAIN, ascc_main_InitPathInfo_failure)
{
    SetUpCommonMocker();
    MOCKER(Ascc::InitPathInfo).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    int argc = 3;
    char *argv[] = {(char *)"ascc", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}

// 验证走进CreateTmpDirectory异常分支
TEST_F(TEST_ASCC_MAIN, ascc_main_CreateTmpDirectory_failure)
{
    SetUpCommonMocker();
    MOCKER(Ascc::CreateTmpDirectory).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    int argc = 3;
    char *argv[] = {(char *)"ascc", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}

// 验证走进ProcessFiles异常分支
TEST_F(TEST_ASCC_MAIN, ascc_main_ProcessFiles_failure)
{
    SetUpCommonMocker();
    MOCKER(Ascc::ProcessFiles).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    int argc = 2;
    char *argv[] = {(char *)"ascc", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}

// ====================================================================
TEST_F(TEST_ASCC_MAIN, ascc_main_with_help)
{
    int argc = 2;
    char *argv[] = {(char *)"ascc",
        (char *)"-h"};
    int result = EntryAsccMain(argc, argv);
    std::string expect = "Usage";
    EXPECT_NE(buffer.str().find(expect), std::string::npos);
}

TEST_F(TEST_ASCC_MAIN, ascc_main_unknown_arg)
{
    int argc = 3;
    char *argv[] = {(char *)"ascc",
        (char *)"-arg", (char *)"-arch=Ascend910B1"};
    int result = EntryAsccMain(argc, argv);
    std::string expect = "Unknown";
    EXPECT_NE(errBuffer.str().find(expect), std::string::npos);
    EXPECT_EQ(result, 1);
}

TEST_F(TEST_ASCC_MAIN, ascc_main_arch_requried)
{
    SetUpCommonMocker();
    int argc = 2;
    char *argv[] = {(char *)"ascc",
        (char *)"--output-file=dir.o"};
    int result = EntryAsccMain(argc, argv);
    std::string expect = "Option --npu-architecture / -arch is required";
    EXPECT_NE(errBuffer.str().find(expect), std::string::npos);
    EXPECT_EQ(result, 1);
}

TEST_F(TEST_ASCC_MAIN, ascc_main_library)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 5;
    char *argv[] = {(char *)"ascc",
        (char *)"-la", (char *)"-l=linc", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_output)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 5;
    char *argv[] = {(char *)"ascc",
        (char *)"--output-file", (char *)"arg.o", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_debug)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"-g", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_debug_error)
{
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"-g=false", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}

// 在args中反复添加debug g的选项，args manager会拦截这种行为
TEST_F(TEST_ASCC_MAIN, ascc_main_debug_repeated_add_error)
{
    static Ascc::Opt<bool> debugOptTest("debug", Ascc::ShortDesc("g"),
        Ascc::ArgOccNumFlag::REQUIRED, Ascc::HelpDesc("debug test"));
    int argc = 3;
    char *argv[] = {(char *)"ascc",
        (char *)"--debug=false", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}

TEST_F(TEST_ASCC_MAIN, ascc_main_option_optional_save_temp)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"-save-temps", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm -rf /tmp/bishengcc");
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_option_optiona_save_temp_dir)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("mkdir test");
    system("touch add_custom.cpp");
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"-save-temps=test", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm -rf test");
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_option_positional_input_files)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    int argc = 3;
    system("touch add_custom.cpp");
    char *argv[] = {(char *)"ascc",
        (char *)"add_custom.cpp", (char *)"-arch=Ascend910B1"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

// 测试 - 选项是否正常报错
TEST_F(TEST_ASCC_MAIN, ascc_main_option_dash)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    int argc = 3;
    char *argv[] = {(char *)"ascc",
        (char *)"-", (char *)"-arch=Ascend910B1"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}

TEST_F(TEST_ASCC_MAIN, ascc_main_option_include_path)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"-I/usr/,/home", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_option_include_path_error)
{
    SetUpCommonMocker();
    system("touch add_custom.cpp");
    int argc = 4;
    char *argv[] = {(char *)"bishengcc",
        (char *)"-I=", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);

    SetUpCommonMocker();
    argc = 4;
    char *argv1[] = {(char *)"bishengcc",
        (char *)"-I", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    result = EntryAsccMain(argc, argv1);
    EXPECT_EQ(result, 1);
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_option_short_desc)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    static Ascc::Opt<uint32_t> portOptTest( Ascc::ShortDesc("port"), Ascc::HasArgFlag::REQUIRED,
        Ascc::HelpDesc("port test"));
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"-I/usr/,/home", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);

    argc = 4;
    char *argv1[] = {(char *)"ascc",
        (char *)"-port=xxxx", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    result = EntryAsccMain(argc, argv1);
    EXPECT_EQ(result, 1);

    argc = 5;
    char *argv2[] = {(char *)"ascc",
        (char *)"-port", (char *)"8080", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    result = EntryAsccMain(argc, argv2);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(portOptTest.GetValue(), 8080);
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_option_init)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    static Ascc::Opt<std::string> initOptTest("init", Ascc::HasArgFlag::REQUIRED,
        Ascc::HelpDesc("port test"), Ascc::Init(std::string("init")));
    static Ascc::Opt<uint32_t> ageOptTest("age", Ascc::HasArgFlag::REQUIRED,
        Ascc::HelpDesc("age test"), Ascc::Init(10));
    static std::vector<uint32_t> x = {10, 11, 12, 13};
    static Ascc::OptList<uint32_t> socreOptTest("socre", Ascc::HasArgFlag::REQUIRED,
        Ascc::HelpDesc("sore test"), Ascc::ListInit(x));
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"-I/usr/,/home", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(initOptTest.GetValue(), "init");
    EXPECT_EQ(ageOptTest.GetValue(), 10);

    uint32_t i = 10;
    for (auto &sore : socreOptTest) {
        EXPECT_EQ(sore, i++);
    }

    argc = 4;
    char *argv2[] = {(char *)"ascc",
        (char *)"--init", (char *)"8080", (char *)"-arch=Ascend910B1"};
    result = EntryAsccMain(argc, argv2);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(initOptTest.GetValue(), "8080");
    system("rm add_custom.cpp");
}

// --verbose --time
TEST_F(TEST_ASCC_MAIN, ascc_main_verbose_time)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 5;
    char *argv[] = {(char *)"ascc",
        (char *)"--verbose", (char *)"--time", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

// -c
TEST_F(TEST_ASCC_MAIN, ascc_main_out_objfile)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 5;
    char *argv[] = {(char *)"ascc",
        (char *)"--verbose", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp", "-c"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

// multiple files with -c + -o
TEST_F(TEST_ASCC_MAIN, ascc_main_multiple_file_with_ofile_outarg)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    system("touch sub_custom.cpp");
    int argc = 8;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1",
        (char*)"add_custom.cpp", (char*)"sub_custom.cpp", (char*)"-c", (char*)"-o", (char*)"fff"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);

    std::string expect = "cannot specify '-o' with '-c' with multiple files";
    EXPECT_NE(errBuffer.str().find(expect), std::string::npos);
    EXPECT_EQ(result, 1);
    system("rm add_custom.cpp");
}

// -shared
TEST_F(TEST_ASCC_MAIN, ascc_main_out_shared_library)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 6;
    char *argv[] = {(char *)"ascc", (char *)"--verbose",
        (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp", (char*)"-shared"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

// --optimize
TEST_F(TEST_ASCC_MAIN, ascc_main_optimize_level)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 6;
    char *argv[] = {(char *)"ascc", (char *)"--verbose",
        (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp", (char*)"--optimize=1"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}

// --save-temps
TEST_F(TEST_ASCC_MAIN, ascc_main_save_temps)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    system("mkdir -p /tmp/ascc_test/");
    int argc = 5;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1",
        (char*)"add_custom.cpp", (char*)"--save-temps=/tmp/ascc_test/"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm -rf /tmp/ascc_test/");
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_MAIN, ascc_main_save_temps_path_not_exist)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    int argc = 5;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1",
        (char*)"add_custom.cpp", (char*)"--save-temps=test/dir/test"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
}

TEST_F(TEST_ASCC_MAIN, ascc_main_ErrorMsgForProcessFile)
{
    SetUpCommonMocker();
    system("touch add_custom.cpp");
    system("mkdir -p /tmp/ascc_test/");
    int argc = 6;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1",
        (char*)"add_custom.cpp", (char*)"--save-temps=/tmp/ascc_test/", (char*)"--verbose"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 1);
    system("rm -rf /tmp/ascc_test/");
    system("rm add_custom.cpp");
}

// when user passed ASCENDC_DUMP
TEST_F(TEST_ASCC_MAIN, ascc_main_ascendc_dump)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"-DASCENDC_DUMP", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");
}
