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
#include <cstdlib>
#include <cstring>
#include <cstdio>
#define private public
#include <sys/types.h>
#include "ascc_host_stub.h"
#include "ascc_log.h"
#include "ascc_argument_manager.h"
#include "ascc_device_stub.h"
#include "ascc_link.h"
#include "ascc_tmp_file_manager.h"
#include "ascc_host_compile.h"
#include "ascc_types.h"
#include "ascc_ast_analyzer.h"
#include "ascc_ast_device_analyzer.h"
#include "ascc_utils.h"

using namespace testing;
using namespace Ascc;

class TEST_ASCGEN_MAIN : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

extern int EntryAscGenMain(int argc, char *argv[]);

TEST_F(TEST_ASCGEN_MAIN, ascc_main_total)
{
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"hello_world.cpp",
        (char *)"-arch",
        (char *)"Ascend910B1"};

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.initSuccess = true;
    envVar.needSaveTmpFile = false;
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.helpRequested_ = false;

    MOCKER(&AsccAstAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstDeviceAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccArgumentManager::ArgumentParse).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccTmpFileManager::Init).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccLink::AscendLink).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccMatchGlobalInfo::IsCalled).stubs().will(returnValue(false));
    MOCKER(PathCheck).stubs().will(returnValue(Ascc::PathStatus::NOT_EXIST));
    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 0);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}

TEST_F(TEST_ASCGEN_MAIN, ascc_main_total_empty)
{
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"hello_world.cpp",
        (char *)"-arch",
        (char *)"Ascend910B1"};

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.initSuccess = true;
    envVar.needSaveTmpFile = false;
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.helpRequested_ = false;

    MOCKER(&AsccAstAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstDeviceAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccArgumentManager::ArgumentParse).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccTmpFileManager::Init).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccLink::AscendLink).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(PathCheck).stubs().will(returnValue(Ascc::PathStatus::NOT_EXIST));
    std::unordered_map<std::string, std::unordered_map<AscCursorTypes, std::shared_ptr<AsccInfoBase>>> fileInfos;
    MOCKER(&AsccInfoStorage::GetAllInfos).stubs().will(returnValue(fileInfos));
    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 0);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}


// remove tmp files when needSaveTmpFile = false
TEST_F(TEST_ASCGEN_MAIN, ascc_main_total_rm_tmf_files)
{
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"hello_world.cpp",
        (char *)"-arch",
        (char *)"Ascend910B1"};

    system("mkdir tmp_files" );

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.initSuccess = true;
    envVar.needSaveTmpFile = false;
    envVar.asccTmpPath = "tmp_files";
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.helpRequested_ = false;

    MOCKER(&AsccAstAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstDeviceAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccArgumentManager::ArgumentParse).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccTmpFileManager::Init).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccLink::AscendLink).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 0);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}

TEST_F(TEST_ASCGEN_MAIN, ascc_main_with_help)
{
    int argc = 2;
    char *argv[] = {(char *)"ascc",
        (char *)"--help"};
    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 0);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}

TEST_F(TEST_ASCGEN_MAIN, ascc_main_with_pre_device)
{

    int argc = 6;
    char *argv[] = {(char *)"ascc",
        (char *)"../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        (char *)"-arch",
        (char *)"Ascend910B1",
        (char *)"--sub_module=device_pre_vec",
        (char *)"--module_path=../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world"};

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.initSuccess = true;
    envVar.needSaveTmpFile = false;
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.helpRequested_ = false;
    manager.parserQueue_.erase("--save-temps");

    MOCKER(&AsccTmpFileManager::Init).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstDeviceAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::GenerateJsonFiles).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));

    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 0);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}

TEST_F(TEST_ASCGEN_MAIN, ascc_main_with_pre_device_cube)
{

    int argc = 6;
    char *argv[] = {(char *)"ascc",
        (char *)"../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        (char *)"-arch",
        (char *)"Ascend910B1",
        (char *)"--sub_module=device_pre_cube",
        (char *)"--module_path=../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world"};

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.initSuccess = true;
    envVar.needSaveTmpFile = false;
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.helpRequested_ = false;
    manager.parserQueue_.erase("--save-temps");

    MOCKER(&AsccTmpFileManager::Init).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstDeviceAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::GenerateJsonFiles).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));

    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 0);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}

TEST_F(TEST_ASCGEN_MAIN, ascc_main_with_pre_device_error)
{

    int argc = 6;
    char *argv[] = {(char *)"ascc",
        (char *)"../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        (char *)"-arch",
        (char *)"Ascend910B1",
        (char *)"--sub_module=device_pre_error",
        (char *)"--module_path=../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world"};

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.initSuccess = true;
    envVar.needSaveTmpFile = false;
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.helpRequested_ = false;
    manager.parserQueue_.erase("--save-temps");

    MOCKER(&AsccTmpFileManager::Init).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstDeviceAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::GenerateJsonFiles).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));

    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 1);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}

TEST_F(TEST_ASCGEN_MAIN, ascc_main_with_pre_host)
{
    int argc = 6;
    char *argv[] = {(char *)"ascc",
        (char *)"../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        (char *)"-arch",
        (char *)"Ascend910B1",
        (char *)"--sub_module=host_pre",
        (char *)"--module_path=../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world"};

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.initSuccess = true;
    envVar.needSaveTmpFile = false;
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.helpRequested_ = false;
    manager.parserQueue_.erase("--save-temps");

    MOCKER(&AsccTmpFileManager::Init).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstDeviceAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::GenerateJsonFiles).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccLink::AscendLink).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));

    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 0);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}

TEST_F(TEST_ASCGEN_MAIN, ascc_main_total_GenerateJsonObj)
{
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"hello_world.cpp",
        (char *)"-arch",
        (char *)"Ascend910B1"};

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.initSuccess = true;
    envVar.needSaveTmpFile = false;
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.helpRequested_ = false;

    MOCKER(&AsccAstAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccAstDeviceAnalyzer::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccArgumentManager::ArgumentParse).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccTmpFileManager::Init).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccDeviceStub::Process).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&AsccLink::AscendLink).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccInfoFunction::FunctionInfo::IsTempDecl).stubs().will(returnValue(true));
    MOCKER(PathCheck).stubs().will(returnValue(Ascc::PathStatus::NOT_EXIST));
    int result = EntryAscGenMain(argc, argv);
    EXPECT_EQ(result, 0);
    if (std::remove("hello_world.cpp") == 0) {
        std::cout << "File Delete Successfully" << std::endl;
    } else {
        std::cerr << "File Delete Failed" << std::endl;
    }
}