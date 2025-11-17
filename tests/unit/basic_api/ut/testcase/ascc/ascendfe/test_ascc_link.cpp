/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
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

#define private public
#include <gmock/gmock.h>

#include "ascc_log.h"
#include "ascc_host_stub.h"
#include "ascc_compile_base.h"
#include "ascc_types.h"
#include "ascc_compile_factory.h"
#include "ascc_link.h"
#include "ascc_argument_manager.h"
#include "ascc_global_env_manager.h"
#include "ascc_tmp_file_manager.h"

class TEST_ASCC_LINK : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

extern int EntryAsccMain(int argc, char *argv[]);

namespace Ascc {
    extern bool FileExists(const std::string& path);
    extern AsccStatus UpdateHostStubByDevice(const std::string& hostStubPath);
}

// when isKernelFuncFound = True
TEST_F(TEST_ASCC_LINK, ascc_AscendLink_with_kernel_func)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    // envVar.asccTmpIncludePath = envVar.asccTmpPath + "/include";
    // envVar.asccTmpAutoGenPath = envVar.asccTmpPath + "/auto_gen";
    // envVar.asccTmpHostGenPath = envVar.asccTmpPath + "/auto_gen/host_files";;
    // envVar.asccTmpDependPath = envVar.asccTmpPath + "/dependence";
    // envVar.asccMergeObjPath = envVar.asccTmpPath + "/link_files/merge_obj";
    // envVar.asccMergeObjFinalPath = envVar.asccTmpPath + "/link_files/merge_obj_final";
    // envVar.asccCompileLogPath = envVar.asccTmpPath + "/compile_log";
    system("mkdir -p /tmp/ascc/auto_gen");
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    std::string inputFileName = "sample_cpp/add_custom.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;

    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.compileMode_ = Ascc::OutputFileType::FILE_EXECUTABLE;
    manager.outputFile_ = "";

    int32_t returnCode = 0;
    std::string a = "DEFAULT";
    MOCKER(&Ascc::AsccLink::IsKernelFuncFound).stubs().will(returnValue(true));
    MOCKER(&Ascc::AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccHostStub::UpdateHostStubByDevice).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true));

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::SUCCESS);
    system("rm -rf /tmp/ascc");
}

// when isKernelFuncFound = false
TEST_F(TEST_ASCC_LINK, ascc_AscendLink_without_kernel_func)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    system("mkdir -p /tmp/ascc/auto_gen");
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    std::string inputFileName = "sample_cpp/add_custom.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;

    int32_t returnCode = 0;
    std::string a = "DEFAULT";
    MOCKER(Ascc::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::FAILURE);
    system("rm -rf /tmp/ascc");
}

// when output is .so
TEST_F(TEST_ASCC_LINK, ascc_AscendLink_output_so)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    system("mkdir -p /tmp/ascc/auto_gen");
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    std::string inputFileName = "sample_cpp/add_custom.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;

    int32_t returnCode = 0;
    std::string a = "DEFAULT";
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.compileMode_ = Ascc::OutputFileType::FILE_SO;
    MOCKER(&Ascc::AsccLink::IsKernelFuncFound).stubs().will(returnValue(true));
    MOCKER(&Ascc::AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccHostStub::UpdateHostStubByDevice).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true));

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::SUCCESS);
    system("rm -rf /tmp/ascc");
}

// when output is .o
TEST_F(TEST_ASCC_LINK, ascc_AscendLink_output_o)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    system("mkdir -p /tmp/ascc/auto_gen");
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    std::string inputFileName = "sample_cpp/add_custom.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;

    int32_t returnCode = 0;
    std::string a = "DEFAULT";
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.compileMode_ = Ascc::OutputFileType::FILE_O;
    MOCKER(&Ascc::AsccLink::IsKernelFuncFound).stubs().will(returnValue(true));
    MOCKER(&Ascc::AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccHostStub::UpdateHostStubByDevice).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true));

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::SUCCESS);
    system("rm -rf /tmp/ascc");
}

// with -o options in parsing   -o testOutput
TEST_F(TEST_ASCC_LINK, ascc_AscendLink_with_parsing_o)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    system("mkdir -p /tmp/ascc/auto_gen");
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    std::string inputFileName = "sample_cpp/add_custom";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;

    int32_t returnCode = 0;
    std::string a = "DEFAULT";
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.outputFile_ = "testOutput";
    MOCKER(&Ascc::AsccLink::IsKernelFuncFound).stubs().will(returnValue(true));
    MOCKER(&Ascc::AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccHostStub::UpdateHostStubByDevice).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true));

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::SUCCESS);
    system("rm -rf /tmp/ascc");
}

// with -D options in parsing
TEST_F(TEST_ASCC_LINK, ascc_AscendLink_with_parsing_D)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    system("mkdir -p /tmp/ascc/auto_gen");
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    std::string inputFileName = "test22.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;

    system("touch test22.cpp");
    system("mkdir -p link_files/merge_obj");
    system("mkdir -p link_files/merge_obj_final");
    system("touch link_files/merge_obj/device_aiv.o");
    system("touch link_files/merge_obj_final/device_aiv.o");

    int32_t returnCode = 0;
    std::string a = "DEFAULT";
    std::vector<std::string> testArgs = {"-arch", "Ascend910B1", "test22.cpp", "-LSSSS", "-lssss", "-DFFFF"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    auto afff = argManager.ArgumentParse(testArgs);

    MOCKER(&Ascc::AsccLink::IsKernelFuncFound).stubs().will(returnValue(true));
    MOCKER(&Ascc::AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccHostStub::UpdateHostStubByDevice).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true));

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    system("rm test22.cpp");
    system("rm -rf link_files");
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::SUCCESS);
    system("rm -rf /tmp/ascc");
}

// Test MergeObj with functions and functionTemplates
TEST_F(TEST_ASCC_LINK, ascc_AscendLink_with_function_template)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    system("mkdir -p /tmp/ascc/auto_gen");
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    std::string inputFileName = "add_custom.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;
    system("touch add_custom.cpp");
    std::vector<std::string> testArgs = {"-arch", "Ascend910B1", "add_custom.cpp", "-LSSSS", "-lssss", "-DFFFF"};
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    auto afff = argManager.ArgumentParse(testArgs);

    auto& storage = Ascc::AsccInfoStorage::GetInstance();

    const std::shared_ptr<Ascc::AsccInfoFunction> &functions1 = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "example_function";
        info.definitionPos = "example_file.cpp:10";
        info.lineNo = 10;
        info.nameSpace = "example_namespace";
        info.returnType = "void";

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", false, Ascc::ParamType::NORMAL_INPUT);
        info.params.push_back(param);

        funcPtr->AddFunction("example_function", info);
        return funcPtr;
    }();

    const std::shared_ptr<Ascc::AsccInfoFunction> &functions2 = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.definitionPos = "hello_world.cpp:13";
        info.lineNo = 13;
        info.nameSpace = "Ascc";
        info.returnType = "void";

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", false, Ascc::ParamType::NORMAL_INPUT);
        info.params.push_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();

    const std::string& inputFile = Ascc::AsccArgumentManager::GetInstance().GetInputFile();

    storage.AddInfo(inputFile, Ascc::AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION, functions1);
    storage.AddInfo(inputFile, Ascc::AscCursorTypes::ASC_CURSOR_TEMPLATE_FUCNTION, functions2);
    int32_t returnCode = 0;
    std::string a = "DEFAULT";
    MOCKER(&Ascc::AsccLink::IsKernelFuncFound).stubs().will(returnValue(true));
    MOCKER(&Ascc::AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccHostStub::UpdateHostStubByDevice).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true));

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    system("rm add_custom.cpp");
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::SUCCESS);
    system("rm -rf /tmp/ascc");
}

// with --sanitizer
TEST_F(TEST_ASCC_LINK, ascc_AscendLink_sanitizer)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    system("mkdir -p /tmp/ascc/auto_gen");
    envVar.ascendCannPackagePath = "llt_cann_stub_path";
    std::string inputFileName = "sample_cpp/add_custom.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;

    int32_t returnCode = 0;
    std::string a = "DEFAULT";
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.sanitizerRequested_ = true;
    MOCKER(&Ascc::AsccLink::IsKernelFuncFound).stubs().will(returnValue(true));
    MOCKER(&Ascc::AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccHostStub::UpdateHostStubByDevice).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true));

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::SUCCESS);
    system("rm -rf /tmp/ascc");
}

TEST_F(TEST_ASCC_LINK, ascc_AscendLink_sanitizer_310p)
{
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    argManager.sanitizerRequested_ = true;
    argManager.npuArch_ = Ascc::ShortSoCVersion::ASCEND310P;
    Ascc::AsccLink asccLink("", "", Ascc::BuildType::DEBUG);
    auto linkStr = asccLink.SanitizerLinkProcess("");
    argManager.npuArch_ = Ascc::ShortSoCVersion::ASCEND910B;
    std::string expectRes= " --dependent-libraries /tools/mssanitizer/lib64/libsanitizer_stub_dav-m200-vec.a /tools/mssanitizer/lib64/libsanitizer_stub_dav-m200.a";
    EXPECT_EQ(linkStr == expectRes, true);

    argManager.sanitizerRequested_ = false;
    linkStr = asccLink.SanitizerLinkProcess("");
    expectRes= "";
    EXPECT_EQ(linkStr == expectRes, true);
}

TEST_F(TEST_ASCC_LINK, ascc_AscendLink_host_pre)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    system("mkdir -p /tmp/ascc/auto_gen");
    std::string inputFileName = "add_custom.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;
    Ascc::AsccArgumentManager::GetInstance().preTaskType_ = Ascc::PreTaskType::HOST;

    MOCKER(&Ascc::AsccHostStub::GenHostStubFile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccHostStub::UpdateHostStubByDevice).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(true));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true));

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_LINK, ascc_AscendLink_failure)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    envVar.ascendLinker = "/tmp/ascc";
    std::string inputFileName = "add_custom.cpp";
    Ascc::BuildType buildType = Ascc::BuildType::DEBUG;

    Ascc::AsccHostStub asccHostStub(Ascc::PreTaskType::NONE);
    Ascc::AsccLink asccLink(envVar.ascendLinker, inputFileName, buildType);
    EXPECT_EQ(asccLink.AscendLink(asccHostStub), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_LINK, ascc_LinkProcessForDeviceO_fail)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(false));
    Ascc::AsccLink asccLink(envVar.ascendLinker, "", Ascc::BuildType::DEBUG);
    EXPECT_EQ(asccLink.LinkProcessForDeviceO(envVar, "", ""), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_LINK, ascc_PackProcessForDeviceO_fail)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(false));
    Ascc::AsccLink asccLink(envVar.ascendLinker, "", Ascc::BuildType::DEBUG);
    EXPECT_EQ(asccLink.PackProcessForDeviceO(envVar, "", "", "", 0) == Ascc::AsccStatus::FAILURE, true);
}

TEST_F(TEST_ASCC_LINK, ascc_MergeObj)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(&Ascc::AsccLink::LinkProcessForDeviceOWithR).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    Ascc::AsccLink asccLink(envVar.ascendLinker, "", Ascc::BuildType::DEBUG);
    EXPECT_EQ(asccLink.MergeObj("", false), Ascc::AsccStatus::SUCCESS);
}