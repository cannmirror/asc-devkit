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
#define private public
#include "ascc_types.h"
#include "ascc_log.h"
#include "ascc_utils.h"
#include "ascc_mangle.h"
#include "ascc_dump_flags.h"
#include "ascc_info_function.h"
#include "ascc_host_stub.h"
#include "ascc_argument_manager.h"
#include "ascc_common_types.h"

class TEST_ASCC_HOST_STUB : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenHostStubFileMix)
{
    const std::shared_ptr<Ascc::AsccInfoFunction> &functions = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.definitionPos = "hello_world.cpp:13";
        info.lineNo = 13;
        info.nameSpace = "Ascc";
        info.returnType = "void";

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
        info.params.emplace_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();
    const std::shared_ptr<Ascc::AsccInfoFunction> &funcTemplates = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.definitionPos = "hello_world.cpp:13";
        info.lineNo = 13;
        info.nameSpace = "Ascc";
        info.returnType = "void";

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
        info.params.emplace_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();
    Ascc::AsccInfoStorage &storage = Ascc::AsccInfoStorage::GetInstance();
    storage.AddInfo("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        Ascc::AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION, functions);
    storage.AddInfo("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        Ascc::AscCursorTypes::ASC_CURSOR_TEMPLATE_FUCNTION, funcTemplates);
    std::string hostStubFilePath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub_tmp.h";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::HOST);
    EXPECT_NO_THROW(hostStubGenerator.GenHostStubFile());
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub_tmp.h");
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenHostStubFileAic)
{
    const std::shared_ptr<Ascc::AsccInfoFunction> &functions = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.definitionPos = "hello_world.cpp:13";
        info.lineNo = 13;
        info.nameSpace = "Ascc";
        info.returnType = "void";
        info.kernelType = Ascc::CodeMode::KERNEL_TYPE_AIC_ONLY;

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
        info.params.emplace_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();
    const std::shared_ptr<Ascc::AsccInfoFunction> &funcTemplates = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.definitionPos = "hello_world.cpp:13";
        info.lineNo = 13;
        info.nameSpace = "Ascc";
        info.returnType = "void";
        info.kernelType = Ascc::CodeMode::KERNEL_TYPE_AIC_ONLY;

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
        info.params.emplace_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();
    Ascc::AsccInfoStorage &storage = Ascc::AsccInfoStorage::GetInstance();
    storage.AddInfo("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        Ascc::AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION, functions);
    storage.AddInfo("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        Ascc::AscCursorTypes::ASC_CURSOR_TEMPLATE_FUCNTION, funcTemplates);
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/";

    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenHostStubFile());
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub.h");
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenHostStubFileAiv)
{
    const std::shared_ptr<Ascc::AsccInfoFunction> &functions = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.definitionPos = "hello_world.cpp:13";
        info.lineNo = 13;
        info.nameSpace = "Ascc";
        info.returnType = "void";
        info.kernelType = Ascc::CodeMode::KERNEL_TYPE_AIV_ONLY;

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
        info.params.emplace_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();
    const std::shared_ptr<Ascc::AsccInfoFunction> &funcTemplates = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.definitionPos = "hello_world.cpp:13";
        info.lineNo = 13;
        info.nameSpace = "Ascc";
        info.returnType = "void";
        info.kernelType = Ascc::CodeMode::KERNEL_TYPE_AIV_ONLY;

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
        info.params.emplace_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();
    Ascc::AsccInfoStorage &storage = Ascc::AsccInfoStorage::GetInstance();
    storage.AddInfo("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        Ascc::AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION, functions);
    storage.AddInfo("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp",
        Ascc::AscCursorTypes::ASC_CURSOR_TEMPLATE_FUCNTION, funcTemplates);
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenHostStubFile());
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub.h");
}

std::string absPath(std::string path)
{
#ifdef WIN32
    char absPath[4096] = {0}:
    _fullpath(absPath, path.c_str(), 4096);
#else
    char absPath[40960] = {0};
    realpath(path.c_str(), absPath);
#endif
    return std::string(absPath);
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenHostStubFile_pretask_host)
{
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = absPath("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/");
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenHostStubFile());
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.preTaskType_ = Ascc::PreTaskType::HOST;
    Ascc::AsccStatus result = hostStubGenerator.GenHostStubFile();
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
    manager.preTaskType_ = Ascc::PreTaskType::NONE;
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenHostStubFileFailure2)
{
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.inputFile_ = "";
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenHostStubFile());
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub.h");
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenHostStubFile_nullptr)
{
    std::shared_ptr<Ascc::AsccInfoBase> ptr = nullptr;
    MOCKER(&Ascc::AsccInfoStorage::GetInfo).stubs().will(returnValue(ptr));

    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenHostStubFile());
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub.h");
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenHostStubHeadCode)
{
    std::ofstream codeStream(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_host_head.txt", std::ios::out);
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    hostStubGenerator.hasMix_ = true;
    hostStubGenerator.hasAic_ = true;
    hostStubGenerator.hasAiv_ = true;
    hostStubGenerator.hasPrintf_ = true;
    hostStubGenerator.hasAssert_ = true;
    EXPECT_NO_THROW(hostStubGenerator.GenHostStubHeadCode());
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_host_head.txt");
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenerateFuncImplCodeReplace)
{
    auto &dumpInfo = Ascc::AsccDumpFlags::GetInstance();
    dumpInfo.SetPrintfFlag();
    std::string funcName = "hello_world";
    Ascc::CodeMode kernelType = Ascc::CodeMode::KERNEL_TYPE_AIV_ONLY;
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> params;
    params.emplace_back("x", "unsigned char*", true, Ascc::ParamType::NORMAL_INPUT);
    params.emplace_back("y", "unsigned char*", true, Ascc::ParamType::NORMAL_INPUT);
    params.emplace_back("z", "unsigned char*", true, Ascc::ParamType::NORMAL_INPUT);
    params.emplace_back("foo", "MyVar<float>", true, Ascc::ParamType::NORMAL_INPUT);
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = funcName;
    info.params = params;
    info.kernelType = kernelType;
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> tmpParams;
    tmpParams.emplace_back("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    Ascc::AsccInfoFunction::FunctionInfo tempInfo;
    tempInfo.funcName = funcName;
    tempInfo.params = params;
    tempInfo.kernelType = kernelType;
    tempInfo.templateParams = tmpParams;
    tempInfo.mangledToInstFuncInfo["mangled1"] = std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(tempInfo);
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenStubFuncImpl(info));
    EXPECT_NO_THROW(hostStubGenerator.GenStubFuncImpl(tempInfo));
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenerateFuncImplCodeReplaceSpec)
{
    auto &dumpInfo = Ascc::AsccDumpFlags::GetInstance();
    dumpInfo.SetPrintfFlag();
    std::string funcName = "hello_world";
    Ascc::CodeMode kernelType = Ascc::CodeMode::KERNEL_TYPE_AIV_ONLY;
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> params;
    params.emplace_back("x", "unsigned char*", true, Ascc::ParamType::NORMAL_INPUT);
    params.emplace_back("y", "unsigned char*", true, Ascc::ParamType::NORMAL_INPUT);
    params.emplace_back("z", "unsigned char*", true, Ascc::ParamType::NORMAL_INPUT);
    params.emplace_back("foo", "MyVar<float>", true, Ascc::ParamType::NORMAL_INPUT);
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = funcName;
    info.params = params;
    info.kernelType = kernelType;
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> tmpParams;
    tmpParams.emplace_back("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    Ascc::AsccInfoFunction::FunctionInfo tempInfo;
    tempInfo.funcName = funcName;
    tempInfo.params = params;
    tempInfo.kernelType = kernelType;
    tempInfo.templateParams = tmpParams;
    tempInfo.isTempExpSpec = true;
    tempInfo.isTempInst = false;
    tempInfo.mangledToInstFuncInfo["mangled1"] = std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(tempInfo);
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenStubFuncImpl(info));
    EXPECT_NO_THROW(hostStubGenerator.GenStubFuncImpl(tempInfo));
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenerateFuncImplCodeReplaceEmptyParam)
{
    auto &dumpInfo = Ascc::AsccDumpFlags::GetInstance();
    dumpInfo.SetPrintfFlag();
    std::string funcName = "hello_world";
    Ascc::CodeMode kernelType = Ascc::CodeMode::KERNEL_TYPE_AIV_ONLY;
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> tmpParams;
    tmpParams.emplace_back("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    Ascc::AsccInfoFunction::FunctionInfo tempInfo;
    tempInfo.funcName = funcName;
    tempInfo.kernelType = kernelType;
    tempInfo.templateParams = tmpParams;
    tempInfo.mangledToInstFuncInfo["mangled1"] = std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(tempInfo);
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenStubFuncImpl(tempInfo));
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenStubFuncImplWithPreTaskType)
{
    auto &dumpInfo = Ascc::AsccDumpFlags::GetInstance();
    dumpInfo.SetPrintfFlag();
    std::string funcName = "hello_world";
    Ascc::CodeMode kernelType = Ascc::CodeMode::KERNEL_TYPE_AIV_ONLY;
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> tmpParams;
    tmpParams.emplace_back("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    Ascc::AsccInfoFunction::FunctionInfo tempInfo;
    tempInfo.funcName = funcName;
    tempInfo.kernelType = kernelType;
    tempInfo.templateParams = tmpParams;
    tempInfo.mangledToInstFuncInfo["mangled1"] = std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(tempInfo);
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    Ascc::AsccArgumentManager::GetInstance().preTaskType_ = Ascc::PreTaskType::HOST;
    EXPECT_NO_THROW(hostStubGenerator.GenStubFuncImpl(tempInfo));
}


TEST_F(TEST_ASCC_HOST_STUB, ascc_UpdateHostStubByDevice)
{
    std::string hostStubFilePath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.cpp";
    std::string objectFilePath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.o";
    system(("touch ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub.h"));
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.UpdateHostStubByDevice());
    system(("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub.h"));
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_UpdateHostStubByDevice_PreTaskNone)
{
    auto& manager = Ascc::AsccArgumentManager::GetInstance();
    manager.preTaskType_ = Ascc::PreTaskType::NONE;
    system(("touch ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub.h"));
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.UpdateHostStubByDevice());
    system(("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/host_stub.h"));
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_UpdateHostStubByDevice_fail)
{
    std::string hostStubFilePath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.cpp";
    std::string objectFilePath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.o";
    system(("cp " + hostStubFilePath + " " + objectFilePath).c_str());
    MOCKER(static_cast<std::string (*)(const std::string &)>(&Ascc::CheckAndGetFullPath))
        .stubs().will(returnValue(std::string("fake")));
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(false));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(false));
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    Ascc::AsccStatus result = hostStubGenerator.UpdateHostStubByDevice();
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);
    system(("rm " + objectFilePath).c_str());
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenAclrtLaunchNormal)
{
    const std::shared_ptr<Ascc::AsccInfoFunction> &functions = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.definitionPos = "hello_world.cpp:13";
        info.lineNo = 13;
        info.nameSpace = "Ascc";
        info.returnType = "void";

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
        info.params.emplace_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();
    std::string tmpFile = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/generate_aclrt_launch_normal.txt";
    std::ofstream codeStream(tmpFile, std::ios::out);

    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenLaunchProfilingBody(functions));
    system(("rm " + tmpFile).c_str());
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_GenAclrtLaunchNormal_nullptr)
{
    std::string tmpFile = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/generate_aclrt_launch_normal.txt";
    std::ofstream codeStream(tmpFile, std::ios::out);

    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenLaunchProfilingBody(nullptr));
    system(("rm " + tmpFile).c_str());
}


TEST_F(TEST_ASCC_HOST_STUB, ascc_GenUpdateFuncHandleTypeMap)
{
    MOCKER(&Ascc::AsccMangle::GetOriginToFixedMangledNames).stubs().will(returnValue(
        std::unordered_map<std::string, std::string>{{"_Z25", "_Z25"}}));
    const std::shared_ptr<Ascc::AsccInfoFunction> &functions = []() {
    auto funcPtr = std::make_shared<Ascc::AsccInfoFunction>();
        Ascc::AsccInfoFunction::FunctionInfo info;
        info.funcName = "hello_world";
        info.kernelType = Ascc::CodeMode::KERNEL_TYPE_AIC_ONLY;
        info.mangledToInstFuncInfo["mangled1"] = std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(info);

        Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
        info.params.emplace_back(param);

        funcPtr->AddFunction("hello_world", info);
        return funcPtr;
    }();
    std::stringstream codeSource;

    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    EXPECT_NO_THROW(hostStubGenerator.GenManglingRegisterBody(functions));
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_ManglingNameJudgeCode)
{
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = "hello_world";
    info.definitionPos = "hello_world.cpp:13";
    info.nameSpace = "Ascc";
    info.isTemplate = true;
    info.isTempInst = false;
    info.isTempExpSpec = false;
    info.returnType = "void";
    info.templateParams.emplace_back("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    info.templateParams.emplace_back("Y", "uint32_t", false, Ascc::ParamType::TEMPLATE_INT);
    info.templateParams.emplace_back("U", "const auto&", false, Ascc::ParamType::TEMPLATE_DECL);
    info.templateParams.emplace_back("I", "template <typename> class", false, Ascc::ParamType::TEMPLATE_TEMPLATE);
    Ascc::AsccInfoFunction::FunctionInfo infoInst;
    infoInst.funcName = "hello_world";
    infoInst.definitionPos = "hello_world.cpp:13";
    infoInst.nameSpace = "Ascc";
    infoInst.manglingName = "test";
    infoInst.returnType = "void";
    infoInst.templateParams.emplace_back("", "int", false, Ascc::ParamType::TEMPLATE_TYPE);
    infoInst.templateParams.emplace_back("", "256", false, Ascc::ParamType::TEMPLATE_INT);
    infoInst.templateParams.emplace_back("", "QWERT", false, Ascc::ParamType::TEMPLATE_DECL);
    infoInst.templateParams.emplace_back("", "Contain", false, Ascc::ParamType::TEMPLATE_TEMPLATE);
    Ascc::AsccInfoFunction::FunctionInfo infoInstNone;
    infoInstNone.manglingName = "None";
    info.mangledToInstFuncInfo.emplace("test", std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(infoInst));
    info.mangledToInstFuncInfo.emplace("None", std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(infoInstNone));
    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::NONE);
    hostStubGenerator.kernelCallStubFile_.open("/tmp/tmp.cpp");
    MOCKER(&Ascc::AsccMangle::GetOriginToFixedMangledNames).stubs().will(returnValue(
            std::unordered_map<std::string, std::string>{{"test", "test"}}));
    EXPECT_NO_THROW(hostStubGenerator.ManglingNameJudgeCode(info));
    hostStubGenerator.kernelCallStubFile_.close();
    system("rm -rf /tmp/tmp.cpp");
}

TEST_F(TEST_ASCC_HOST_STUB, ascc_KernelIsCall)
{
    Ascc::AsccArgumentManager::GetInstance().preTaskType_ = Ascc::PreTaskType::HOST;
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = "hello_world";
    info.definitionPos = "hello_world.cpp:13";
    info.nameSpace = "Ascc";
    info.isTemplate = false;
    info.returnType = "void";
    info.manglingName = "xxxxxvvvvv";
    info.templateParams.emplace_back("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    info.templateParams.emplace_back("Y", "uint32_t", false, Ascc::ParamType::TEMPLATE_INT);
    info.templateParams.emplace_back("U", "const auto&", false, Ascc::ParamType::TEMPLATE_DECL);
    info.templateParams.emplace_back("I", "template <typename> class", false, Ascc::ParamType::TEMPLATE_TEMPLATE);

    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath = "";
    Ascc::AsccHostStub hostStubGenerator(Ascc::PreTaskType::HOST);
    hostStubGenerator.kernelCallStubFile_.open("/tmp/tmp.cpp");
    EXPECT_NO_THROW(hostStubGenerator.GenStubFuncImpl(info));
    hostStubGenerator.kernelCallStubFile_.close();
    system("rm -rf /tmp/tmp.cpp");
}
