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
#define protected public
#include "ascc_utils.h"
#include "ascc_global_env_manager.h"
#include "ascc_dump_flags.h"
#include "ascc_device_stub.h"
#include "ascc_info_function.h"
#include "ascc_compile_factory.h"
#include "ascc_compile_v220.h"
#include "ascc_mangle.h"
#include "ascc_match_global_info.h"

class TEST_ASCC_DEVICE_STUB : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GeneratDeviceStub)
{
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> paramList = {
        {"a", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT},
        {"b", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT},
        {"workspace", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT}};

    Ascc::AsccGlobalEnvManager::GetInstance().asccTmpPath = "./";
    Ascc::AsccDumpFlags::GetInstance().SetAssertFlag();
    Ascc::AsccDumpFlags::GetInstance().SetPrintfFlag();
    Ascc::AsccInfoStorage &storage = Ascc::AsccInfoStorage::GetInstance();
    std::shared_ptr<Ascc::AsccInfoFunction> normalFuncInfo = std::make_shared<Ascc::AsccInfoFunction>();
    Ascc::AsccInfoFunction::FunctionInfo funcInfoStruct;
    funcInfoStruct.returnType = "void";
    funcInfoStruct.funcName = "Foo";
    funcInfoStruct.definitionPos = "1";
    funcInfoStruct.kernelType = Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
    funcInfoStruct.nameSpace = "AsccTest";
    funcInfoStruct.lineNo = 1;
    funcInfoStruct.params = paramList;
    normalFuncInfo->AddFunction("Foo", funcInfoStruct);
    system("touch test.cpp");
    storage.AddInfo("./test.cpp", Ascc::AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION, normalFuncInfo);
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    MOCKER(&Ascc::AsccCompileBase::ExecuteCompile).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    EXPECT_EQ(generator.Process(), Ascc::AsccStatus::SUCCESS);
    MOCKER(&Ascc::AsccDeviceStub::GenDeviceStubCode).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    EXPECT_EQ(generator.Process(), Ascc::AsccStatus::SUCCESS);
}

std::string absolutePath(std::string path)
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

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GenTmpDeviceCode)
{
    std::string inputFile = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world.cpp";
    std::string outputFile = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_device.txt";
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::unordered_map<std::string, std::unordered_set<std::string>> kernelCallLines;
    
    std::string parent = absolutePath("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/");
    std::string inputFileReal = parent + "/hello_world.cpp";
    std::string outputFileReal = parent + "/hello_world_device.txt";

    std::unordered_set<std::string> value1 = {"27:5", "3:4"};
    kernelCallLines.insert({inputFileReal, value1});

    Ascc::AsccStatus result = generator.GenTmpDeviceCode(inputFileReal, outputFileReal, kernelCallLines);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_device.txt");

    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(true)).then(returnValue(false));
    result = generator.GenTmpDeviceCode(inputFileReal, outputFileReal, kernelCallLines);
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GenTemplateStubFuncDefinition)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_definition.txt", std::ios::out);
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = "hello_world";
    info.returnType = "void";

    Ascc::AsccInfoFunction::ParameterInfo templateParam("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    info.templateParams.push_back(templateParam);
    info.templateParams.push_back(templateParam);

    std::string originCallTempParams = "";
    generator.GenTemplateStubFuncDefinition(stubCode, info, originCallTempParams);
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_definition.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_StubFuncTemplateInstanceImpl)
{
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_definition.txt", std::ios::out);
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = "hello_world";
    info.definitionPos = "hello_world.cpp:13";
    info.lineNo = 13;
    info.nameSpace = "Ascc";
    info.returnType = "void";

    Ascc::AsccInfoFunction::ParameterInfo templateParam("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    info.templateParams.push_back(templateParam);
    info.templateParams.push_back(templateParam);

    Ascc::AsccInfoFunction::ParameterInfo param("x", "int", true, Ascc::ParamType::NORMAL_INPUT);
    info.params.push_back(param);

    size_t paramsSize = 2;
    info.mangledToInstFuncInfo["mangled1"] = std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(info);
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    Ascc::AsccStatus result = generator.StubFuncInstImpl(stubCode, info);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
    result = generator.StubFuncInstImpl(stubCode, info);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_definition.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GenKtypeSection)
{
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_definition.txt", std::ios::out);
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = "hello_world";
    info.definitionPos = "hello_world.cpp:13";
    info.lineNo = 13;
    info.nameSpace = "Ascc";
    info.returnType = "void";

    Ascc::AsccInfoFunction::ParameterInfo templateParam("T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE);
    info.templateParams.push_back(templateParam);
    info.templateParams.push_back(templateParam);

    Ascc::AsccInfoFunction::ParameterInfo param("x", "int", false, Ascc::ParamType::NORMAL_INPUT);
    info.params.push_back(param);

    std::string mangName = "mangled1";
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    generator.codeModeMask_ = 1;
    Ascc::AsccStatus result =
        generator.GenKtypeSection(stubCode, info, mangName);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);

    generator.kernelType_ = Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_0;
    result = generator.GenKtypeSection(stubCode, info, mangName);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);

    generator.kernelType_ = Ascc::CodeMode::KERNEL_TYPE_MIX_AIV_1_0;
    result = generator.GenKtypeSection(stubCode, info, mangName);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);

    generator.codeModeMask_ = 0;
    result = generator.GenKtypeSection(stubCode, info, mangName);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);

    MOCKER(&Ascc::AsccDeviceStub::GetKernelType).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    result = generator.GenKtypeSection(stubCode, info, mangName);
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_definition.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GetKtypeSectionVariable)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    Ascc::AsccDeviceStub::KtypesSectionParam params{"", "", ""};
    Ascc::FuncMetaType funcMetaType = Ascc::FuncMetaType::F_TYPE_KTYPE;
    Ascc::AsccDeviceStub::KernelSectionMode genMode = Ascc::AsccDeviceStub::KernelSectionMode::MIX_AIV;
    std::string mangName = "mangled1";
    std::string result = generator.GetKtypeSectionVariable(params, funcMetaType, genMode, mangName);
    EXPECT_EQ(result,
        "static const struct  _mix_aiv_section_0 __attribute__ ((used, section (\".ascend.meta.mangled1_mix_aiv\"))) = "
        "{ { {F_TYPE_KTYPE, sizeof(unsigned int)}, }, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };\n");
    genMode = Ascc::AsccDeviceStub::KernelSectionMode::MIX_AIV;
    result = generator.GetKtypeSectionVariable(params, funcMetaType, genMode, mangName);
    EXPECT_EQ(result,
        "static const struct  _mix_aiv_section_0 __attribute__ ((used, section (\".ascend.meta.mangled1_mix_aiv\"))) = "
        "{ { {F_TYPE_KTYPE, sizeof(unsigned int)}, }, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };\n");
    genMode = Ascc::AsccDeviceStub::KernelSectionMode::NO_MIX;
    result = generator.GetKtypeSectionVariable(params, funcMetaType, genMode, mangName);
    EXPECT_EQ(result,
        "static const struct  _section_0 __attribute__ ((used, section (\".ascend.meta.mangled1\"))) = { { "
        "{F_TYPE_KTYPE, sizeof(unsigned int)}, }, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };\n");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GetManglingList)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::vector<std::string> manglingNameList;
    Ascc::AsccInfoFunction::FunctionInfo newFuncInfo;
    newFuncInfo.funcName = "hello_world";
    Ascc::AsccInfoFunction::FunctionInfo funcInfo;
    funcInfo.funcName = "hello_world";
    Ascc::AsccStatus result =
        generator.GetManglingList(manglingNameList, newFuncInfo);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);

    MOCKER(&Ascc::CheckAndGetFullPath, std::string(const std::string &))
        .stubs().will(returnValue(std::string("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world")));
    MOCKER(Ascc::CompileTask<Ascc::AsccCompileV220>).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::NOT_EXIST));
    result = generator.GetManglingList(manglingNameList, newFuncInfo);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GetManglingList_All)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::vector<std::string> manglingNameList;
    Ascc::AsccInfoFunction::FunctionInfo newFuncInfo;
    newFuncInfo.funcName = "hello_world";
    newFuncInfo.manglingName = "manglingascc_GetManglingList_All";
    newFuncInfo.isTemplate = false;
    newFuncInfo.isTempExpSpec = false;
    EXPECT_EQ(generator.GetManglingList(manglingNameList, newFuncInfo), Ascc::AsccStatus::SUCCESS);
    newFuncInfo.isTemplate = true;
    newFuncInfo.isTempExpSpec = false;
    EXPECT_EQ(generator.GetManglingList(manglingNameList, newFuncInfo), Ascc::AsccStatus::SUCCESS);
    auto& ascFixedMangleMap = Ascc::AsccMangle::GetInstance().GetOriginToFixedMangledNames();
    ascFixedMangleMap.emplace("manglingascc_GetManglingList_All", "mangling");
    EXPECT_EQ(generator.GetManglingList(manglingNameList, newFuncInfo), Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_StubFuncKtypeSectionImpl)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_ktype.txt", std::ios::out);
    Ascc::AsccInfoFunction::FunctionInfo newFuncInfo;
    newFuncInfo.mangledToInstFuncInfo["mangled1"] = std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(newFuncInfo);
    Ascc::AsccInfoFunction::FunctionInfo funcInfo;
    funcInfo.funcName = "hello_world";
    Ascc::AsccStatus result =
        generator.StubFuncKtypeSectionImpl(stubCode, newFuncInfo);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_ktype.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_StubFuncKtypeSectionImpl_310)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND310P, "./");
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_ktype.txt", std::ios::out);
    Ascc::AsccInfoFunction::FunctionInfo newFuncInfo;
    Ascc::AsccInfoFunction::FunctionInfo funcInfo;
    Ascc::AsccStatus result =
        generator.StubFuncKtypeSectionImpl(stubCode, newFuncInfo);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_ktype.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_CompileDeviceStub)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::string dstFilePath_Chek = Ascc::CheckAndGetFullPath("./");
    Ascc::AsccStatus result = generator.CompileDeviceStub();
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
    generator.stubIsGen_.emplace(dstFilePath_Chek + "/device_stub_aic.cpp");
    generator.codeModeMask_ = 1;
    MOCKER(Ascc::CompileTask<Ascc::AsccCompileV220>).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    result = generator.CompileDeviceStub();
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);

    generator.codeModeMask_ =
        (static_cast<uint32_t>(1) << static_cast<uint32_t>(Ascc::CodeMode::KERNEL_TYPE_AIC_ONLY)) |
        (static_cast<uint32_t>(1) << static_cast<uint32_t>(Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_0));
    result = generator.CompileDeviceStub();
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_CompileDeviceStub_mix_1_1)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    generator.kernelType_ = Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_1;
    std::string dstFilePath_Chek = Ascc::CheckAndGetFullPath("./");
    generator.stubIsGen_.emplace(dstFilePath_Chek + "/device_stub_mix_1_2.cpp");
    generator.stubIsGen_.emplace(dstFilePath_Chek + "/device_stub_mix_1_1.cpp");
    generator.stubIsGen_.emplace(dstFilePath_Chek + "/device_stub_aiv.cpp");
    generator.stubIsGen_.emplace(dstFilePath_Chek + "/device_stub_aic.cpp");
    generator.codeModeMask_ = static_cast<uint32_t>(1) << static_cast<uint32_t>(generator.kernelType_);
    MOCKER(Ascc::CompileTask<Ascc::AsccCompileV220>).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    Ascc::AsccStatus result = generator.CompileDeviceStub();
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GenStubFunc)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = "hello_world";
    info.definitionPos = "hello_world.cpp:13";
    info.lineNo = 13;
    info.nameSpace = "Ascc";
    info.returnType = "void";
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/get_stub_func.txt", std::ios::out);
    Ascc::AsccStatus result =
        generator.GenStubFunc(stubCode, info);
    MOCKER(&Ascc::AsccDeviceStub::StubFuncInstImpl).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    result = generator.GenStubFunc(stubCode, info);
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/get_stub_func.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GetAllNestedNameSpace)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::string nameSpacePrefix = "";
    generator.GetAllNestedNameSpace(nameSpacePrefix);
    nameSpacePrefix = "AAA::BBB::CCC";
    generator.GetAllNestedNameSpace(nameSpacePrefix);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_StubFuncWorkSpaceImpl)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_workspace.txt", std::ios::out);
    MOCKER(&Ascc::AsccDeviceStub::IsMix).stubs().will(returnValue(true));
    generator.haveWorkspace_ = true;
    generator.StubFuncWorkSpaceImpl(stubCode, true, false);
    stubCode.close();
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_workspace.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_Process)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    using FileInfos = std::unordered_map<Ascc::AscCursorTypes, std::shared_ptr<Ascc::AsccInfoBase>>;
    static const std::unordered_map<std::string, FileInfos> empty_map{};
    MOCKER(&Ascc::AsccInfoStorage::GetAllInfos).stubs().will(returnValue(empty_map));
    Ascc::AsccStatus result = generator.Process();
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_ProcessWithPreTaskType)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    using FileInfos = std::unordered_map<Ascc::AscCursorTypes, std::shared_ptr<Ascc::AsccInfoBase>>;
    static const std::unordered_map<std::string, FileInfos> empty_map{};
    MOCKER(&Ascc::AsccInfoStorage::GetAllInfos).stubs().will(returnValue(empty_map));
    Ascc::AsccArgumentManager::GetInstance().preTaskType_ = Ascc::PreTaskType::DEVICE_AIV;
    Ascc::AsccStatus result = generator.Process();
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_UpdateNewWorkflowFlag)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    static std::unordered_map<std::string, std::unordered_set<std::string>> mocker_map{
        {"../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.cpp", {"0:1"}}
    };
    std::string inputFile("../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.cpp");
    MOCKER(&Ascc::AsccMatchGlobalInfo::GetGlobalKernelCallLineColumn).stubs().will(returnValue(mocker_map));
    MOCKER(&Ascc::AsccDeviceStub::GenTmpDeviceCode).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    std::string filePath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.cpp";
    Ascc::AsccStatus result = generator.UpdateNewWorkflowFlag();
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
    MOCKER(&Ascc::AsccArgumentManager::GetInputFile).stubs().will(returnValue(inputFile));
    result = generator.UpdateNewWorkflowFlag();
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_DeviceStubCodeImpl)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::string filePath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.cpp";
    std::shared_ptr<Ascc::AsccInfoFunction> funcInfo = nullptr;
    Ascc::AsccStatus result =
        generator.DeviceStubCodeImpl(filePath, funcInfo);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);

    std::shared_ptr<Ascc::AsccInfoFunction> normalFuncInfo = std::make_shared<Ascc::AsccInfoFunction>();
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = "hello_world";
    normalFuncInfo->AddFunction("hello_world", info);
    MOCKER(&Ascc::AsccDeviceStub::GenHeadCode).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccDeviceStub::GenStubFunc)
        .stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccDeviceStub::CompileDeviceStub).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    result = generator.DeviceStubCodeImpl(
        filePath, normalFuncInfo);
    EXPECT_EQ(result, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_DeviceStubCodeImplFailed)
{
    Ascc::AsccDeviceStub generatorFailed(Ascc::ShortSoCVersion::ASCEND910B, "/test");
    std::string filePath = "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/hello_world_empty.cpp";
    std::shared_ptr<Ascc::AsccInfoFunction> normalFuncInfo = std::make_shared<Ascc::AsccInfoFunction>();
    Ascc::AsccInfoFunction::FunctionInfo info;
    info.funcName = "hello_world";
    normalFuncInfo->AddFunction("hello_world", info);
    MOCKER(&Ascc::AsccDeviceStub::GenStubFunc).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    Ascc::AsccStatus result = generatorFailed.DeviceStubCodeImpl(filePath, normalFuncInfo);
    EXPECT_EQ(result, Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_StubFuncDumpAndHardSyncImpl)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_dump.txt", std::ios::out);
    MOCKER(&Ascc::AsccDeviceStub::IsMix).stubs().will(returnValue(true));
    generator.StubFuncDumpAndHardSyncImpl(stubCode, true, true, true);
    stubCode.close();
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_dump.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_StubFuncCallImpl)
{
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_call.txt", std::ios::out);
    Ascc::AsccInfoFunction::FunctionInfo funcInfo;
    generator.StubFuncCallImpl(stubCode, funcInfo, "add");
    stubCode.close();
    system("rm ../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_call.txt");
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GenTemplateExpSpecStubFuncDefinition)
{
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> paramList = {
        {"a", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT},
        {"b", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT},
        {"workspace", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT}};
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_definition.txt", std::ios::out);
    std::string code = "";
    Ascc::AsccInfoFunction::FunctionInfo funcInfoStruct;
    funcInfoStruct.returnType = "void";
    funcInfoStruct.funcName = "Foo";
    funcInfoStruct.definitionPos = "1";
    funcInfoStruct.kernelType = Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
    funcInfoStruct.nameSpace = "AsccTest";
    funcInfoStruct.lineNo = 1;
    funcInfoStruct.params = paramList;
    funcInfoStruct.templateParams.push_back({"T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE});
    funcInfoStruct.templateParams.push_back({"U", "typename", false, Ascc::ParamType::TEMPLATE_TYPE});
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    generator.GenTemplateExpSpecStubFuncDefinition(stubCode, funcInfoStruct, code);
    EXPECT_EQ(code, std::string("T, U"));
}

TEST_F(TEST_ASCC_DEVICE_STUB, ascc_GenSymbolImpl)
{
    std::vector<Ascc::AsccInfoFunction::ParameterInfo> paramList = {
        {"a", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT},
        {"b", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT},
        {"workspace", "uint8_t*", true, Ascc::ParamType::NORMAL_INPUT}};
    std::ofstream stubCode(
        "../../../../../tests/unit/basic_api/ut/testcase/ascc/hello_world/stub_func_definition.txt", std::ios::out);
    Ascc::AsccInfoFunction::FunctionInfo funcInfoStruct;
    funcInfoStruct.returnType = "void";
    funcInfoStruct.funcName = "Foo";
    funcInfoStruct.definitionPos = "1";
    funcInfoStruct.kernelType = Ascc::CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
    funcInfoStruct.nameSpace = "AsccTest";
    funcInfoStruct.lineNo = 1;
    funcInfoStruct.params = paramList;
    funcInfoStruct.isTempExpSpec = true;
    funcInfoStruct.templateParams.push_back({"T", "typename", false, Ascc::ParamType::TEMPLATE_TYPE});
    funcInfoStruct.templateParams.push_back({"U", "typename", false, Ascc::ParamType::TEMPLATE_TYPE});
    funcInfoStruct.mangledToInstFuncInfo.emplace(
        "aaaa", std::make_shared<Ascc::AsccInfoFunction::FunctionInfo>(funcInfoStruct));
    Ascc::AsccDeviceStub generator(Ascc::ShortSoCVersion::ASCEND910B, "./");
    MOCKER(&Ascc::AsccMatchGlobalInfo::IsCalled).stubs().will(returnValue(true));
    EXPECT_EQ(generator.GenNormalFuncSymbolImpl(stubCode, funcInfoStruct), Ascc::AsccStatus::SUCCESS);
    generator.endExtraCounter_ = 2;
    EXPECT_EQ(generator.GenTempFuncSymbolImpl(stubCode, funcInfoStruct), Ascc::AsccStatus::SUCCESS);
}