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

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <gtest/gtest.h>
#include <unistd.h>
#include <mockcpp/mockcpp.hpp>
#include <sys/syscall.h>
#include "asc_struct.h"
#include "asc_utils.h"
#define private public
#include "asc_log.h"
#include "asc_info_manager.h"
#include "asc_auto_identify_ktype.h"

class TEST_ASC_AUTO_IDENTIFY_KTYPE : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_normal_func)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    int32_t returnCode = 0;
    std::string a = " 0550 ac02c240 3060d703 0014e040 90b79c70\n0560 42601703 0050e303 86e09f08 40eee103";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_normal_func_return_cube)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    int32_t returnCode = 0;
    std::string a = " 0550 ac02c240 3060d703 0014e040 90b79c70\n0560 42601703 0050e303 86e09f08 40eee103";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    uint8_t mode = 1;
    MOCKER(AscPlugin::GetV220CoreMode).stubs().will(returnValue(mode));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_normal_func_return_mix)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    int32_t returnCode = 0;
    std::string a = " 0550 ac02c240 3060d703 0014e040 90b79c70\n0560 42601703 0050e303 86e09f08 40eee103";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    uint8_t mode = 3;
    MOCKER(AscPlugin::GetV220CoreMode).stubs().will(returnValue(mode));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_normal_func_unexpect_return)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    int32_t returnCode = 0;
    std::string a = " 0550 ac02c240 3060d703 0014e040 90b79c70\n0560 42601703 0050e303 86e09f08 40eee103";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    uint8_t mode = 4;
    MOCKER(AscPlugin::GetV220CoreMode).stubs().will(returnValue(mode));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_template_func)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    AscPlugin::TemplateInstance tempInst = {
        {},
        {},
        "testMangledName",
        "__device_stub__testMangledName"
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances = {tempInst};
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    int32_t returnCode = 0;
    std::string a = " 0550 ac02c240 3060d703 0014e040 90b79c70\n0560 42601703 0050e303 86e09f08 40eee103";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    std::unordered_map<std::string, std::string> textMap;
    textMap["testMangledName"] = ".text.testMangledName";
    MOCKER(&AscPlugin::GetTextMap).stubs().will(returnValue(textMap));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_template_unknown_func)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    AscPlugin::TemplateInstance tempInst = {
        {},
        {},
        "testMangledName",
        "__device_stub__testMangledName"
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances = {tempInst};
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    int32_t returnCode = 0;
    std::string a = " 0550 ac02c240 3060d703 0014e040 90b79c70\n0560 42601703 0050e303 86e09f08 40eee103";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, returnCode)));
    std::unordered_map<std::string, std::string> textMap;
    MOCKER(&AscPlugin::GetTextMap).stubs().will(returnValue(textMap));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_ExecuteCommand_error)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    AscPlugin::TemplateInstance tempInst = {
        {},
        {},
        "testMangledName",
        "__device_stub__testMangledName"
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances = {tempInst};
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    std::string a = " 0550 ac02c240 3060d703 0014e040 90b79c70\n0560 42601703 0050e303 86e09f08 40eee103";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, -1)));
    std::unordered_map<std::string, std::string> textMap;
    textMap["testMangledName"] = ".text.testMangledName";
    MOCKER(&AscPlugin::GetTextMap).stubs().will(returnValue(textMap));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_ExecuteCommand_output_empty)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    AscPlugin::TemplateInstance tempInst = {
        {},
        {},
        "testMangledName",
        "__device_stub__testMangledName"
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances = {tempInst};
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>("", 0)));
    std::unordered_map<std::string, std::string> textMap;
    textMap["testMangledName"] = ".text.testMangledName";
    MOCKER(&AscPlugin::GetTextMap).stubs().will(returnValue(textMap));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_identify_ktype_not_saveTemp)
{
    AscPlugin::KernelFuncInfo kernelKey = {
        "_Z11hello_worldv",
        "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp",
        13,
        28
    };
    AscPlugin::TemplateInstance tempInst = {
        {},
        {},
        "testMangledName",
        "__device_stub__testMangledName"
    };
    std::vector<AscPlugin::TemplateInstance> templateInstances = {tempInst};
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = false;
    AscPlugin::InfoManager::GetInstance().sourceFile_ = "./llt/atc/tikcpp/ut/testcase/ascc/hello_world/hello_world.cpp";
    std::string a = " 0550 ac02c240 3060d703 0014e040 90b79c70\n0560 42601703 0050e303 86e09f08 40eee103";
    MOCKER(&AscPlugin::AscCompileV220::Compile).stubs().will(returnValue(0));
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, 0)));
    std::unordered_map<std::string, std::string> textMap;
    MOCKER(&AscPlugin::GetTextMap).stubs().will(returnValue(textMap));
    EXPECT_EQ(IdentifyKtypeImpl(kernelKey, templateInstances), AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_Compile)
{
    std::string testCoreType;
    testCoreType = "aic";
    AscPlugin::AscCompileV220 compileUnitAic(testCoreType);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.saveTempRequested_ = true;
    manager.pathInfo_.bishengPath = std::string("bisheng");
    MOCKER(AscPlugin::ExecuteCompile).stubs().will(returnValue(0));
    EXPECT_EQ(compileUnitAic.Compile(), 0);

    testCoreType = "aiv";
    AscPlugin::AscCompileV220 compileUnitAiv(testCoreType);
    manager.saveTempRequested_ = false;
    EXPECT_EQ(compileUnitAiv.Compile(), 0);
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_GetTextMap_failed_cmd)
{
    std::string command = "failed command";
    std::string outputFile = "test.o";
    std::string a = "0000000000000000 g     F .text._Z11mmad_customIDhfLj30EEvPhS0_S0_S0_    0000000000001074 _Z11mmad_customIDhfLj30EEvPhS0_S0_S0_";
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, -1)));
    EXPECT_NO_THROW(AscPlugin::GetTextMap(command, outputFile));
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_GetTextMap_output_empty)
{
    std::string command = "failed command";
    std::string outputFile = "test.o";
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>("", 0)));
    EXPECT_NO_THROW(AscPlugin::GetTextMap(command, outputFile));
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_GetTextMap_Invalid_format)
{
    std::string command = "failed command";
    std::string outputFile = "test.o";
    std::string a = " F .text._Z11mmad_customIDhfLj30EEvPhS0_S0_S0_    0000000000001074 _Z11mmad_customIDhfLj30EEvPhS0_S0_S0_";
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, 0)));

    EXPECT_NO_THROW(AscPlugin::GetTextMap(command, outputFile));
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_GetTextMap)
{
    std::string command = "failed command";
    std::string outputFile = "test.o";
    std::string a = "0000000000000000 g     F .text._Z11mmad_customIDhfLj30EEvPhS0_S0_S0_    0000000000001074 _Z11mmad_customIDhfLj30EEvPhS0_S0_S0_";
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(std::pair<std::string, int>(a, 0)));
    EXPECT_NO_THROW(AscPlugin::GetTextMap(command, outputFile));
}

TEST_F(TEST_ASC_AUTO_IDENTIFY_KTYPE, asc_GetV220CoreMode_return_cube)
{
    std::string output;
    output = "0550 ac02826b 3060d789 0014e0f0 08b33c71";
    EXPECT_NO_THROW(AscPlugin::GetV220CoreMode(output));
}

