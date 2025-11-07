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
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#define private public
#include "asc_struct.h"
#include "asc_json_string.h"
#include "asc_utils.h"

class TEST_ASC_JSON_STRING : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_JSON_STRING, asc_fromjson_PrologueConfig)
{
    AscPlugin::PrologueConfig config = {true, true, AscPlugin::GenMode::SIMT_ONLY, "Ascend910B1", "dav-c220-cube",
        "tmp-path", "log-path", "a.cpp", "binaryptrname", "binarylenname", {"-DASCENDC_DUMP=1", "-DASCC_HOST"}};

    std::string expectRes = "{\"BinaryLenName\":\"binarylenname\",\"BinaryPtrName\":\"binaryptrname\",\"CompileArgs\":[\"-DASCENDC_DUMP=1\",\"-DASCC_HOST\"],\"GenMode\":3,\"LogPath\":\"log-path\",\"NpuArch\":\"dav-c220-cube\",\"NpuSoc\":\"Ascend910B1\",\"SaveTemp\":true,\"Source\":\"a.cpp\",\"TmpPath\":\"tmp-path\",\"Verbose\":true}";

    const char* a = expectRes.c_str();
    AscPlugin::PrologueConfig configFromJsonStr;
    int32_t fromJsonRes = FromJson(configFromJsonStr, a);
    EXPECT_EQ(configFromJsonStr.saveTemp, config.saveTemp);
    EXPECT_EQ(configFromJsonStr.verbose, config.verbose);
    EXPECT_EQ(configFromJsonStr.genMode, config.genMode);
    EXPECT_EQ(configFromJsonStr.npuSoc, config.npuSoc);
    EXPECT_EQ(configFromJsonStr.npuArch, config.npuArch);
    EXPECT_EQ(configFromJsonStr.logPath, config.logPath);
    EXPECT_EQ(configFromJsonStr.tmpPath, config.tmpPath);
    EXPECT_EQ(configFromJsonStr.source, config.source);
    EXPECT_EQ(configFromJsonStr.binaryPtrName, config.binaryPtrName);
    EXPECT_EQ(configFromJsonStr.binaryLenName, config.binaryLenName);
    EXPECT_EQ(configFromJsonStr.compileArgs, config.compileArgs);
}

TEST_F(TEST_ASC_JSON_STRING, asc_fromjson_PrologueConfig_arg_missing)
{
    AscPlugin::PrologueConfig config = {true, true, AscPlugin::GenMode::SIMT_ONLY, "Ascend910B1", "dav-c220-cube",
        "tmp-path", "log-path", "a.cpp", "binaryptrname", "binarylenname", {"-DASCENDC_DUMP=1", "-DASCC_HOST"}};

    std::string expectRes = "{\"BinaryPtrName\":\"binaryptrname\",\"CompileArgs\":[\"-DASCENDC_DUMP=1\",\"-DASCC_HOST\"],\"GenMode\":3,\"LogPath\":\"log-path\",\"NpuArch\":\"dav-c220-cube\",\"NpuSoc\":\"Ascend910B1\",\"SaveTemp\":true,\"Source\":\"a.cpp\",\"TmpPath\":\"tmp-path\",\"Verbose\":true}";

    const char* a = expectRes.c_str();
    AscPlugin::PrologueConfig configFromJsonStr;
    int32_t fromJsonRes = FromJson(configFromJsonStr, a);
    EXPECT_EQ(fromJsonRes, AscPlugin::ASC_JSONSTR_ARG_MISSING);
}

TEST_F(TEST_ASC_JSON_STRING, asc_tojson_PrologueResult)
{
    AscPlugin::PrologueResult result = {"prefixA", "devicePrefix"};

    std::string expectRes = "{\"DeviceStubPrefix\":\"devicePrefix\",\"OriginPrefix\":\"prefixA\"}";
    nlohmann::json j = result;
    std::string jsonStr = j.dump();
    EXPECT_EQ(expectRes, jsonStr);
}

TEST_F(TEST_ASC_JSON_STRING, asc_fromjson_KernelInfo)
{
    AscPlugin::Param a = {"int32_t", "fff", true, "abc", ""};
    AscPlugin::Param b = {"float", "abcd", false, "abcd", ""};
    AscPlugin::CoreRatio c = {false, 1, 1};
    AscPlugin::TemplateInstance inst1 = {{"A", "B", "C"}, {a, b}, "add_custom", "stub", c };
    AscPlugin::TemplateInstance inst2 = {{"D", "E", "F"}, {b, a}, "add_custom", "stub", c};
    AscPlugin::KernelInfo result = {"add_custom", "stub_add_custom", "prefix", "add.cpp", 72, 52,
        {"ascc", "ascplugin"}, {a, b}, {"global", "global"}, c, true, {b, b}, {inst1, inst2}};

    std::string expectRes = R"({"ColNum":52,"FileName":"add.cpp","IsTemplate":true,"KernelAttributes":["global","global"],"KernelMangledName":"stub_add_custom","KernelMangledNameConsiderPrefix":"prefix","KernelName":"add_custom","KernelParameters":[{"Attribute":"","DefaultValue":"abc","HasDefaultValue":true,"Name":"fff","Type":"int32_t","TypeClass":0},{"Attribute":"","DefaultValue":"abcd","HasDefaultValue":false,"Name":"abcd","Type":"float","TypeClass":0}],"LineNum":72,"Namespaces":["ascc","ascplugin"],"Ratio":{"CubeNum":1,"IsCoreRatio":false,"VecNum":1},"TemplateInstances":[{"InstanceKernelParameters":[{"Attribute":"","DefaultValue":"abc","HasDefaultValue":true,"Name":"fff","Type":"int32_t","TypeClass":0},{"Attribute":"","DefaultValue":"abcd","HasDefaultValue":false,"Name":"abcd","Type":"float","TypeClass":0}],"InstanceMangledName":"add_custom","InstanceMangledNameConsiderPrefix":"stub","Ratio":{"CubeNum":1,"IsCoreRatio":false,"VecNum":1},"TemplateInstantiationArguments":["A","B","C"]},{"InstanceKernelParameters":[{"Attribute":"","DefaultValue":"abcd","HasDefaultValue":false,"Name":"abcd","Type":"float","TypeClass":0},{"Attribute":"","DefaultValue":"abc","HasDefaultValue":true,"Name":"fff","Type":"int32_t","TypeClass":0}],"InstanceMangledName":"add_custom","InstanceMangledNameConsiderPrefix":"stub","Ratio":{"CubeNum":1,"IsCoreRatio":false,"VecNum":1},"TemplateInstantiationArguments":["D","E","F"]}],"TemplateParameters":[{"Attribute":"","DefaultValue":"abcd","HasDefaultValue":false,"Name":"abcd","Type":"float","TypeClass":0},{"Attribute":"","DefaultValue":"abcd","HasDefaultValue":false,"Name":"abcd","Type":"float","TypeClass":0}]})";

    const char* aa = expectRes.c_str();
    AscPlugin::KernelInfo configFromJsonStr;
    int32_t fromJsonRes = FromJson(configFromJsonStr, aa);

    EXPECT_EQ(configFromJsonStr.kernelName, result.kernelName);
    EXPECT_EQ(configFromJsonStr.kernelMangledName, result.kernelMangledName);
    EXPECT_EQ(configFromJsonStr.kernelMangledNameConsiderPrefix, result.kernelMangledNameConsiderPrefix);
    EXPECT_EQ(configFromJsonStr.fileName, result.fileName);
    EXPECT_EQ(configFromJsonStr.lineNum, result.lineNum);
    EXPECT_EQ(configFromJsonStr.colNum, result.colNum);
    EXPECT_EQ(configFromJsonStr.namespaces, result.namespaces);
}

TEST_F(TEST_ASC_JSON_STRING, asc_fromjson_KernelInfo_arg_missing)
{
    AscPlugin::Param a = {"int32_t", "fff", true, "abc", ""};
    AscPlugin::Param b = {"float", "abcd", false, "abcd", ""};
    AscPlugin::CoreRatio c = {false, 1, 1};
    AscPlugin::TemplateInstance inst1 = {{"A", "B", "C"}, {a, b}, "add_custom", "stub", c};
    AscPlugin::TemplateInstance inst2 = {{"D", "E", "F"}, {b, a}, "add_custom", "stub", c};
    AscPlugin::KernelInfo result = {"add_custom", "stub_add_custom", "prefix", "add.cpp", 72, 52,
        {"ascc", "ascplugin"}, {a, b}, {"global", "global"}, c, true, {b, b}, {inst1, inst2}};

    std::string expectRes = "{\"KernelAttributes\":[\"global\",\"global\"],\"KernelMangledName\":\"stub_add_custom\",\"KernelMangledNameConsiderPrefix\":\"prefix\",\"KernelName\":\"add_custom\",\"FileName\":\"add.cpp\",\"LineNum\":72,\"ColNum\":52,\"KernelParameters\":[{\"DefaultValue\":\"abc\",\"HasDefaultValue\":true,\"Name\":\"fff\",\"Type\":\"int32_t\"},{\"DefaultValue\":\"abcd\",\"HasDefaultValue\":false,\"Name\":\"abcd\",\"Type\":\"float\"}],\"Namespaces\":[\"ascc\",\"ascplugin\"],\"TemplateInstances\":[{\"InstanceKernelParameters\":[{\"DefaultValue\":\"abc\",\"HasDefaultValue\":true,\"Name\":\"fff\",\"Type\":\"int32_t\"},{\"DefaultValue\":\"abcd\",\"HasDefaultValue\":false,\"Name\":\"abcd\",\"Type\":\"float\"}],\"InstanceMangledName\":\"add_custom\",\"InstanceMangledNameConsiderPrefix\":\"stub\",\"TemplateInstantiationArguments\":[\"A\",\"B\",\"C\"]},{\"InstanceKernelParameters\":[{\"DefaultValue\":\"abcd\",\"HasDefaultValue\":false,\"Name\":\"abcd\",\"Type\":\"float\"},{\"DefaultValue\":\"abc\",\"HasDefaultValue\":true,\"Name\":\"fff\",\"Type\":\"int32_t\"}],\"InstanceMangledName\":\"add_custom\",\"InstanceMangledNameConsiderPrefix\":\"stub\",\"TemplateInstantiationArguments\":[\"D\",\"E\",\"F\"]}],\"TemplateParameters\":[{\"DefaultValue\":\"abcd\",\"HasDefaultValue\":false,\"Name\":\"abcd\",\"Type\":\"float\"},{\"DefaultValue\":\"abcd\",\"HasDefaultValue\":false,\"Name\":\"abcd\",\"Type\":\"float\"}]}";

    const char* aa = expectRes.c_str();
    AscPlugin::KernelInfo configFromJsonStr;
    int32_t fromJsonRes = FromJson(configFromJsonStr, aa);
    EXPECT_EQ(fromJsonRes, AscPlugin::ASC_JSONSTR_ARG_MISSING);
}

TEST_F(TEST_ASC_JSON_STRING, asc_tojson_EpilogueResult)
{
    AscPlugin::EpilogueResult a = {"", {"-DASCC"}, {"-DASCC_DEVICE"}, {"-DASCC_DEVICE"}};
    nlohmann::json j = a;
    std::string jsonStr = j.dump();
    std::string expectRes = "{\"DeviceCubeExtraCompileOptions\":[\"-DASCC_DEVICE\"],\"DeviceVecExtraCompileOptions\":[\"-DASCC_DEVICE\"],\"FunctionRegisterCode\":\"\",\"HostExtraCompileOptions\":[\"-DASCC\"]}";

    EXPECT_EQ(expectRes, jsonStr);
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_PrologueResult)
{
    AscPlugin::PrologueResult prologueResult;
    std::string tempFilePath = "asc_WriteFields_PrologueResult.txt";
    std::ofstream outFile(tempFilePath);

    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, prologueResult));
    outFile.close();
    remove(tempFilePath.c_str());
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_GenKernelResult)
{
    AscPlugin::GenKernelResult genKernelResult;
    std::string tempFilePath = "asc_WriteFields_GenKernelResult.txt";
    std::ofstream outFile(tempFilePath);

    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, genKernelResult));
    outFile.close();
    remove(tempFilePath.c_str());
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_PreCompileOptsResult)
{
    AscPlugin::PreCompileOptsResult preCompileOptsResult;
    preCompileOptsResult.compileOptions.emplace_back("-D1");
    preCompileOptsResult.compileOptions.emplace_back("-D2");
    std::string tempFilePath = "asc_WriteFields_PreCompileOptsResult.txt";
    std::ofstream outFile(tempFilePath);

    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, preCompileOptsResult));
    outFile.close();
    remove(tempFilePath.c_str());
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_EpilogueResult)
{
    AscPlugin::EpilogueResult epilogueResult;
    std::string tempFilePath = "asc_WriteFields_EpilogueResult.txt";
    std::ofstream outFile(tempFilePath);

    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, epilogueResult));
    outFile.close();
    remove(tempFilePath.c_str());
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_FatbinLinkResult)
{
    AscPlugin::FatbinLinkResult fatbinLinkResult;
    std::string tempFilePath = "asc_WriteFields_FatbinLinkResult.txt";
    std::ofstream outFile(tempFilePath);

    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, fatbinLinkResult));
    outFile.close();
    remove(tempFilePath.c_str());
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_GenKernelResult_AIC)
{
    AscPlugin::GenKernelResult genKernelResult;
    genKernelResult.type = AscPlugin::PluginKernelType::AIC;
    std::string tempFilePath = "asc_WriteFields_GenKernelResult.txt";
    std::ofstream outFile(tempFilePath);

    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, genKernelResult));
    outFile.close();
    remove(tempFilePath.c_str());
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_GenKernelResult_AIV)
{
    AscPlugin::GenKernelResult genKernelResult;
    genKernelResult.type = AscPlugin::PluginKernelType::AIV;
    std::string tempFilePath = "asc_WriteFields_GenKernelResult.txt";
    std::ofstream outFile(tempFilePath);

    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, genKernelResult));
    outFile.close();
    remove(tempFilePath.c_str());
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_GenKernelResult_MIX)
{
    AscPlugin::GenKernelResult genKernelResult;
    genKernelResult.type = AscPlugin::PluginKernelType::MIX;
    std::string tempFilePath = "asc_WriteFields_GenKernelResult.txt";
    std::ofstream outFile(tempFilePath);
    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, genKernelResult));
    outFile.close();
    remove(tempFilePath.c_str());
}

TEST_F(TEST_ASC_JSON_STRING, asc_WriteFields_GenKernelResult_UNKNOWN)
{
    AscPlugin::GenKernelResult genKernelResult;
    genKernelResult.type = static_cast<AscPlugin::PluginKernelType>(99);
    std::string tempFilePath = "asc_WriteFields_GenKernelResult.txt";
    std::ofstream outFile(tempFilePath);
    EXPECT_NO_THROW(AscPlugin::WriteFields(outFile, genKernelResult));
    outFile.close();
    remove(tempFilePath.c_str());
}