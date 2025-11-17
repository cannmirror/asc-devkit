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
#include <mockcpp/mockcpp.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <llvm/Support/JSON.h>
#define private public
#include "ascc_option.h"
#include "ascc_task_manager.h"
#include "task_executor.h"

using namespace testing;
using namespace Ascc;

class TEST_ASCC_TASK_MANAGER : public testing::Test {
protected:
    void SetUp()
    {
    }

    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

extern void SetUpCommonMocker();
extern void MockerProcessFileProcess();
extern int EntryAsccMain(int argc, char *argv[]);
namespace Ascc {
extern const std::string FindExecPath(std::string execName);
extern const std::string GetSocForSimulator(const std::string& socVersion);
extern Task TaskPreprocessDevice(
    const std::string &file, const PathInfo &pathInfo, const ArgInfo &argInfo, const CoreType coreType);
extern Task TaskPreprocessHost(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo);
extern AsccStatus TaskSharedLibrary(const std::vector<std::string>& objFile, const PathInfo& pathInfo,
    const ArgInfo& argInfo);
extern AsccStatus TaskExecutable(const std::vector<std::string>& objFile, const PathInfo& pathInfo,
    const ArgInfo& argInfo);
extern const std::vector<std::string> GetCompileOptionContent(const std::vector<std::string>& compileOptions);
extern const std::vector<std::string> GetDependencyCompileOptionContent(const std::string& filename);

extern AsccStatus TaskDeviceStubCompileO(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo,
    const CoreType coreType);

extern AsccStatus GetJsonInfo(std::unordered_map<uint8_t, std::vector<std::string>>& kernelType,
    std::unordered_map<uint8_t, uint32_t>& mixNumLineMap, std::unordered_map<uint8_t, uint32_t>& dumpUintLineMap,
    std::unordered_map<uint8_t, uint32_t>& dumpWorkspaceLineMap, std::unordered_map<uint8_t, uint32_t>& dumpSizeMap,
    std::unordered_map<uint8_t, std::string>& stubFileMap, const std::string& jsonName);

extern int64_t GetDeviceFileSize(const std::string& basePath, const std::string& suffix);

extern AsccStatus ReplaceStubFile(const std::string& fileName, CodeMode kernelTypeEnum, const uint32_t mixNumLineNo,
    const uint32_t dumpUintLineNo, const uint32_t dumpWorkspaceLineNo, const uint32_t dumpSize);
extern std::string GetDeviceStubMacro(const CoreType coreType, const std::string& mangleName,
    const CodeMode codeMode);
extern std::string GetDeviceStubFileName(CodeMode kernelTypeEnum, const CoreType coreType);
extern AsccStatus TaskHostStubCompileO(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo);
extern AsccStatus TaskMergeDeviceObjFiles(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo);
}

// ==============================================================================
TEST_F(TEST_ASCC_TASK_MANAGER, ascc_FindExecPath)
{
    auto res = FindExecPath("g++");
    EXPECT_EQ(!res.empty(), true);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetSocForSimulator)
{
    auto res = GetSocForSimulator("Ascend910B4-1");
    EXPECT_EQ(res, "Ascend910B4");
    res = GetSocForSimulator("Ascend910B1");
    EXPECT_EQ(res, "Ascend910B1");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetCompileOptionContent)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 7;
    char *argv[] = {(char *)"ascc",
        (char *)"--verbose", (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp", "--optimize=3",
        "--sanitizer"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);

    auto res = GetCompileOptionContent({"O"});
    std::vector<std::string> expect_res = {"-O3"};
    EXPECT_EQ(res, expect_res);

    res = GetCompileOptionContent({"sanitizer"});
    expect_res = {"--cce-enable-sanitizer"};
    EXPECT_EQ(res, expect_res);
    system("rm add_custom.cpp");
}

// with -MD + -MF + -MT
TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetCompileOptionContent_with_MF_MT)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 9;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp",
        "-MD", "-MF", "add.d", "-MT", "ssss"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);

    auto res = GetCompileOptionContent({"MD", "MMD", "MP", "MF", "MT"});
    const std::vector<std::string> expect_res = {"-MD", "-MF", "add.d", "-MT", "ssss"};
    EXPECT_EQ(res, expect_res);
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetAscendFeCompileOptionContent)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 9;
    char *argv[] = {(char *)"ascc",
        (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp",
        (char*)"-DGG", (char*)"--debug", (char*)"-sanitizer", (char*)"-Lfff", (char*)"-lsfsf"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);

    system("rm add_custom.cpp");
}

// only with -MD
TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetDependencyCompileOptionContent1)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 5;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp",
        "-MD"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);

    auto res = GetDependencyCompileOptionContent("add_custom.cpp");
    const std::vector<std::string> expect_res = {"-MD", "-MF", "add_custom.d", "-MT", "add_custom.o"};
    EXPECT_EQ(res, expect_res);
    system("rm add_custom.cpp");
}

// only with -MD + -o
TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetDependencyCompileOptionContent2)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 7;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp",
        "-MD", "-o", "fff.sf"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);

    auto res = GetDependencyCompileOptionContent("add_custom.cpp");
    const std::vector<std::string> expect_res = {"-MD", "-MF", "fff.d", "-MT", "fff.sf"};
    EXPECT_EQ(res, expect_res);
    system("rm add_custom.cpp");
}

// only with -MD + -MT + -MF + -o
TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetDependencyCompileOptionContent3)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 11;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp",
        "-MD", "-o", "fff.sf", "-MF", "abc", "-MT", "DEF"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);

    auto res = GetDependencyCompileOptionContent("add_custom.cpp");
    const std::vector<std::string> expect_res = {"-MD", "-MF", "abc", "-MT", "DEF"};
    EXPECT_EQ(res, expect_res);
    system("rm add_custom.cpp");
}

// without MD or MMD
TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetDependencyCompileOptionContent_noMDMMD)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 6;
    char *argv[] = {(char *)"ascc", (char *)"-arch" , (char*)"Ascend910B1", (char*)"add_custom.cpp",
        "-o", "fff.sf"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);

    auto res = GetDependencyCompileOptionContent("add_custom.cpp");
    const std::vector<std::string> expect_res = {};
    EXPECT_EQ(res, expect_res);
    system("rm add_custom.cpp");
}

// =============================
// 不同的task用例
TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskPreprocessDevice)
{
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    system("touch add_custom.cpp");

    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    EXPECT_NO_THROW(TaskPreprocessDevice("add_custom.cpp", pathInfo, argInfo, Ascc::CoreType::SPLIT_VEC));
    system("rm add_custom.cpp");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskPreprocessHost)
{
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    EXPECT_NO_THROW(TaskPreprocessHost("samples/add_custom.cpp", pathInfo, argInfo));
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskSharedLibrary)
{
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    std::vector<std::string> objFile = {"add_custom.o"};
    auto res = TaskSharedLibrary(objFile, pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskSharedLibrary_with_o)
{
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    argInfo.outputFileName = "a.out";
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    std::vector<std::string> objFile = {"add_custom.o"};
    auto res = TaskSharedLibrary(objFile, pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskExecutable)
{
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    std::vector<std::string> objFile = {"add_custom.o"};
    auto res = TaskExecutable(objFile, pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskExecutable_with_o)
{
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    argInfo.outputFileName = "a.out";
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    std::vector<std::string> objFile = {"add_custom.o"};
    auto res = TaskExecutable(objFile, pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}



TEST_F(TEST_ASCC_TASK_MANAGER, ascc_ProcessFiles_so)
{
    std::vector<std::string> files = {"add_custom.cpp"};
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    argInfo.outputFileName = "a.out";
    argInfo.outputMode = OutputFileType::FILE_SO;
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    auto res = ProcessFiles(files, pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_ProcessFiles_exec)
{
    std::vector<std::string> files = {"add_custom.cpp"};
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    argInfo.outputFileName = "a.out";
    argInfo.outputMode = OutputFileType::FILE_EXECUTABLE;
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    auto res = ProcessFiles(files, pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_ProcessFiles_o)
{
    std::vector<std::string> files = {"add_custom.cpp"};
    PathInfo pathInfo("bishengcc", "bisheng", "/usr/local/g++", "cann", "tmp", {});
    ArgInfo argInfo;
    argInfo.outputFileName = "a.out";
    argInfo.outputMode = OutputFileType::FILE_O;
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    auto res = ProcessFiles(files, pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}


TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetJsonInfo)
{
    const char* command = R"(
        echo '{
  "_Z25__device_stub__add_customPhS_S_": {
    "dump_size": 1048576,
    "dump_uint_lineno": 87101,
    "dump_workspace_lineno": 87315,
    "enable_dfx": false,
    "func_name": "add_custom",
    "kernel_type": 5,
    "mix_num_lineno": 94812,
    "stub_filename": "/root/c00636611/0627_Clang/AscendC-Clang/samples/single_src/add_custom/20250627154051_476593_476593/stub_files_device/add_custom/vec/stub_add_custom_cpp_vec_ii.cpp"
  }
}
' > data_test.json
    )";
    system(command);
    std::unordered_map<uint8_t, std::vector<std::string>> kernelType;
    std::unordered_map<uint8_t, uint32_t> mixNumLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpUintLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpWorkspaceLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpSizeMap;
    std::unordered_map<uint8_t, std::string> stubFileMap;

    auto res = GetJsonInfo(kernelType, mixNumLineMap, dumpUintLineMap, dumpWorkspaceLineMap, dumpSizeMap, stubFileMap,
        "data_test.json");
    EXPECT_EQ(res, Ascc::AsccStatus::FAILURE);    // no json parse function
    system("rm -rf data_test.json");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetDeviceFileSize)
{
    system("touch test.cpp");
    system("g++ test.cpp -c -o device_aic.o");
    std::string fileName = "./device_aic.o";
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    auto expected_res = static_cast<int64_t>(file.tellg());
    auto res = GetDeviceFileSize(".", "aic");
    EXPECT_EQ(res, expected_res);

    res = GetDeviceFileSize(".", "gg");
    EXPECT_EQ(res, 0);
    system("rm -rf test.cpp");
    system("rm -rf device_aic.o");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetCoreTypeStr)
{
    EXPECT_EQ(Ascc::GetCoreTypeStr(Ascc::CoreType::UNKNOWN), "");
    EXPECT_EQ(Ascc::GetCoreTypeStr(Ascc::CoreType::SPLIT_CUBE), "cube");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_ReplaceStubFile_MIX_1_1)
{
    std::string fileName = "test.txt";
    std::string content = R"(constexpr int32_t MIX_NUM = 0;\n
    constexpr size_t DUMP_UINTSIZE = 1000;\n
    const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * 1000;\n
    )";

    std::ofstream outFile(fileName);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file" << std::endl;
    }
    outFile << content;
    outFile.close();

    auto codeMode = CodeMode::KERNEL_TYPE_MIX_AIC_1_1;
    uint32_t dumpSize = 1024;
    auto res = ReplaceStubFile(fileName, codeMode, 1, 2, 3, dumpSize);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);

    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Cannot open file" << std::endl;
    }

    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "constexpr int32_t MIX_NUM = 1;");
    std::getline(file, line);
    EXPECT_EQ(line, std::string("constexpr size_t DUMP_UINTSIZE = 1024;"));
    std::getline(file, line);
    EXPECT_EQ(line, std::string("const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * 1024;"));
    file.close();
    system("rm -rf test.txt");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_ReplaceStubFile_MIX_1_2)
{
    std::string fileName = "test.txt";
    std::string content = R"(constexpr int32_t MIX_NUM = 0;\n
    constexpr size_t DUMP_UINTSIZE = 1000;\n
    const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * 1000;\n
    )";

    std::ofstream outFile(fileName);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file" << std::endl;
    }
    outFile << content;
    outFile.close();

    auto codeMode = CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
    uint32_t dumpSize = 1024;
    auto res = ReplaceStubFile(fileName, codeMode, 1, 2, 3, dumpSize);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);

    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Cannot open file" << std::endl;
    }

    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "constexpr int32_t MIX_NUM = 2;");
    std::getline(file, line);
    EXPECT_EQ(line, std::string("constexpr size_t DUMP_UINTSIZE = 1024;"));
    std::getline(file, line);
    EXPECT_EQ(line, std::string("const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * 1024;"));
    file.close();
    system("rm -rf test.txt");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetDeviceStubMacro)
{
    auto res = GetDeviceStubMacro(Ascc::CoreType::SPLIT_VEC, "mangleName", CodeMode::KERNEL_TYPE_MIX_AIC_1_1);
    EXPECT_EQ(res, "-DmangleName=mangleName_mix_aiv");
    res = GetDeviceStubMacro(Ascc::CoreType::SPLIT_VEC, "mangleName", CodeMode::KERNEL_TYPE_MIX_AIC_1_2);
    EXPECT_EQ(res, "-DmangleName=mangleName_mix_aiv");
    res = GetDeviceStubMacro(Ascc::CoreType::SPLIT_CUBE, "mangleName", CodeMode::KERNEL_TYPE_MIX_AIC_1_0);
    EXPECT_EQ(res, "-DmangleName=mangleName_mix_aic");
    res = GetDeviceStubMacro(Ascc::CoreType::SPLIT_CUBE, "mangleName", CodeMode::KERNEL_TYPE_MIX_AIV_1_0);
    EXPECT_EQ(res, "");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetDeviceStubFileName)
{
    auto res = GetDeviceStubFileName(CodeMode::KERNEL_TYPE_MIX_AIC_1_1, Ascc::CoreType::SPLIT_VEC);
    EXPECT_EQ(res, "device_stub_mix_1_1_aiv.cpp");
    res = GetDeviceStubFileName(CodeMode::KERNEL_TYPE_MIX_AIC_1_2, Ascc::CoreType::SPLIT_CUBE);
    EXPECT_EQ(res, "device_stub_mix_aic.cpp");
    res = GetDeviceStubFileName(CodeMode::KERNEL_TYPE_MIX_AIC_1_0, Ascc::CoreType::SPLIT_VEC);
    EXPECT_EQ(res, "device_stub_mix_aic.cpp");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskHostStubCompileO)
{
    MOCKER(Ascc::GetDeviceFileSize).stubs().will(returnValue(100));
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    PathInfo pathInfo;
    ArgInfo argInfo;
    auto res = TaskHostStubCompileO("add_custom.cpp", pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskMergeDeviceObjFiles)
{
    SetUpCommonMocker();
    MockerProcessFileProcess();
    system("touch add_custom.cpp");
    int argc = 4;
    char *argv[] = {(char *)"ascc",
        (char *)"--sanitizer", (char *)"-arch=Ascend910B1", (char*)"add_custom.cpp"};
    int result = EntryAsccMain(argc, argv);
    EXPECT_EQ(result, 0);
    system("rm add_custom.cpp");

    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    PathInfo pathInfo;
    ArgInfo argInfo;
    auto res = TaskMergeDeviceObjFiles("add_custom.cpp", pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskPackKernel)
{
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    PathInfo pathInfo;
    ArgInfo argInfo;
    argInfo.outputMode = OutputFileType::FILE_O;
    argInfo.outputFileName = "fffff";
    std::vector<std::string> objFile;
    auto res = TaskPackKernel(objFile, "add_custom.cpp", pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_ProcessFiles)
{
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::EXIST));
    MOCKER(Ascc::HasJsonInStubDir).stubs().will(returnValue(true));
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    MOCKER(&Ascc::TaskDeviceStubCompileO).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    PathInfo pathInfo;
    ArgInfo argInfo;
    argInfo.outputMode = OutputFileType::FILE_O;
    argInfo.outputFileName = "fffff";
    std::vector<std::string> files = {"add_custom.cpp"};
    auto res = ProcessFiles(files, pathInfo, argInfo);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_GetJsonInfo_cannot_open)
{
    std::unordered_map<uint8_t, std::vector<std::string>> kernelType;
    std::unordered_map<uint8_t, uint32_t> mixNumLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpUintLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpWorkspaceLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpSizeMap;
    std::unordered_map<uint8_t, std::string> stubFileMap;
    const std::string jsonName = "a.json";
    std::ofstream file(jsonName);
    file.close();
    chmod(jsonName.c_str(), 0000);

    auto res = GetJsonInfo(kernelType, mixNumLineMap, dumpUintLineMap, dumpWorkspaceLineMap, dumpSizeMap, stubFileMap,
        jsonName);
    EXPECT_EQ(res, Ascc::AsccStatus::FAILURE);
}

Ascc::AsccStatus GetJsonInfoStub(std::unordered_map<uint8_t, std::vector<std::string>>& kernelType,
    std::unordered_map<uint8_t, uint32_t>& mixNumLineMap, std::unordered_map<uint8_t, uint32_t>& dumpUintLineMap,
    std::unordered_map<uint8_t, uint32_t>& dumpWorkspaceLineMap, std::unordered_map<uint8_t, uint32_t>& dumpSizeMap,
    std::unordered_map<uint8_t, std::string>& stubFileMap, const std::string& jsonName)
{
    kernelType[2] = {"mangle"};
    mixNumLineMap[2] = 0;
    dumpUintLineMap[2] = 0;
    dumpWorkspaceLineMap[2] = 0;
    dumpSizeMap[2] = 0;
    stubFileMap[2] = "stubFile";
    return Ascc::AsccStatus::SUCCESS;
}

Ascc::AsccStatus GetJsonInfoStubCube(std::unordered_map<uint8_t, std::vector<std::string>>& kernelType,
    std::unordered_map<uint8_t, uint32_t>& mixNumLineMap, std::unordered_map<uint8_t, uint32_t>& dumpUintLineMap,
    std::unordered_map<uint8_t, uint32_t>& dumpWorkspaceLineMap, std::unordered_map<uint8_t, uint32_t>& dumpSizeMap,
    std::unordered_map<uint8_t, std::string>& stubFileMap, const std::string& jsonName)
{
    kernelType[1] = {"mangle"};
    mixNumLineMap[1] = 0;
    dumpUintLineMap[1] = 0;
    dumpWorkspaceLineMap[1] = 0;
    dumpSizeMap[1] = 0;
    stubFileMap[1] = "stubFile";
    return Ascc::AsccStatus::SUCCESS;
}


TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskDeviceStubCompileO)
{
    MOCKER(Ascc::GetJsonInfo, Ascc::AsccStatus (*)(std::unordered_map<unsigned char, std::vector<std::basic_string<char> > >&, std::unordered_map<unsigned char, unsigned int>&, std::unordered_map<unsigned char, unsigned int>&, std::unordered_map<unsigned char, unsigned int>&, std::unordered_map<unsigned char, unsigned int>&, std::unordered_map<unsigned char, std::basic_string<char> >&, const std::string&)).stubs().will(invoke(GetJsonInfoStub));
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    system("touch stubFile");
    PathInfo pathInfo;
    ArgInfo argInfo;
    argInfo.outputMode = OutputFileType::FILE_O;
    argInfo.outputFileName = "fffff";
    auto res = TaskDeviceStubCompileO("add_custom.cpp", pathInfo, argInfo, Ascc::CoreType::SPLIT_VEC);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
    res = TaskDeviceStubCompileO("add_custom.cpp", pathInfo, argInfo, Ascc::CoreType::SPLIT_CUBE);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
    system("rm -rf stubFile");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskDeviceStubCompileO_no_json)
{
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    system("touch stubFile");
    PathInfo pathInfo;
    ArgInfo argInfo;
    argInfo.outputMode = OutputFileType::FILE_O;
    argInfo.outputFileName = "fffff";
    auto res = TaskDeviceStubCompileO("add_custom.cpp", pathInfo, argInfo, Ascc::CoreType::SPLIT_VEC);
    EXPECT_EQ(res, Ascc::AsccStatus::FAILURE);
    res = TaskDeviceStubCompileO("add_custom.cpp", pathInfo, argInfo, Ascc::CoreType::SPLIT_CUBE);
    EXPECT_EQ(res, Ascc::AsccStatus::FAILURE);
    system("rm -rf stubFile");
}

TEST_F(TEST_ASCC_TASK_MANAGER, ascc_TaskDeviceStubCompileO_Cube)
{
    MOCKER(Ascc::GetJsonInfo, Ascc::AsccStatus (*)(std::unordered_map<unsigned char, std::vector<std::basic_string<char> > >&, std::unordered_map<unsigned char, unsigned int>&, std::unordered_map<unsigned char, unsigned int>&, std::unordered_map<unsigned char, unsigned int>&, std::unordered_map<unsigned char, unsigned int>&, std::unordered_map<unsigned char, std::basic_string<char> >&, const std::string&)).stubs().will(invoke(GetJsonInfoStubCube));
    MOCKER(&Ascc::TaskExecutor::ExecuteTasks).stubs().will(returnValue(true));
    system("touch stubFile");
    PathInfo pathInfo;
    ArgInfo argInfo;
    argInfo.outputMode = OutputFileType::FILE_O;
    argInfo.outputFileName = "fffff";
    auto res = TaskDeviceStubCompileO("add_custom.cpp", pathInfo, argInfo, Ascc::CoreType::SPLIT_CUBE);
    EXPECT_EQ(res, Ascc::AsccStatus::SUCCESS);
    system("rm -rf stubFile");
}