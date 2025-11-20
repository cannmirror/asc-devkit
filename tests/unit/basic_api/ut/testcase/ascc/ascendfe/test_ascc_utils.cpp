/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <fstream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>
#define private public
#include "ascc_utils.h"
#include "ascc_global_env_manager.h"
#include "ascc_tmp_file_manager.h"
#include "ascc_types.h"

using namespace Ascc;

class TEST_ASCC_UTILS : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

// 测试 PathCheck 函数
TEST_F(TEST_ASCC_UTILS, WritePermission) {
    const char* testPath = "/tmp/test_file";
    std::ofstream file(testPath);
    file.close();
    EXPECT_EQ(PathCheck(testPath, true), PathStatus::WRITE);
    remove(testPath);
}

TEST_F(TEST_ASCC_UTILS, ReadPermission) {
    // 创建一个只读的临时文件
    std::string tempFile = "test_path_check.txt";
    std::ofstream file(tempFile);
    file.close();
    chmod(tempFile.c_str(), 0444); // 设置只读权限

    EXPECT_EQ(PathCheck(tempFile.c_str(), true), PathStatus::READ);

    // 清理临时文件
    remove(tempFile.c_str());
}

TEST_F(TEST_ASCC_UTILS, ExistPermission) {
    // 创建一个不可读不可写的文件
    std::string tempFile = "test_exist.txt";
    std::ofstream file(tempFile);
    file.close();
    chmod(tempFile.c_str(), 0000);

    EXPECT_EQ(PathCheck(tempFile.c_str(), true), PathStatus::EXIST);

    remove(tempFile.c_str());
}

TEST_F(TEST_ASCC_UTILS, NotExistPermission) {
    std::string nonExistentPath = "non_existent_file.txt";
    EXPECT_EQ(PathCheck(nonExistentPath.c_str(), true), PathStatus::NOT_EXIST);
    EXPECT_EQ(PathCheck(nonExistentPath.c_str(), false), PathStatus::NOT_EXIST);
}

// 测试 CheckAndGetFullPath 函数
TEST_F(TEST_ASCC_UTILS, ValidPath) {
    const char* validPath1 = ".";
    const std::string validPath2 = ".";
    std::string result1 = CheckAndGetFullPath(validPath1);
    std::string result2 = CheckAndGetFullPath(validPath2);
    EXPECT_FALSE(result1.empty());
    EXPECT_FALSE(result2.empty());
}

TEST_F(TEST_ASCC_UTILS, InvalidPath) {
    const char* invalidPath1 = "non_existent_dir/non_existent_file.txt";
    const char* invalidPath2 = "";
    const std::string invalidPath3 = "non_existent_dir/non_existent_file.txt";
    std::string result1 = CheckAndGetFullPath(invalidPath1);
    std::string result2 = CheckAndGetFullPath(invalidPath2);
    std::string result3 = CheckAndGetFullPath(invalidPath3);
    EXPECT_TRUE(result1.empty());
    EXPECT_TRUE(result2.empty());
    EXPECT_TRUE(result3.empty());
}

#ifdef TEST_GETCWD_FAILURE
extern "C" {
    char *getcwd(char *buf, size_t size) {
        (void)buf;
        (void)size;
        errno = ERANGE;
        return nullptr;
    }
}
#endif

// 测试 GetCurrentDirectory 函数
TEST_F(TEST_ASCC_UTILS, CurrentDirectory) {
    std::string currentDir = GetCurrentDirectory();
    EXPECT_FALSE(currentDir.empty());
}

TEST_F(TEST_ASCC_UTILS, CurrentDirectoryEmpty) {
    #define TEST_GETCWD_FAILURE
    std::string result = GetCurrentDirectory();
    #undef TEST_GETCWD_FAILURE
    EXPECT_FALSE(result.empty());
}

// 测试 GetFileName 函数
TEST_F(TEST_ASCC_UTILS, NormalPath) {
    std::string filePath = "./test_exist.txt";
    std::string fileName = GetFileName(filePath);
    EXPECT_EQ(fileName, "test_exist.txt");
}

TEST_F(TEST_ASCC_UTILS, WrongPath) {
    std::string filePath = "test_exist.txt";
    std::string fileName = GetFileName(filePath);
    EXPECT_EQ(fileName, "test_exist.txt");
}

TEST_F(TEST_ASCC_UTILS, NoExtensionFile) {
    std::string filePath = "/home/user/test";
    std::string fileName = GetFileName(filePath);
    EXPECT_EQ(fileName, "test");
}

// 测试 GetFilePath 函数
TEST_F(TEST_ASCC_UTILS, AbsolutePath) {
    std::string filePath = "./test_exist.txt";
    std::string currentDir = GetCurrentDirectory();
    std::string filePathResult = GetFilePath(filePath);
    EXPECT_EQ(filePathResult, currentDir);
}

TEST_F(TEST_ASCC_UTILS, WrongAbsolutePath) {
    Ascc::AsccGlobalEnvManager::GetInstance().currentPath = "";
    std::string filePath = "test_exist.txt";
    std::string filePathResult = GetFilePath(filePath);
    EXPECT_EQ(filePathResult, "");
}

// 测试 RemoveSuffix 函数
TEST_F(TEST_ASCC_UTILS, WithExtension) {
    std::string fileName = "test.txt";
    std::string result = RemoveSuffix(fileName);
    EXPECT_EQ(result, "test");
}

TEST_F(TEST_ASCC_UTILS, NoExtension) {
    std::string fileName = "test";
    std::string result = RemoveSuffix(fileName);
    EXPECT_EQ(result, "test");
}

// 测试 ToUpper 函数
TEST_F(TEST_ASCC_UTILS, Lowercase) {
    std::string str = "hello";
    std::string result = ToUpper(str);
    EXPECT_EQ(result, "HELLO");
}

TEST_F(TEST_ASCC_UTILS, MixedCase) {
    std::string str = "HeLlO";
    std::string result = ToUpper(str);
    EXPECT_EQ(result, "HELLO");
}

// 测试 SaveCompileLogFile 函数
TEST_F(TEST_ASCC_UTILS, SaveCompileLogFileSuccess) {
    // 创建临时目录和子目录
    const std::string tempDir = "/tmp";
    const std::string compileLogDir = tempDir + "/compile_log";
    const std::string logFile = compileLogDir + "/compile.log";
    system("mkdir -p /tmp/compile_log");
    system("touch /tmp/compile_log/compile.log");

    // 设置临时环境变量
    AsccGlobalEnvManager& envVar = AsccGlobalEnvManager::GetInstance();
    std::string originalTmpPath = envVar.asccTmpPath;
    envVar.asccTmpPath = tempDir;

    const std::string note = "Test note";
    const std::string content = "Test content";

    SaveCompileLogFile(note, content);

    // 检查文件内容
    std::ifstream file(logFile);
    std::string readNote, readContent;
    std::getline(file, readNote);
    std::getline(file, readContent);
    EXPECT_EQ(readNote, "# " + note);
    EXPECT_EQ(readContent, content);

    // 清理临时文件和目录
    unlink(logFile.c_str());
    rmdir(compileLogDir.c_str());
    rmdir(tempDir.c_str());

    // 恢复环境变量
    envVar.asccTmpPath = originalTmpPath;
}

TEST_F(TEST_ASCC_UTILS, SaveCompileLogFileFailNoSuchDirectory) {
    // 设置无效的临时路径
    AsccGlobalEnvManager& envVar = AsccGlobalEnvManager::GetInstance();
    std::string originalTmpPath = envVar.asccTmpPath;
    envVar.asccTmpPath = "non_existent_dir";

    const std::string note = "Test note";
    const std::string content = "Test content";

    SaveCompileLogFile(note, content);

    // 恢复环境变量
    envVar.asccTmpPath = originalTmpPath;
}

// 测试 HandleError 函数
TEST_F(TEST_ASCC_UTILS, HandleErrorSuccess) {
    const std::string errorMessage = "Test error message";

    // 捕获 cerr 的输出
    std::stringstream cerrBuffer;
    std::streambuf* originalCerrBuf = std::cerr.rdbuf(cerrBuffer.rdbuf());

    HandleError(errorMessage);

    std::string output = cerrBuffer.str();
    EXPECT_NE(output.find("[ERROR] " + errorMessage + "\n"), std::string::npos);

    // 恢复 cerr 的原始缓冲区
    std::cerr.rdbuf(originalCerrBuf);
}

TEST_F(TEST_ASCC_UTILS, SuccessfulCommand) {
    std::pair<std::string, int> result = ExecuteCommand("echo \"Hello, World!\"");
    EXPECT_EQ(result.second, 0);
    EXPECT_NE(result.first.find("Hello, World!"), std::string::npos);
}

TEST_F(TEST_ASCC_UTILS, FailedCommand) {
    std::pair<std::string, int> result = ExecuteCommand("ls non_existent_file 2>&1");
    EXPECT_NE(result.second, 0);
    EXPECT_NE(result.first.find(""), std::string::npos);
}

TEST_F(TEST_ASCC_UTILS, InvalidCommand) {
    std::pair<std::string, int> result = ExecuteCommand("nonexistentcommand 2>&1");
    EXPECT_NE(result.second, 0);
    EXPECT_NE(result.first.find(""), std::string::npos);
}

TEST_F(TEST_ASCC_UTILS, ascendFe_ExecMkdir_exists) {
    struct stat fileStatus;
    std::string a = "test_path";
    system("mkdir test_path");
    MOCKER(stat).stubs().will(returnValue(1));
    const auto& tmpFileManager = Ascc::AsccTmpFileManager::GetInstance();
    EXPECT_EQ(tmpFileManager.ExecMkdir(fileStatus, a.c_str()), AsccStatus::SUCCESS);
    system("rm -rf test_path");
}

TEST_F(TEST_ASCC_UTILS, ascendFe_ExecMkdir_failed) {
    struct stat fileStatus;
    std::string a = "ff/test_path";
    MOCKER(stat).stubs().will(returnValue(1));
    const auto& tmpFileManager = Ascc::AsccTmpFileManager::GetInstance();
    EXPECT_EQ(tmpFileManager.ExecMkdir(fileStatus, a.c_str()), AsccStatus::FAILURE);
}


#ifdef TEST_POPEN_FAILURE
extern "C" {
    FILE *popen(const char *, const char *) {
        errno = ENOENT; // "No such file or directory"
        return nullptr;
    }
}
#endif

#ifdef TEST_PCLOSE_FAILURE
extern "C" {
    int pclose(FILE *) {
        return -1;
    } // 占位，实际不会执行
}
#endif

#ifdef TEST_NORMALLY_FAILURE
extern "C" {
    int pclose(FILE *stream) {
        // 返回一个既非正常退出、也非信号终止的状态码
        return 0x7F; // 假设对应未知状态
    }
}
#endif

TEST_F(TEST_ASCC_UTILS, FailedReturnCommand) {
    #define TEST_POPEN_FAILURE
    std::pair<std::string, int> result1 = ExecuteCommand("invalid_command");
    EXPECT_EQ(result1.first, "");
    EXPECT_EQ(result1.second, 127);
    #undef TEST_POPEN_FAILURE

    #define TEST_PCLOSE_FAILURE
    std::pair<std::string, int> result2 = ExecuteCommand("any_command");
    EXPECT_EQ(result2.first, "");
    EXPECT_EQ(result2.second, 127);
    #undef TEST_PCLOSE_FAILURE

    std::pair<std::string, int> result3 = ExecuteCommand("sleep 0.5; kill -9 $$");
    EXPECT_EQ(result3.first, "");
    EXPECT_EQ(result3.second, -1);

    #define TEST_NORMALLY_FAILURE
    std::pair<std::string, int> result4 = ExecuteCommand("dummy_command");
    EXPECT_EQ(result4.first, "");
    EXPECT_EQ(result4.second, 127);
    #undef TEST_NORMALLY_FAILURE
}
