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
#include <stdexcept>
#define private public
#include "asc_utils.h"

class TEST_ASC_UTILS : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_UTILS, asc_plugin_CheckAndGetFullPath)
{
    system("rm test_exist.txt");
    std::string tempFile = "test_exist.txt";
    MOCKER(AscPlugin::PathCheck).stubs().will(returnValue(AscPlugin::PathStatus::EXIST));
    EXPECT_EQ(AscPlugin::CheckAndGetFullPath(tempFile.c_str()).empty(), true);
    EXPECT_EQ(AscPlugin::CheckAndGetFullPath(tempFile).empty(), true);
}

TEST_F(TEST_ASC_UTILS, asc_plugin_CheckAndGetFullPath_noFile)
{
    std::string tempFile = "a.cpp";
    EXPECT_EQ(AscPlugin::CheckAndGetFullPath(tempFile.c_str()), "");
}

TEST_F(TEST_ASC_UTILS, asc_plugin_CheckAndGetFullPath_hasFile)
{
    system("touch a.cpp");
    std::string tempFile = "a.cpp";
    auto res = AscPlugin::CheckAndGetFullPath(tempFile.c_str());
    auto subStr = res.substr(res.length()-5, res.length());
    EXPECT_EQ(subStr, "a.cpp");
    system("rm a.cpp");
}

TEST_F(TEST_ASC_UTILS, asc_PathCheck_read_permission)
{
    // 创建一个只读的临时文件
    std::string tempFile = "test_path_check.txt";
    std::ofstream file(tempFile);
    file.close();
    chmod(tempFile.c_str(), 0444); // 设置只读权限
    EXPECT_EQ(AscPlugin::PathCheck(tempFile.c_str(), true), AscPlugin::PathStatus::READ);
    // 清理临时文件
    remove(tempFile.c_str());
}

TEST_F(TEST_ASC_UTILS, asc_PathCheck_exist) {
    // 创建一个不可读不可写的文件
    std::string tempFile = "test_exist.txt";
    std::ofstream file(tempFile);
    file.close();
    chmod(tempFile.c_str(), 0000);

    EXPECT_EQ(AscPlugin::PathCheck(tempFile.c_str(), true), AscPlugin::PathStatus::EXIST);

    remove(tempFile.c_str());
}

TEST_F(TEST_ASC_UTILS, asc_PathCheck_not_exist) {
    std::string nonExistentPath = "non_existent_file.txt";
    EXPECT_EQ(AscPlugin::PathCheck(nonExistentPath.c_str(), true), AscPlugin::PathStatus::NOT_EXIST);
    EXPECT_EQ(AscPlugin::PathCheck(nonExistentPath.c_str(), false), AscPlugin::PathStatus::NOT_EXIST);
}

TEST_F(TEST_ASC_UTILS, asc_GetFilePath_with_slash) {
    system("mkdir test_asc");
    system("touch test_asc/test_exist.txt");
    std::string tempFile = "test_asc/test_exist.txt";
    EXPECT_EQ(AscPlugin::GetFilePath(tempFile).empty(), false);
    system("rm -rf test_asc");
}

TEST_F(TEST_ASC_UTILS, asc_GetFilePath_without_slash) {
    std::string tempFile = "test_exist.txt";
    system("touch test_exist.txt");
    EXPECT_EQ(AscPlugin::GetFilePath(tempFile).empty(), false);
    system("rm -rf test_exist.txt");
}

TEST_F(TEST_ASC_UTILS, asc_GetFilePath_without_slash_nullptr) {
    std::string tempFile = "test_exist.txt";
    system("touch test_exist.txt");
    char* a = nullptr;
    MOCKER(getcwd).stubs().will(returnValue(a));
    EXPECT_EQ(AscPlugin::GetFilePath(tempFile).empty(), true);
    system("rm -rf test_exist.txt");
}

TEST_F(TEST_ASC_UTILS, asc_StartsWith) {
    EXPECT_EQ(AscPlugin::StartsWith("ABC", "ABCD"), false);
    EXPECT_EQ(AscPlugin::StartsWith("ABCD", "ABC"), true);
}

TEST_F(TEST_ASC_UTILS, asc_GetTempFolder) {
    std::string op_path_1 = "add_custom.cpp";
    std::string op_path_2 = "/home/xpu_ops/add_custom.cpp";
    std::string op_path_3 = "./xpu_ops/add_custom.cpp";
    std::string timeStamp = AscPlugin::GenerateTimestamp();
    std::string tempFolder1 = AscPlugin::GetTempFolder("A", op_path_1, timeStamp, "temp");
    std::string tempFolder2 = AscPlugin::GetTempFolder("A", op_path_2, timeStamp, "temp");
    std::string tempFolder3 = AscPlugin::GetTempFolder("A", op_path_3, timeStamp, "log");
    EXPECT_EQ(tempFolder1.find(timeStamp) != std::string::npos, true);
    EXPECT_EQ(tempFolder2.find(timeStamp) != std::string::npos, true);
    EXPECT_EQ(tempFolder3.find(timeStamp) != std::string::npos, true);
}

TEST_F(TEST_ASC_UTILS, asc_ExecMkdir) {
    struct stat fileStatus;
    std::string a = "test_path";
    MOCKER(stat).stubs().will(returnValue(1));
    MOCKER(mkdir).stubs().will(returnValue(1));
    EXPECT_EQ(AscPlugin::ExecMkdir(fileStatus, a.c_str()), AscPlugin::ASC_FAILURE);
}

TEST_F(TEST_ASC_UTILS, asc_ExecMkdir_exists) {
    struct stat fileStatus;
    std::string a = "test_path";
    system("mkdir test_path");
    MOCKER(stat).stubs().will(returnValue(1));
    EXPECT_EQ(AscPlugin::ExecMkdir(fileStatus, a.c_str()), AscPlugin::ASC_SUCCESS);
    system("rm -rf test_path");
}

TEST_F(TEST_ASC_UTILS, asc_CreateDirectory) {
    MOCKER(AscPlugin::ExecMkdir).stubs().will(returnValue(AscPlugin::ASC_SUCCESS));
    EXPECT_EQ(AscPlugin::CreateDirectory("//ff"), AscPlugin::ASC_SUCCESS);
}

TEST_F(TEST_ASC_UTILS, asc_CreateDirectory_failed_1) {
    MOCKER(AscPlugin::ExecMkdir).stubs().will(returnValue(AscPlugin::ASC_FAILURE));
    EXPECT_EQ(AscPlugin::CreateDirectory("/tmp/asc_plugin"), AscPlugin::ASC_FAILURE);
}

TEST_F(TEST_ASC_UTILS, asc_CreateDirectory_failed_2) {
    MOCKER(AscPlugin::ExecMkdir).stubs().will(returnValue(AscPlugin::ASC_FAILURE));
    EXPECT_EQ(AscPlugin::CreateDirectory("/"), AscPlugin::ASC_FAILURE);
}

TEST_F(TEST_ASC_UTILS, asc_CreateDirectory_empty) {
    EXPECT_EQ(AscPlugin::CreateDirectory(""), AscPlugin::ASC_FAILURE);
}

TEST_F(TEST_ASC_UTILS, asc_GenerateTimestamp) {
    MOCKER(strftime).stubs().will(returnValue((unsigned long)0));
    EXPECT_EQ(AscPlugin::GenerateTimestamp(), "");
}

TEST_F(TEST_ASC_UTILS, asc_ExecuteCommand_empty) {
    std::pair<std::string, int32_t> res = {"", -1};
    EXPECT_EQ(AscPlugin::ExecuteCommand(nullptr), res);
}

TEST_F(TEST_ASC_UTILS, asc_ExecuteCommand_popen_failed) {
    std::pair<std::string, int32_t> res = {"", -1};
    MOCKER(popen).stubs().will(returnValue((FILE*)nullptr));
    EXPECT_EQ(AscPlugin::ExecuteCommand("/nonexistent/command_that_never_existsd"), res);
}

TEST_F(TEST_ASC_UTILS, asc_ExecuteCommand_ferror) {
    std::pair<std::string, int32_t> res = {"", -1};
    MOCKER(ferror).stubs().will(returnValue(1));
    EXPECT_EQ(AscPlugin::ExecuteCommand("abort"), res);
}

TEST_F(TEST_ASC_UTILS, asc_ExecuteCommand_pclose_failed) {
    std::pair<std::string, int32_t> res = {"", -1};
    MOCKER(pclose).stubs().will(returnValue(-1));
    EXPECT_EQ(AscPlugin::ExecuteCommand("abort"), res);
}

TEST_F(TEST_ASC_UTILS, asc_ExecuteCommand_singaled) {
    std::pair<std::string, int32_t> res = {"", -1};
    EXPECT_EQ(AscPlugin::ExecuteCommand("bash -c 'kill -SEGV $$'"), res);
}

TEST_F(TEST_ASC_UTILS, asc_ExecuteCommand_not_normal) {
    std::pair<std::string, int32_t> res = {"", -1};
    MOCKER(pclose).stubs().will(returnValue(0x7F));
    EXPECT_EQ(AscPlugin::ExecuteCommand("dummy_command"), res);
}

TEST_F(TEST_ASC_UTILS, asc_ExecuteCompile_code_not_zero) {
    std::pair<std::string, int32_t> res = {"", -1};
    MOCKER(AscPlugin::ExecuteCommand).stubs().will(returnValue(res));
    EXPECT_EQ(AscPlugin::ExecuteCompile("test cmd"), AscPlugin::ASC_FAILURE);
}