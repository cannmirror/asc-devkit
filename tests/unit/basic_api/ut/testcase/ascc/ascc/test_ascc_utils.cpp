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
#include <fstream>
#define private public
#include "ascc_utils.h"


using namespace testing;
using namespace Ascc;

class TEST_ASCC_UTILS : public testing::Test {
protected:
    void SetUp()
    {
    }

    void TearDown()
    {
        GlobalMockObject::verify();
    }
};


// ==============================================================================
TEST_F(TEST_ASCC_UTILS, ascc_PathCheck_read_permission)
{
    // 创建一个只读的临时文件
    std::string tempFile = "test_path_check.txt";
    std::ofstream file(tempFile);
    file.close();
    chmod(tempFile.c_str(), 0444); // 设置只读权限

    EXPECT_EQ(PathCheck(tempFile.c_str(), true), PathStatus::READ);

    // 清理临时文件
    remove(tempFile.c_str());
}

TEST_F(TEST_ASCC_UTILS, ascc_PathCheck_exist) {
    // 创建一个不可读不可写的文件
    std::string tempFile = "test_exist.txt";
    std::ofstream file(tempFile);
    file.close();
    chmod(tempFile.c_str(), 0000);

    EXPECT_EQ(PathCheck(tempFile.c_str(), true), PathStatus::EXIST);

    remove(tempFile.c_str());
}

TEST_F(TEST_ASCC_UTILS, ascc_PathCheck_not_exist) {
    std::string nonExistentPath = "non_existent_file.txt";
    EXPECT_EQ(PathCheck(nonExistentPath.c_str(), true), PathStatus::NOT_EXIST);
    EXPECT_EQ(PathCheck(nonExistentPath.c_str(), false), PathStatus::NOT_EXIST);
}

TEST_F(TEST_ASCC_UTILS, ascc_CheckAndGetFullPath) {
    std::string tempFile = "test_exist.txt";
    MOCKER(PathCheck).stubs().will(returnValue(PathStatus::EXIST));
    EXPECT_EQ(CheckAndGetFullPath(tempFile.c_str()).empty(), true);
}

TEST_F(TEST_ASCC_UTILS, ascc_GetFilePath) {
    std::string tempFile = "samples/test_exist.txt";
    EXPECT_EQ(GetFilePath(tempFile).empty(), true);
}

TEST_F(TEST_ASCC_UTILS, ascc_GetCurrentDirectory) {
    char* a = nullptr;
    MOCKER(getcwd).stubs().will(returnValue(a));
    EXPECT_EQ(GetCurrentDirectory().empty(), true);
}

TEST_F(TEST_ASCC_UTILS, ascc_ExecMkdir) {
    struct stat fileStatus;
    std::string a = "test_path";
    MOCKER(stat).stubs().will(returnValue(1));
    MOCKER(mkdir).stubs().will(returnValue(1));
    EXPECT_EQ(ExecMkdir(fileStatus, a.c_str()), AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_UTILS, ascc_ExecMkdir_exists) {
    struct stat fileStatus;
    std::string a = "test_path";
    system("mkdir test_path");
    MOCKER(stat).stubs().will(returnValue(1));
    EXPECT_EQ(ExecMkdir(fileStatus, a.c_str()), AsccStatus::SUCCESS);
    system("rm -rf test_path");
}

TEST_F(TEST_ASCC_UTILS, ascc_CreateDirectory) {
    MOCKER(ExecMkdir).stubs().will(returnValue(AsccStatus::SUCCESS));
    EXPECT_EQ(CreateDirectory("//ff"), AsccStatus::SUCCESS);
}
