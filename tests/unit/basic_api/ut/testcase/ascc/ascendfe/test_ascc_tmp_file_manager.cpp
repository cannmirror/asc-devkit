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
#include <cstdlib>
#define private public
#include "ascc_tmp_file_manager.h"

using namespace testing;
using namespace Ascc;

class TEST_ASCC_TMP_FILE_MANAGER : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCC_TMP_FILE_MANAGER, Init) {
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    std::string inputFile = "hello_world.cpp";
    const auto& tmpFileManager = Ascc::AsccTmpFileManager::GetInstance();
    EXPECT_EQ(tmpFileManager.Init(inputFile), AsccStatus::SUCCESS);
    system("rm -rf /tmp/ascc");
}

TEST_F(TEST_ASCC_TMP_FILE_MANAGER, Init_fail) {
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    std::string inputFile = "hello_world.cpp";
    const auto& tmpFileManager = Ascc::AsccTmpFileManager::GetInstance();
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(false));
    EXPECT_EQ(tmpFileManager.Init(inputFile), AsccStatus::FAILURE);
    system("rm -rf /tmp/ascc");
}

TEST_F(TEST_ASCC_TMP_FILE_MANAGER, Init_fail_2) {
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "/tmp/ascc";
    std::string inputFile = "hello_world.cpp";
    const auto& tmpFileManager = Ascc::AsccTmpFileManager::GetInstance();
    MOCKER(Ascc::IsPathLegal).stubs().will(returnValue(false));
    MOCKER(Ascc::IsParentDirValid).stubs().will(returnValue(false));
    EXPECT_EQ(tmpFileManager.Init(inputFile), AsccStatus::FAILURE);
    system("rm -rf /tmp/ascc");
}

TEST_F(TEST_ASCC_TMP_FILE_MANAGER, GenerateStubFiles) {
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = "./";
    const auto& tmpFileManager = Ascc::AsccTmpFileManager::GetInstance();
    EXPECT_EQ(tmpFileManager.GenerateStubFiles(envVar.asccTmpPath ), AsccStatus::SUCCESS);
}