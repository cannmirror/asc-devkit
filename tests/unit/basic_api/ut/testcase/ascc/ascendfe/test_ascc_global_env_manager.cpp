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
#include "ascc_log.h"
#include "ascc_utils.h"
#include <gmock/gmock.h>
#include "ascc_host_stub.h"
#include "ascc_compile_base.h"
#include "ascc_types.h"
#include "ascc_compile_factory.h"
#include "ascc_link.h"
#include "ascc_argument_manager.h"
#include "ascc_global_env_manager.h"

class TEST_ASCC_GLOBAL_ENV_MANAGER : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ValueCheck_Fail)
{
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::NOT_EXIST));
    EXPECT_EQ(manager.ValueCheck(), Ascc::AsccStatus::FAILURE);
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ValueCheck_Success)
{
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(Ascc::PathCheck).stubs().will(returnValue(Ascc::PathStatus::WRITE));
    EXPECT_EQ(manager.ValueCheck(), Ascc::AsccStatus::SUCCESS);
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, InitDeviceCommonOption_Success)
{
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    EXPECT_NO_THROW(manager.InitDeviceCommonOption());
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_InitHostCommonOption)
{
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    EXPECT_NO_THROW(manager.InitHostCommonOption());
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_PrintOutInfo)
{
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    manager.PrintOutInfo();
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_cannot_find_executable)
{
    std::string path = "/usr/local/Ascend/latest";
    char* res = nullptr;
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const char*)).stubs().will(returnValue(path));
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(fgets).stubs().will(returnValue(res));
    manager.Init(path.c_str());
    EXPECT_EQ(manager.cppCompilerPath, "");
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_Init_Path_Empty)
{
    std::string path = "";
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const char*)).stubs().will(returnValue(path));
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    EXPECT_EQ(manager.Init("str"), Ascc::ASCC_FAILURE);
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_Init_CurPath_Empty)
{
    std::string path0 = "";
    std::string path = "str";
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const char*)).stubs().will(returnValue(path));
    MOCKER(Ascc::GetCurrentDirectory).stubs().will(returnValue(path0));
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    EXPECT_EQ(manager.Init("str"), Ascc::ASCC_FAILURE);
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_Init_ToolPath_Empty)
{
    std::string path = "str";
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const char*)).stubs().will(returnValue(path));
    MOCKER(Ascc::GetCurrentDirectory).stubs().will(returnValue(path));
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(&Ascc::AsccGlobalEnvManager::ValueCheck).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    EXPECT_EQ(manager.Init("str"), Ascc::ASCC_FAILURE);
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_Init_Device_Option_Fail)
{
    std::string returnPath = "str";
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const char*)).stubs().will(returnValue(returnPath));
    MOCKER(Ascc::GetCurrentDirectory).stubs().will(returnValue(returnPath));
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(&Ascc::AsccGlobalEnvManager::ValueCheck).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    EXPECT_EQ(manager.Init("str"), Ascc::ASCC_FAILURE);
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_Init)
{
    std::string returnPath = "str";
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const char*)).stubs().will(returnValue(returnPath));
    MOCKER(Ascc::GetCurrentDirectory).stubs().will(returnValue(returnPath));
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(&Ascc::AsccGlobalEnvManager::ValueCheck).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    MOCKER(&Ascc::AsccGlobalEnvManager::InitDeviceCommonOption).stubs().will(returnValue(Ascc::AsccStatus::SUCCESS));
    EXPECT_EQ(manager.Init("str"), Ascc::ASCC_SUCCESS);
}

TEST_F(TEST_ASCC_GLOBAL_ENV_MANAGER, ascc_Init_ToolPath_Empty_1)
{
    std::string path = "str";
    MOCKER(Ascc::CheckAndGetFullPath, std::string(const char*)).stubs().will(returnValue(path));
    MOCKER(Ascc::GetCurrentDirectory).stubs().will(returnValue(path));
    Ascc::AsccGlobalEnvManager &manager = Ascc::AsccGlobalEnvManager::GetInstance();
    MOCKER(&Ascc::AsccGlobalEnvManager::ValueCheck).stubs().will(returnValue(Ascc::AsccStatus::FAILURE));
    EXPECT_EQ(manager.Init("str"), Ascc::ASCC_FAILURE);
}