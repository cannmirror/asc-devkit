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
#define private public
#include "asc_log.h"

class TEST_ASC_LOG : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_LOG, LogToFile) {
    AscPlugin::InfoManager::GetInstance().SetSaveTempRequested(true);
    AscPlugin::InfoManager::GetInstance().logPath_ = "/tmp/test";
    system("mkdir -p /tmp/test");
    const char* outEnvValue = "1";
    const char* levelEnvValue = "0";
    MOCKER(&AscPlugin::LogManager::GetOutEnv).stubs().will(returnValue(outEnvValue));
    MOCKER(&AscPlugin::LogManager::GetLevelEnv).stubs().will(returnValue(levelEnvValue));
    const char* c = "Test";
    EXPECT_NO_THROW(ASC_LOGD("AscPlugin Debug level %s", c));
    EXPECT_NO_THROW(ASC_LOGI("AscPlugin Info level Test"));
    EXPECT_NO_THROW(ASC_LOGW("AscPlugin Warn level Test"));
    EXPECT_NO_THROW(ASC_LOGE("AscPlugin Error level Test"));
    system("rm -rf /tmp/test");
}

TEST_F(TEST_ASC_LOG, WarnAndErrorPrint) {
    AscPlugin::InfoManager::GetInstance().logPath_ = "/tmp/test";
    system("mkdir -p /tmp/test");
    const char* outEnvValue = "0";
    const char* levelEnvValue = "0";
    MOCKER(&AscPlugin::LogManager::GetOutEnv).stubs().will(returnValue(outEnvValue));
    MOCKER(&AscPlugin::LogManager::GetLevelEnv).stubs().will(returnValue(levelEnvValue));
    EXPECT_NO_THROW(ASC_LOGD("AscPlugin Debug level Test"));
    EXPECT_NO_THROW(ASC_LOGI("AscPlugin Info level Test"));
    EXPECT_NO_THROW(ASC_LOGW("AscPlugin Warn level Test"));
    EXPECT_NO_THROW(ASC_LOGE("AscPlugin Error level Test"));
    system("rm -rf /tmp/test");
}

TEST_F(TEST_ASC_LOG, LogToFileNoPrintf) {
    AscPlugin::InfoManager::GetInstance().logPath_ = "/tmp/test";
    system("mkdir -p /tmp/test");
    EXPECT_EQ(AscPlugin::LogManager::GetOutEnv(), nullptr);
    EXPECT_EQ(AscPlugin::LogManager::GetLevelEnv(), nullptr);
    system("rm -rf /tmp/test");
}

TEST_F(TEST_ASC_LOG, LogToFileFailed) {
    std::string fullPath = "./no_file/AscPlugin.log";
    std::unique_ptr<AscPlugin::LogManager> log_ptr;
    log_ptr = std::make_unique<AscPlugin::LogManager>(fullPath);
    EXPECT_EQ(log_ptr->logFile_, nullptr);
}
