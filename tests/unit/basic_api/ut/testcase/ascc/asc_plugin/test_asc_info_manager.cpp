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
#include <iostream>
#include <gtest/gtest.h>
#include <unistd.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "asc_ast_utils.h"
#include "asc_utils.h"
#include "asc_info_manager.h"

class TEST_ASC_INFO_MANAGER : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_INFO_MANAGER, asc_get_method)
{
    AscPlugin::CompileArgs a;
    a.definitions = {"-DASCENDC_DUMP=1"};
    a.includePaths = {"-IBdir"};
    a.linkFiles = {"-llinklib"};
    a.linkPaths = {"-Llinkpath"};

    a.definitions = {"-DASCENDC_DUMP=1"};
    a.includePaths = {"-IBdir"};
    a.linkFiles = {"-llinklib"};
    a.linkPaths = {"-Llinkpath"};

    std::vector<std::string> b = {"-DASCENDC_DUMP=1", "-IBdir", "-Llinkpath", "-llinklib", "--cce-auto-sync=off"};
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCompileArgs(b);
    manager.isAutoSyncOn_ = true;

    EXPECT_EQ(manager.GetCompileArgs().definitions, a.definitions);
    EXPECT_EQ(manager.GetCompileArgs().includePaths, a.includePaths);
    EXPECT_EQ(manager.GetCompileArgs().linkPaths, a.linkPaths);
    EXPECT_EQ(manager.GetCompileArgs().linkFiles, a.linkFiles);
}

TEST_F(TEST_ASC_INFO_MANAGER, asc_set_method)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetOptimizeLevel("-O2");
    manager.SetUserDumpStatus(true);
    manager.SetHasPrintf(false);
    manager.SetHasAssert(true);
    manager.SetSourceFile("test.cpp");

    EXPECT_EQ(manager.GetOptimizeLevel(), "-O2");
    EXPECT_EQ(manager.UserDumpRequested(), true);
    EXPECT_EQ(manager.HasPrintf(), false);
    EXPECT_EQ(manager.HasAssert(), true);
    EXPECT_EQ(manager.GetSourceFile(), "test.cpp");

    manager.SetHasAssert(false);
}

TEST_F(TEST_ASC_INFO_MANAGER, asc_set_global_symbol_info)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.AddGlobalSymbolInfo(
        "__device_stub__mangling_global_add", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "test.cpp", 1, 0, AscPlugin::KfcScene::Close);
    const auto& info = manager.GetGlobalSymbolInfo();
    const auto& [ktype, fileName, lineNo, colNo, kfcScene] = info.at("__device_stub__mangling_global_add");
    EXPECT_EQ(ktype, AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2);
    EXPECT_EQ(fileName, std::string("test.cpp"));
    EXPECT_EQ(lineNo, 1);
    EXPECT_EQ(colNo, 0);
}

TEST_F(TEST_ASC_INFO_MANAGER, asc_SetCompileArgs)
{
    AscPlugin::CompileArgs a;

    auto& manager = AscPlugin::InfoManager::GetInstance();

    std::vector<std::string> compileArgs = {"-DFFF", "-UFFF", "-UAAAA", "-DAAA","-include", "F.h", "-DSSS", "-l", "AAA", "-lBBB"};
    std::vector<std::string> expectDef = {"-DFFF", "-UFFF", "-UAAAA", "-DAAA", "-DSSS"};
    std::vector<std::string> expectIncFiles = {"-include", "", "-include", "F.h"};
    std::vector<std::string> expectLinkFiles = {"-l", "AAA", "-lBBB"};
    manager.compileArgs_ = a;   // clean up

    manager.SetCompileArgs(compileArgs);
    AscPlugin::CompileArgs info = manager.GetCompileArgs();
    EXPECT_EQ(info.definitions, expectDef);
    EXPECT_EQ(info.includeFiles, expectIncFiles);
    EXPECT_EQ(info.linkFiles, expectLinkFiles);
}

TEST_F(TEST_ASC_INFO_MANAGER, asc_SetCompileArgs_with_host_options)
{
    AscPlugin::CompileArgs a;

    auto& manager = AscPlugin::InfoManager::GetInstance();

    std::vector<std::string> compileArgs = {"-DFFF", "-Xaicore-start", "-DAAA", "-Xaicore-end",
        "-Xhost-start", "-DBBBB", "-Xhost-end",};
    std::vector<std::string> expectDef = {"-DFFF", "-DAAA"};
    std::vector<std::string> expectHostDef = {"-DBBBB"};
    std::vector<std::string> expectIncFiles = {"-include", ""};
    manager.compileArgs_ = a;   // clean up

    manager.SetCompileArgs(compileArgs);
    AscPlugin::CompileArgs info = manager.GetCompileArgs();
    EXPECT_EQ(info.definitions, expectDef);
    EXPECT_EQ(info.includeFiles, expectIncFiles);
    EXPECT_EQ(info.hostDefinitions, expectHostDef);
}

// // For -UXXX,A => -U XXX
// TEST_F(TEST_ASC_INFO_MANAGER, asc_SetCompileArgs_Undef_Def)
// {
//     AscPlugin::CompileArgs a;

//     auto& manager = AscPlugin::InfoManager::GetInstance();

//     std::vector<std::string> compileArgs = {"-DAAA=3", "-D", "FFF=5", "-UFFF,ababs",};
//     std::vector<std::string> expectDef = {"-DAAA=3", "-DFFF=5", "-UFFF"};
//     manager.compileArgs_ = a;   // clean up

//     manager.SetCompileArgs(compileArgs);
//     AscPlugin::CompileArgs info = manager.GetCompileArgs();
//     EXPECT_EQ(info.definitions, expectDef);
// }

// -D dump = false => -U DUMP = true
TEST_F(TEST_ASC_INFO_MANAGER, asc_SetCompileArgs_Undef_Def_dump)
{
    AscPlugin::CompileArgs a;

    auto& manager = AscPlugin::InfoManager::GetInstance();

    std::vector<std::string> compileArgs = {"-DASCENDC_DUMP=0"};
    manager.compileArgs_ = a;   // clean up
    manager.SetCompileArgs(compileArgs);
    EXPECT_EQ(manager.UserDumpRequested(), false);

    compileArgs = {"-DASCENDC_DUMP=0", "-U", "ASCENDC_DUMP"};
    manager.compileArgs_ = a;   // clean up
    manager.SetCompileArgs(compileArgs);
    EXPECT_EQ(manager.UserDumpRequested(), true);
}

TEST_F(TEST_ASC_INFO_MANAGER, asc_SetCompileArgs_Undef)
{
    AscPlugin::CompileArgs a;

    auto& manager = AscPlugin::InfoManager::GetInstance();
    std::vector<std::string> compileArgs = { "-DASCENDC_DUMP=0", "-DHAVE_WORKSPACE", "-DHAVE_TILING",
        "-DASCENDC_TIME_STAMP_ON", "-DASCENDC_DEBUG"};
    manager.compileArgs_ = a;   // clean up
    manager.SetCompileArgs(compileArgs);
    EXPECT_EQ(manager.UserDumpRequested(), false);
    EXPECT_EQ(manager.HasTimeStamp(), true);
    EXPECT_EQ(manager.HasWorkspace(), true);
    EXPECT_EQ(manager.HasTiling(), true);
    EXPECT_EQ(manager.IsL2CacheEnabled(), false);

    compileArgs = { "-UASCENDC_DUMP", "-UHAVE_WORKSPACE", "-UHAVE_TILING",
        "-UASCENDC_TIME_STAMP_ON", "-UASCENDC_DEBUG"};
    manager.SetCompileArgs(compileArgs);
    EXPECT_EQ(manager.UserDumpRequested(), true);
    EXPECT_EQ(manager.HasTimeStamp(), false);
    EXPECT_EQ(manager.HasWorkspace(), false);
    EXPECT_EQ(manager.HasTiling(), false);
    EXPECT_EQ(manager.IsL2CacheEnabled(), true);
}

// for -DHAVE_WORKSPACE and -DHAVE_TILING
TEST_F(TEST_ASC_INFO_MANAGER, asc_workspace_tiling)
{
    AscPlugin::CompileArgs a;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.compileArgs_ = a;   // clean up
    EXPECT_EQ(manager.HasWorkspace(), false);
    EXPECT_EQ(manager.HasTiling(), false);

    std::vector<std::string> compileArgs = {"-DHAVE_WORKSPACE","-DHAVE_TILING"};

    manager.SetCompileArgs(compileArgs);
    EXPECT_EQ(manager.HasWorkspace(), true);
    EXPECT_EQ(manager.HasTiling(), true);
    manager.hasWorkspace_ = false;
    manager.hasTiling_ = false;
}

// for -DASCENDC_TIME_STAMP_ON
TEST_F(TEST_ASC_INFO_MANAGER, asc_timestamp_on)
{
    AscPlugin::CompileArgs a;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.compileArgs_ = a;   // clean up
    EXPECT_EQ(manager.HasTimeStamp(), false);

    std::vector<std::string> compileArgs = {"-DASCENDC_TIME_STAMP_ON"};
    manager.SetCompileArgs(compileArgs);
    EXPECT_EQ(manager.HasTimeStamp(), true);
}