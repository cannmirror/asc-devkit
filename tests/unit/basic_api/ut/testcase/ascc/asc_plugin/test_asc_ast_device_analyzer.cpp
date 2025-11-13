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

#include "asc_ast_device_analyzer.h"

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <llvm/Support/Error.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/CommonOptionsParser.h>
#define private public
#include "asc_log.h"
#include "asc_utils.h"
#include "asc_ast_device_consumer.h"
#include "asc_info_manager.h"

class TEST_ASC_AST_DEVICE_ANALYZER : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_AST_DEVICE_ANALYZER, asc_plugin_CommonOptionsParser_create_failed)
{
    std::string file = "test.cpp";
    AscPlugin::AscAstDeviceAnalyzer deviceAnalyzer(file);
    MOCKER(&clang::tooling::ClangTool::run).stubs().will(returnValue(1));
    EXPECT_EQ(deviceAnalyzer.Process(), AscPlugin::ASC_FAILURE);
}