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

class TEST_ASC_AST_UTILS : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_AST_UTILS, asc_CompileArgs_GetCmd)
{
    AscPlugin::CompileArgs a;
    a.definitions = {"-DASCENDC_DUMP=1"};
    a.includePaths = {"-IBdir"};
    a.linkFiles = {"-llinklib"};
    a.linkPaths = {"-Llinkpath"};
    a.file = "a.cpp";
    a.outputPath = "a.out";

    EXPECT_EQ(a.GetCmd("ascc"), "ascc -c -DASCENDC_DUMP=1 -IBdir -llinklib -Llinkpath -o a.out -- a.cpp ");
}

TEST_F(TEST_ASC_AST_UTILS, asc_CompileArgs_RemoveOptions)
{
    AscPlugin::CompileArgs a;
    a.definitions = {"-DASCENDC_DUMP=1", "-DTEST"};
    a.includePaths = {"-IBdir"};
    a.linkFiles = {"-llinklib"};
    a.linkPaths = {"-Llinkpath"};
    a.file = "a.cpp";
    a.outputPath = "a.out";
    std::vector<std::string> removeOpts = {"-DASCENDC_DUMP", "-DASCENDC_DUMP=1"};
    EXPECT_NO_THROW(a.RemoveOptions(removeOpts));
}