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
#include "ascc_option.h"


using namespace testing;
using namespace Ascc;

class TEST_ASCC_OPTION : public testing::Test {
protected:
    void SetUp()
    {
    }

    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

namespace Ascc {
    extern void PrintRequireOptInfo(Option* optPtr);
}

// ==============================================================================
TEST_F(TEST_ASCC_OPTION, ascc_PrintRequireOptInfo)
{
    // have longArgStr + shortStr
    static Opt<std::string> lltOpt1("llt_test1", ShortDesc("llt"), ValueDesc("<test>"),
        HelpDesc("llt test1 only.\n"));
    PrintRequireOptInfo(&lltOpt1);
    // only have longArgStr
    static Opt<std::string> lltOpt2("llt_test2", ValueDesc("<test>"),
        HelpDesc("llt test2 only.\n"));
    lltOpt2.shortArgStr.clear();
    PrintRequireOptInfo(&lltOpt2);
    // only have shortArgStr
    lltOpt2.longArgStr.clear();
    lltOpt2.shortArgStr = "shortArg";
    PrintRequireOptInfo(&lltOpt2);
}