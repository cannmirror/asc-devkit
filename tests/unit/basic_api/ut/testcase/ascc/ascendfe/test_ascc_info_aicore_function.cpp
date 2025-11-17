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
#include <vector>
#define private public
#include "ascc_info_aicore_function.h"

class TEST_ASCC_INFO_AICORE_FUNCTION : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASCC_INFO_AICORE_FUNCTION, ascc_StoreKernelData)
{
    std::string filePath = "./hello_world.cpp";
    std::vector<std::pair<uint32_t, std::string>> kernelCalls = {
        {1, "void hello_world()"},
        {2, "void hello_world()"},
    };
    Ascc::AsccInfoAicoreFunc func;
    func.StoreKernelCallLineCode(filePath, kernelCalls);
}