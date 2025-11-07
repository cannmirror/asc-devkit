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

/*!
 * \file op_build.cpp
 * \brief
 */

#include <iostream>
#include <dlfcn.h>
#include "register/op_def.h"
#include "register/op_def_factory.h"
#include "op_generator_factory.h"
#include "ascendc_tool_log.h"
#include "op_build_error_codes.h"

namespace {
constexpr uint32_t ARG_NUM_BIN = 0;
constexpr uint32_t ARG_NUM_LIB = 1;
constexpr uint32_t ARG_NUM_PATH = 2;
constexpr uint32_t ARG_NUM_VALID = 3;
}

#ifndef UT_TEST
int main(int argc, char* argv[])
{
#else
int opbuild_main(int argc, std::vector<std::string> args)
{
    char* argv[ARG_NUM_VALID];
    for (int ind = 0; ind < args.size() && ind < ARG_NUM_VALID; ind++) {
        argv[ind] = (char*)args[ind].c_str();
    }
#endif
    if (static_cast<uint32_t>(argc) < ARG_NUM_VALID) {
        ASCENDLOGE("usage: %s <op shared library> <output directory>", argv[ARG_NUM_BIN]);
        return 1;
    }
    void* handle = dlopen(argv[ARG_NUM_LIB], RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        ASCENDLOGE("dlopen error : [%s] : %s", argv[ARG_NUM_LIB], dlerror());
        return 1;
    }
    auto& allOps = ops::GeneratorFactory::FactoryGetAllOp();
    if (allOps.empty()) {
        dlclose(handle);
        ASCENDLOGE("No operator is registered!");
        return 1;
    }
    std::vector<std::string> stdOps = {};
    for (auto& op : allOps) {
        const std::string opType = op.GetString();
        if (!ops::IsVaildOpTypeName(opType)) {
            ASCENDLOGW("Optype: [%s] does not comply with the naming convention; it is recommended to \
use the PascalCase format.", opType.c_str());
        }
        stdOps.emplace_back(opType);
    }
    opbuild::Status result = ops::GeneratorFactory::SetGenPath(static_cast<const char*>(argv[ARG_NUM_PATH]));
    if (result == opbuild::OPBUILD_FAILED) {
        dlclose(handle);
        ASCENDLOGE("set generate path faield!");
        return 1;
    }
    result = ops::GeneratorFactory::Build(stdOps);
    if (result == opbuild::OPBUILD_FAILED) {
        dlclose(handle);
        ASCENDLOGE("file generate failed!");
        return 1;
    }
    std::vector<std::string> errMessage = ops::GeneratorFactory::GetErrorMessage();
    if (errMessage.size() > 0U) {
        for (std::string& str : errMessage) {
            std::cerr << str << std::endl;
        }
        return 1;
    }
    return 0;
}
