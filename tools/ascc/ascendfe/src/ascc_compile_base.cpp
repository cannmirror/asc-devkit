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
 * \file ascc_compile_base.cpp
 * \brief
 */

#include "ascc_compile_base.h"

#include <iostream>
#include <string>
#include <cstdio>
#include <sstream>
#include <memory>
#include <vector>

#include "ascc_dump_flags.h"
#include "ascc_utils.h"

namespace Ascc {
void AsccCompileBase::AddOption(const std::string &option)
{
    this->args_.options.emplace_back(option);
}
void AsccCompileBase::AddDefinition(const std::string &definition)
{
    this->args_.definitions.emplace_back(definition);
}
void AsccCompileBase::AddIncPath(const std::string &path)
{
    this->args_.incPaths.emplace_back(path);
}
void AsccCompileBase::AddIncFile(const std::string &file)
{
    this->args_.incFiles.insert(this->args_.incFiles.begin(), file);
}
void AsccCompileBase::SetCustomOption(const std::string &option)
{
    this->args_.customOption = option;
}

AsccStatus AsccCompileBase::ExecuteCompile(const std::string &cmd) const
{
    Ascc::SaveCompileLogFile("Compile command:", cmd);
    ASC_LOG_ASC_INFO(COMPILE, "Compile cmd : [%s].", cmd.c_str());
    try {
        // Redirect standard error output to standard output
        std::string output = "DEFAULT";
        int returnCode = 0;
        std::tie(output, returnCode) = Ascc::ExecuteCommand((cmd + " 2>&1").c_str());

        if (returnCode != 0) {
            std::cerr << "Function " << __FUNCTION__ << " at line " << __LINE__
                      << " Command execution failed, the returnCode is non-zero!\n";
            std::cout << "Output of ascendc_compiler:\n" << output;
            return AsccStatus::FAILURE;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error in function " << __FUNCTION__ << " at line " << __LINE__ << ": " << e.what() << '\n';
        return AsccStatus::FAILURE;
    }
    return AsccStatus::SUCCESS;
}

std::string AsccCompileBase::GetDependencyCmd() const
{
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    std::string dependencyCmd = argManager.GetDependencyOption();
    // if -MF, -MT is not parsed, then need to pass -MF, -MT to make .d file not in tmp directory
    if (argManager.MFMTRequested()) {
        if (argManager.GetMFFileName() == "") {
            std::string srcFileName;
            if (argManager.GetOutputFile().empty()) {     // when without -o, filename=replace input file suffix with .d
                srcFileName = argManager.GetInputFile();
            } else {                                      // when with -o, filename=replace output file suffix with .d
                srcFileName = argManager.GetOutputFile();
            }
            std::string expectedDFileName = Ascc::RemoveSuffix(srcFileName) + ".d";
            dependencyCmd += " -MF " + expectedDFileName;
        }
        if (argManager.GetMTFileName() == "") {
            std::string expectedFileName;
            if (argManager.GetOutputFile().empty()) {     // when without -o, filename=replace input file suffix with .o
                expectedFileName = Ascc::RemoveSuffix(argManager.GetInputFile()) + ".o";
            } else {                                      // when with -o, filename = output filename
                expectedFileName = argManager.GetOutputFile();
            }
            dependencyCmd += " -MT " + expectedFileName;
        }
    }
    return dependencyCmd;
}

void AsccCompileBase::MergeCommonOption(const Ascc::CompileArgs& commonArgs)
{
    args_.definitions.insert(args_.definitions.end(), commonArgs.definitions.begin(), commonArgs.definitions.end());
    args_.incPaths.insert(args_.incPaths.end(), commonArgs.incPaths.begin(), commonArgs.incPaths.end());
    args_.options.insert(args_.options.end(), commonArgs.options.begin(), commonArgs.options.end());
    args_.incFiles.insert(args_.incFiles.end(), commonArgs.incFiles.begin(), commonArgs.incFiles.end());
    args_.linkFiles.insert(args_.linkFiles.end(), commonArgs.linkFiles.begin(), commonArgs.linkFiles.end());
    args_.linkPath.insert(args_.linkPath.end(), commonArgs.linkPath.begin(), commonArgs.linkPath.end());
    if (args_.outputPath.empty()) {
        args_.outputPath = commonArgs.outputPath;
    }
    if (args_.file.empty()) {
        args_.file = commonArgs.file;
    }
    if (args_.customOption.empty()) {
        args_.customOption = commonArgs.customOption;
    }
}
}