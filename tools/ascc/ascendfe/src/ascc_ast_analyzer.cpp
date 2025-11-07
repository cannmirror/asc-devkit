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
 * \file ascc_ast_analyzer.cpp
 * \brief
 */

#include "ascc_ast_analyzer.h"

#include <clang/Tooling/CommonOptionsParser.h>

#include "ascc_ast_info_collector.h"
#include "ascc_compile_base.h"
#include "ascc_argument_manager.h"
#include "ascc_global_env_manager.h"
#include "ascc_log.h"
#include "ascc_utils.h"

namespace Ascc {
AsccAstAnalyzer::AsccAstAnalyzer(const std::string &source)
{
    InitCompileArgs(source);
}

AsccStatus AsccAstAnalyzer::Process()
{
    const std::vector<std::string> &argList =
        astArgs_.GetCmdVector("./ascc", /*IsClang=*/true, CompileType::NONE, /*isAst=*/true);
    std::vector<const char*> argv;
    const char **argvData = ConvertStringVecToCStringVec(argv, argList);
    int argc = static_cast<int>(argList.size());

    llvm::cl::OptionCategory asccToolCategory("ASCC AST Analyzer Options");
    // init llvm argument parser
    auto optionsParser = clang::tooling::CommonOptionsParser::create(argc, argvData, asccToolCategory);
    if (!optionsParser) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "Failed to parse command in AsccAstAnalyzer!");
        return AsccStatus::FAILURE;
    }

    // init clang tool
    clang::tooling::ClangTool tool(optionsParser->getCompilations(), optionsParser->getSourcePathList());
    tool.appendArgumentsAdjuster([](const clang::tooling::CommandLineArguments &args, llvm::StringRef file) {
        (void)file;
        return args;
    });
    // clang tool run
    int result = tool.run(clang::tooling::newFrontendActionFactory<AsccFrontendAction>().get());
    if (result != 0) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "ClangTool run failed in AsccAstAnalyzer!");
        return AsccStatus::FAILURE;
    }
    return AsccStatus::SUCCESS;
}

void AsccAstAnalyzer::InitCompileArgs(const std::string &source)
{
    static const std::vector<std::string> innerOpts = {
        "-x", "cu", "-nocudainc", "--cuda-host-only", "-std=c++17", "-Wno-c++11-narrowing", "-fsyntax-only",
        "-Wno-unknown-cuda-version", "-Wno-cuda-compat"
    };
    static const std::vector<std::string> innerDefinitions = {
        "__host_aicore__=", "__forceinline__=", "__NPU_HOST__",
        "__asccPCC__=__cudaPushCallConfiguration", "__asccCC__=cudaConfigureCall"
    };
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    const auto& inputArgs = Ascc::AsccArgumentManager::GetInstance().GetCompileOptions();
    const auto& npuArch = Ascc::AsccArgumentManager::GetInstance().GetNpuArch();
    const std::string intputFileDir = GetFilePath(Ascc::AsccArgumentManager::GetInstance().GetInputFile());
    const std::string kernelStubHeader = envVar.asccTmpPath + "/include/kernel_operator_stub.h";
    astArgs_.file = source;
    astArgs_.options = innerOpts;
    astArgs_.incPaths = {
        intputFileDir,
        envVar.ascendClangIncPath,
        envVar.ascendCannIncludePath,
        envVar.ascendHighlevelApiPath,
        envVar.ascendTikcfwPath,
        envVar.ascendHighlevelApiImplPath,
        envVar.ascendTikcfwImplPath,
        envVar.ascendTikcfwInterfacePath
    };
    astArgs_.incPaths.insert(astArgs_.incPaths.end(), inputArgs.incPaths.begin(), inputArgs.incPaths.end());
    astArgs_.incFiles = {kernelStubHeader};
    astArgs_.definitions = inputArgs.definitions;
    astArgs_.definitions.insert(astArgs_.definitions.end(), innerDefinitions.begin(), innerDefinitions.end());
    if (AsccArgumentManager::GetInstance().GetPreTaskType() != PreTaskType::NONE) {
        astArgs_.definitions.emplace_back("_GLIBCXX_TYPE_TRAITS");
    }
    astArgs_.definitions.emplace_back(CCE_AICORE_MAP.at(npuArch));
    astArgs_.definitions.emplace_back(NPU_ARCH_MAP.at(npuArch));
}

}  // namespace Ascc
