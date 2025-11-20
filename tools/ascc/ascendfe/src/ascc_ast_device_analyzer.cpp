/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ascc_ast_device_analyzer.cpp
 * \brief
 */

#include "ascc_ast_device_analyzer.h"

#include <string>
#include <vector>
#include <llvm/Support/Error.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/CommonOptionsParser.h>

#include "ascc_log.h"
#include "ascc_utils.h"
#include "ascc_dump_flags.h"
#include "ascc_ast_device_consumer.h"
#include "ascc_compile_base.h"
#include "ascc_argument_manager.h"
#include "ascc_global_env_manager.h"

namespace Ascc {
static const std::unordered_map<Ascc::ShortSoCVersion, const char*> FIND_KFC_DAV_MACRO_MAP = {
    {Ascc::ShortSoCVersion::ASCEND910B, "__DAV_C220_CUBE__"},
    {Ascc::ShortSoCVersion::ASCEND310P, "__DAV_M200__"},
    {Ascc::ShortSoCVersion::ASCEND910, "__DAV_C100__"},
    {Ascc::ShortSoCVersion::ASCEND310B, "__DAV_M300__"},
};

AsccAstDeviceAnalyzer::AsccAstDeviceAnalyzer(const std::string &source)
{
    InitCompileDeviceArgs(source);
}

AsccStatus AsccAstDeviceAnalyzer::Process()
{
    const std::vector<std::string> &argList =
        astDeviceArgs_.GetCmdVector("./ascc", /*IsClang=*/true, CompileType::NONE, /*isAst=*/true);
    std::vector<const char*> argv;
    const char **argvData = ConvertStringVecToCStringVec(argv, argList);
    int argc = static_cast<int>(argList.size());
    llvm::cl::OptionCategory deviceToolCategory("AST Device Analyzer Options");
    auto optionsParser = clang::tooling::CommonOptionsParser::create(argc, argvData, deviceToolCategory);
    if (!optionsParser) {
        ASC_LOG_ASC_ERROR(PREPROCESS,
            "ASCC Device analyzer: command line options parsing failed for device analysis");
        return AsccStatus::FAILURE;
    }

    clang::tooling::ClangTool tool(optionsParser->getCompilations(), optionsParser->getSourcePathList());
    tool.appendArgumentsAdjuster([](const clang::tooling::CommandLineArguments &args, llvm::StringRef file) {
        (void)file;
        return args;
    });
    int result = tool.run(clang::tooling::newFrontendActionFactory<DeviceAnalyzeAction>().get());
    if (result != 0) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "ASCC Device analyzer: ClangTool runnig failed");
        return AsccStatus::FAILURE;
    }
    return AsccStatus::SUCCESS;
}
void AsccAstDeviceAnalyzer::InitCompileDeviceArgs(const std::string &source)
{
    static const std::vector<std::string> innerOpts = {
        "-x", "c++", "-std=c++17", "-Wno-c++11-narrowing"
    };
    static const std::vector<std::string> innerDefinitions = {"__global__=__attribute__((annotate(\"global\")))",
        "__aicore__=__attribute__((annotate(\"device\")))",
        "__gm__= __attribute__((annotate(\"cce_global\")))",
        "__host_aicore__=",
        "ASCENDC_DUMP=1",
        "__CHECK_FEATURE_AT_PRECOMPILE",
        "__NPU_DEVICE__"};
    const auto& npuArch = Ascc::AsccArgumentManager::GetInstance().GetNpuArch();
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    const auto& inputArgs = Ascc::AsccArgumentManager::GetInstance().GetCompileOptions();
    const std::string intputFileDir = GetFilePath(Ascc::AsccArgumentManager::GetInstance().GetInputFile());
    astDeviceArgs_.file = source;
    astDeviceArgs_.options = innerOpts;
    astDeviceArgs_.incPaths = {
        intputFileDir,
        envVar.ascendCannIncludePath,
        envVar.ascendHighlevelApiPath,
        envVar.ascendHighlevelApiImplPath,
        envVar.ascendHighlevelApiLibPath,
        envVar.ascendHighlevelApiLibMatmulPath,
        envVar.ascendTikcfwPath,
        envVar.ascendTikcfwImplPath,
        envVar.ascendTikcfwInterfacePath,
        envVar.ascendClangIncPath
    };
    astDeviceArgs_.incPaths.insert(astDeviceArgs_.incPaths.end(), inputArgs.incPaths.begin(), inputArgs.incPaths.end());
    astDeviceArgs_.definitions = inputArgs.definitions;
    astDeviceArgs_.definitions.insert(
        astDeviceArgs_.definitions.end(), innerDefinitions.begin(), innerDefinitions.end());
    astDeviceArgs_.definitions.emplace_back(CCE_AICORE_MAP.at(npuArch));
    astDeviceArgs_.definitions.emplace_back(NPU_ARCH_MAP.at(npuArch));
    astDeviceArgs_.definitions.emplace_back(FIND_KFC_DAV_MACRO_MAP.at(npuArch));
}
}  // namespace Ascc