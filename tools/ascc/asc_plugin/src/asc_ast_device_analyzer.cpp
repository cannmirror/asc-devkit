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
 * \file asc_ast_device_analyzer.cpp
 * \brief
 */

#include "asc_ast_device_analyzer.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <llvm/Support/Error.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/CommonOptionsParser.h>

#include "asc_log.h"
#include "asc_utils.h"
#include "asc_ast_device_consumer.h"
#include "asc_info_manager.h"

namespace AscPlugin {

AscAstDeviceAnalyzer::AscAstDeviceAnalyzer(const std::string &source)
{
    InitCompileDeviceArgs(source);
}

int32_t AscAstDeviceAnalyzer::Process()
{
    std::vector<std::string> argList = astDeviceArgs_.GetCmdVector("./ascplugin", /*IsClang=*/true, /*isAst=*/true);
    std::vector<const char*> argv;
    argv.reserve(argList.size());
    for (const auto& arg : argList) {
        argv.push_back(arg.c_str());
    }
    int argc = static_cast<int>(argList.size());
    llvm::cl::OptionCategory deviceToolCategory("AST Device Analyzer Options");
    auto optionsParser = clang::tooling::CommonOptionsParser::create(argc, argv.data(), deviceToolCategory);
    if (!optionsParser) {
        ASC_LOGE("AscPlugin Device analyzer: command line options parsing failed for device analysis");
        return ASC_FAILURE;
    }

    clang::tooling::ClangTool tool(optionsParser->getCompilations(), optionsParser->getSourcePathList());
    tool.appendArgumentsAdjuster([](const clang::tooling::CommandLineArguments &args, llvm::StringRef file) {
        (void)file;
        std::string commandLine = llvm::join(args, " ");
        ASC_LOGD("Clang tool: %s", commandLine.c_str());
        return args;
    });
    int result = tool.run(clang::tooling::newFrontendActionFactory<DeviceAnalyzeAction>().get());
    if (result != 0) {
        ASC_LOGE("AscPlugin Device analyzer: ClangTool runnig failed");
        return ASC_FAILURE;
    }
    return ASC_SUCCESS;
}
void AscAstDeviceAnalyzer::InitCompileDeviceArgs(const std::string &source)
{
    auto npuArch = InfoManager::GetInstance().GetShortSocVersion();
    static const std::unordered_map<ShortSocVersion, std::vector<const char*>> DAV_VERSION_MAP = {
            {ShortSocVersion::ASCEND910B, {"-D__DAV_C220_CUBE__", "-D__CCE_AICORE__=220", "-D__NPU_ARCH__=2201"}},
            {ShortSocVersion::ASCEND310P, {"-D__DAV_M200__", "-D__CCE_AICORE__=200", "-D__NPU_ARCH__=2002"}},
            {ShortSocVersion::ASCEND910, {"-D__DAV_C100__", "-D__CCE_AICORE__=100", "-D__NPU_ARCH__=1001"}},
            {ShortSocVersion::ASCEND310B, {"-D__DAV_M300__", "-D__CCE_AICORE__=300", "-D__NPU_ARCH__=3002"}},
    };

    static const std::vector<std::string> innerOpts = {
        "-x", "c++", "-std=c++17", "-Wno-c++11-narrowing"
    };
    static const std::vector<std::string> innerDefinitions = {
        "-D__global__=__attribute__((annotate(\"global\")))",
        "-D__aicore__=__attribute__((annotate(\"device\")))",
        "-D__CCE__",
        "-DGM_ADDR= __gm__ uint8_t*",
        "-D__gm__= __attribute__((annotate(\"cce_global\")))",
        "-D__host_aicore__=",
        "-DASCENDC_DUMP=1",
        "-D__CHECK_FEATURE_AT_PRECOMPILE",
        "-Dhalf=__fp16",
        "-Dbfloat16_t=__bf16",
        "-D__NPU_DEVICE__",
        // bisheng kernel type attribute
        "-D__mix__(cube, vec)=__attribute__((annotate(\"device\")))",
        "-D__cube__=__attribute__((annotate(\"device\")))",
        "-D__vector__=__attribute__((annotate(\"device\")))"};
    const CompileArgs &inputArgs = AscPlugin::InfoManager::GetInstance().GetCompileArgs();
    const std::string intputFileDir = GetFilePath(source);
    PathInfo pathInfo = InfoManager::GetInstance().GetPathInfo();
    astDeviceArgs_.includePaths = {
        "-I" + intputFileDir,
        "-I" + pathInfo.cannIncludePath,
        "-I" + pathInfo.hostApiPath,
        "-I" + pathInfo.highLevelApiPath,
        "-I" + pathInfo.tikcfwPath,
        "-I" + pathInfo.tikcfwLibPath,
        "-I" + pathInfo.tikcfwLibMatmulPath,
        "-I" + pathInfo.tikcfwImplPath,
        "-I" + pathInfo.tikcfwInterfacePath,
        "-I" + pathInfo.ascendClangIncludePath
    };
    astDeviceArgs_.file = source;
    astDeviceArgs_.options = innerOpts;
    astDeviceArgs_.includePaths.insert(astDeviceArgs_.includePaths.end(), inputArgs.includePaths.begin(),
        inputArgs.includePaths.end());
    astDeviceArgs_.definitions = inputArgs.definitions;
    std::vector<std::string> removeOpts = {"-DL2_CACHE_HINT"};
    astDeviceArgs_.RemoveOptions(removeOpts);
    astDeviceArgs_.definitions.insert(astDeviceArgs_.definitions.end(), innerDefinitions.begin(), innerDefinitions.end());
    const auto& archOptionList = DAV_VERSION_MAP.at(npuArch);
    for (const auto& option : archOptionList) {
        astDeviceArgs_.definitions.emplace_back(option);
    }
}
}  // namespace Asc