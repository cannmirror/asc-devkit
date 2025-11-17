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
 * \file ascc_global_env_manager.cpp
 * \brief
 */

#include <stdexcept>
#include "ascc_log.h"
#include "ascc_utils.h"
#include "ascc_global_env_manager.h"

namespace Ascc {

namespace {
// use which to find path of executable
const std::string FindExecPath(std::string execName)
{
    std::string pathCmd = ("which " + execName + " 2>/dev/null");
    FILE* pipe = popen(pathCmd.c_str(), "r");
    if (pipe) {
        char buffer[128];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            std::string path(buffer);
            // remove \n symbol
            size_t pos = path.find_last_not_of("\n");
            if (pos != std::string::npos) {
                path.erase(pos + 1);
            }
            pclose(pipe);
            if (access(path.c_str(), X_OK) == 0) {
                ASC_LOG_ASC_DEBUG(INIT, "Path of executable %s is: %s.", execName.c_str(), path.c_str());
                return path;
            }
        }
        pclose(pipe);
    }
    ASC_LOG_ASC_ERROR(INIT, "Cannot find path of executable: [%s].", execName.c_str());
    return "";   // return empty if not found g++
}
} // namespace

void AsccGlobalEnvManager::InitDeviceCommonOption()
{
    commonDeviceArgs.options.emplace_back("-DTILING_KEY_VAR=0 -std=c++17 --cce-aicore-block-local-init");
    commonDeviceArgs.incPaths.emplace_back(ascendCannIncludePath);
    commonDeviceArgs.incPaths.emplace_back(ascendHighlevelApiPath);
    commonDeviceArgs.incPaths.emplace_back(ascendTikcfwLibPath);
    commonDeviceArgs.incPaths.emplace_back(ascendTikcfwPath);
    commonDeviceArgs.incPaths.emplace_back(ascendTikcfwInterfacePath);
    commonDeviceArgs.incPaths.emplace_back(ascendTikcfwImplPath);
    commonDeviceArgs.incFiles.emplace_back(ascendVersionHeader);
}

void AsccGlobalEnvManager::InitHostCommonOption()
{
    commonHostArgs.incPaths.emplace_back(ascendCannIncludePath);
    commonHostArgs.incPaths.emplace_back(ascendHostApiPath);
    commonHostArgs.incPaths.emplace_back(ascendHighlevelApiPath);
    commonHostArgs.incPaths.emplace_back(ascendTikcfwPath);
    commonHostArgs.incPaths.emplace_back(ascendTikcfwLibPath);
    commonHostArgs.incPaths.emplace_back(ascendTikcfwLibMatmulPath);
    commonHostArgs.incPaths.emplace_back(ascendTikcfwImplPath);
    commonHostArgs.incPaths.emplace_back(ascendTikcfwInterfacePath);
    commonHostArgs.incPaths.emplace_back(ascendTikcpulibPath);
}

void AsccGlobalEnvManager::PrintOutInfo() const
{
    ASC_LOG_ASC_DEBUG(INIT, "ASCEND_GLOBAL_LOG_LEVEL: [%u].", AsccGlobalEnvManager::ascendGlobalLogLevel);
    ASC_LOG_ASC_DEBUG(INIT, "ASCEND_SLOG_PRINT_TO_STDOUT: [%u].", AsccGlobalEnvManager::ascendSlogPrintToStdout);
    ASC_LOG_ASC_DEBUG(INIT, "ASCEND_HOME_PATH : [%s].", this->ascendCannPackagePath.c_str());
    ASC_LOG_ASC_DEBUG(INIT, "CCEC_PATH : [%s].", this->ccecPath.c_str());
    ASC_LOG_ASC_DEBUG(INIT, "ASCEND_LINKER : [%s].", this->ascendLinker.c_str());
    ASC_LOG_ASC_DEBUG(INIT, "ASCEND_AR : [%s].", this->ascendAr.c_str());
}

AsccStatus AsccGlobalEnvManager::ValueCheck() const
{
    const std::vector<std::string> checkList = {
        ascendCannIncludePath,
        ascendHighlevelApiPath,
        ascendHostApiPath,
        ascendTikcfwPath,
        ascendTikcfwLibPath,
        ascendTikcfwLibMatmulPath,
        ascendTikcfwImplPath,
        ascendTikcfwInterfacePath,
        ascendTikcpulibPath,
        ascendVersionHeader,
        ccecPath,
        ascendLinker,
        ascendAr,
        ascendHighlevelApiImplPath,
        ascendClangIncPath,
        ascendCompiler
    };
    for (const auto& path : checkList) {
        if (PathCheck(path.c_str(), true) == PathStatus::NOT_EXIST) {
            Ascc::HandleError(std::string("CANN PathCheck Fail : [") + path + "]!");
            return AsccStatus::FAILURE;
        }
    }
    return AsccStatus::SUCCESS;
}

uint32_t AsccGlobalEnvManager::Init(const char* cannPath)
{
    std::string curCannPath = Ascc::CheckAndGetFullPath(cannPath);
    if (curCannPath.empty()) {
        Ascc::HandleError(std::string("CANN PathCheck Fail : [") + cannPath + "]!");
        return ASCC_FAILURE;
    }
    this->currentPath = Ascc::GetCurrentDirectory();
    if (this->currentPath.empty()) {
        Ascc::HandleError("Get current path failed!");
        return ASCC_FAILURE;
    }
    this->ascendCannPackagePath = curCannPath;
    this->ascendCannIncludePath = curCannPath + "/include";
    this->ascendHighlevelApiPath = curCannPath + "/compiler/ascendc/include/highlevel_api";
    this->ascendHighlevelApiLibPath = curCannPath + "/compiler/ascendc/include/highlevel_api/lib";
    this->ascendHighlevelApiLibMatmulPath = curCannPath + "/compiler/ascendc/include/highlevel_api/lib/matmul";
    this->ascendHighlevelApiImplPath = curCannPath + "/compiler/ascendc/include/highlevel_api/impl";
    this->ascendHostApiPath = curCannPath + "/include/ascendc/host_api";
    this->ascendTikcfwPath = curCannPath + "/compiler/tikcpp/tikcfw";
    this->ascendTikcfwLibPath = curCannPath + "/compiler/tikcpp/tikcfw/lib";
    this->ascendTikcfwLibMatmulPath = curCannPath + "/compiler/tikcpp/tikcfw/lib/matmul";
    this->ascendTikcfwImplPath = curCannPath + "/compiler/tikcpp/tikcfw/impl";
    this->ascendTikcfwInterfacePath = curCannPath + "/compiler/tikcpp/tikcfw/interface";
    this->ascendTikcpulibPath = curCannPath + "/tools/tikicpulib/lib/include";
    this->ascendClangIncPath = curCannPath + "/compiler/ccec_compiler/lib/clang/15.0.5/include";
    this->ascendVersionHeader = curCannPath + "/include/version/cann_version.h";
    this->ccecPath = curCannPath + "/tools/ccec_compiler/bin";
    this->ascendCompiler = curCannPath + "/tools/ccec_compiler/bin/bisheng";
    this->ascendLinker = curCannPath + "/tools/ccec_compiler/bin/ld.lld";
    this->ascendAr = curCannPath + "/tools/ccec_compiler/bin/llvm-ar";
    this->asccTmpPath = "/tmp/ascc";
    this->asccTmpIncludePath = this->asccTmpPath + "/include";
    this->asccTmpAutoGenPath = this->asccTmpPath + "/auto_gen";
    this->asccTmpHostGenPath = this->asccTmpPath + "/auto_gen/host_files";
    this->asccTmpDependPath = this->asccTmpPath + "/dependence";
    this->asccMergeObjPath = this->asccTmpPath + "/link_files/merge_obj";
    this->asccMergeObjFinalPath = this->asccTmpPath + "/link_files/merge_obj_final";
    this->asccCompileLogPath = this->asccTmpPath + "/compile_log";
    this->cppCompilerPath = FindExecPath("c++");
    this->ldPath = FindExecPath("ld");
    if (ValueCheck() == AsccStatus::FAILURE) {
        ASC_LOG_ASC_ERROR(INIT, "Init host options failed!");
        return ASCC_FAILURE;
    }
    InitDeviceCommonOption();
    InitHostCommonOption();
    this->initSuccess = true;
    PrintOutInfo();
    return ASCC_SUCCESS;
}

static uint32_t g_envVarInit = []() -> uint32_t {
    AsccGlobalEnvManager& envVar = AsccGlobalEnvManager::GetInstance();
    const char* logLevel = getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (logLevel != nullptr) {
        try {
            AsccGlobalEnvManager::ascendGlobalLogLevel = std::stoi(logLevel);
        } catch (...) {
            Ascc::HandleError(
                std::string("ASCEND_GLOBAL_LOG_LEVEL [") + logLevel + "] is invalid!");
            return ASCC_FAILURE;
        }
    }
    const char* stdoutFlag = getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    if (stdoutFlag != nullptr) {
        try {
            AsccGlobalEnvManager::ascendSlogPrintToStdout = std::stoi(stdoutFlag);
        } catch (...) {
            Ascc::HandleError(
                std::string("ASCEND_SLOG_PRINT_TO_STDOUT [") + stdoutFlag + "] is invalid!");
            return ASCC_FAILURE;
        }
    }
    const char* ascHomePath = getenv("ASCEND_HOME_PATH");
    if (ascHomePath != nullptr) {
        return envVar.Init(ascHomePath);
    }
    const char* ascAicpuPath = getenv("ASCEND_AICPU_PATH");
    if (ascAicpuPath != nullptr) {
        return envVar.Init(ascAicpuPath);
    }
    const char* toolChainPath = getenv("TOOLCHAIN_HOME");
    if (toolChainPath != nullptr) {
        std::string fullPath = Ascc::CheckAndGetFullPath(std::string(toolChainPath) + "/../");
        if (!fullPath.empty()) {
            return envVar.Init(fullPath.c_str());
        }
    }
    ASC_LOG_ASC_ERROR(INIT, "Environment path can not set! please source setenv.bash!");
    return ASCC_FAILURE;
}();
} // namespace Ascc