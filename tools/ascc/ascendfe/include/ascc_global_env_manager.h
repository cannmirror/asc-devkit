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
 * \file ascc_global_env_manager.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_GLOBAL_ENV_MANAGER_H__
#define __INCLUDE_ASCC_GLOBAL_ENV_MANAGER_H__
#include <string>

#include "ascc_types.h"

namespace Ascc {
class AsccGlobalEnvManager {
public:
    inline static AsccGlobalEnvManager& GetInstance()
    {
        static AsccGlobalEnvManager instance;
        return instance;
    }
    uint32_t Init(const char* cannPath);
public:
    static inline uint32_t ascendSlogPrintToStdout = 0;
    static inline uint32_t ascendGlobalLogLevel = 0;
    bool initSuccess = false;
    bool needSaveTmpFile = false;                    // whether need to store autogen files in envVar.asccTmpPath

    uint32_t mixNumLineNum;              // tell which line in preprocess files has AscendC::MIX_NUM =
    uint32_t dumpWorkspaceLineNum;       // tell which line in preprocess files has const uint32_t DUMP_WORKSPACE_SIZE =
    uint32_t dumpUintLineNum;            // tell which line in preprocess files has constexpr size_t DUMP_UINTSIZE =
    std::string ascendCannPackagePath;
    std::string ascendCannIncludePath;
    std::string ascendHighlevelApiPath;
    std::string ascendHighlevelApiLibPath;
    std::string ascendHighlevelApiLibMatmulPath;
    std::string ascendHighlevelApiImplPath;
    std::string ascendHostApiPath;
    std::string ascendTikcfwPath;
    std::string ascendTikcfwLibPath;
    std::string ascendTikcfwLibMatmulPath;
    std::string ascendTikcfwImplPath;
    std::string ascendTikcfwInterfacePath;
    std::string ascendTikcpulibPath;
    std::string ascendClangIncPath;
    std::string ascendVersionHeader;

    std::string ccecPath;
    std::string ascendCompiler;
    std::string ascendLinker;
    std::string ascendAr;

    std::string asccTmpPath;
    std::string asccTmpIncludePath;
    std::string asccTmpAutoGenPath;
    std::string asccTmpHostGenPath;
    std::string asccTmpDependPath;
    std::string asccMergeObjPath;
    std::string asccMergeObjFinalPath;
    std::string asccCompileLogPath;

    std::string currentPath;
    std::string cppCompilerPath;    // c++ compiler path
    std::string ldPath;             // ld path
    std::string hostCompileFile;    // host compile file
    std::string kernelCallStubFile; // host compile input file
    CompileArgs commonDeviceArgs;
    CompileArgs commonHostArgs;
    CompileArgs astAnalysisArgs;
private:
    AsccGlobalEnvManager() = default;
    ~AsccGlobalEnvManager() = default;
    AsccGlobalEnvManager(const AsccGlobalEnvManager&) = delete;
    AsccGlobalEnvManager& operator=(const AsccGlobalEnvManager&) = delete;
    AsccGlobalEnvManager(AsccGlobalEnvManager&&) = delete;
    AsccGlobalEnvManager& operator=(AsccGlobalEnvManager&&) = delete;

    void InitDeviceCommonOption();
    void InitHostCommonOption();
    AsccStatus ValueCheck() const;
    void PrintOutInfo() const;
};

}
#endif // __INCLUDE_ASCC_GLOBAL_ENV_MANAGER_H__