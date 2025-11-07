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
 * \file asc_info_manager.h
 * \brief
 */


#ifndef __INCLUDE_INTERNAL_ASC_INFO_MANAGER_H__
#define __INCLUDE_INTERNAL_ASC_INFO_MANAGER_H__

#include <tuple>

#include "asc_ast_utils.h"
#include "asc_utils.h"

namespace AscPlugin {

struct PathInfo {
    PathInfo() = default;

    PathInfo(const std::string& cannPathIn)
    {
        cannPath = cannPathIn;
        cannIncludePath = cannPathIn + "/include";
        // highlevel
        highLevelApiPath = cannPathIn + "/compiler/ascendc/include/highlevel_api";
        highLevelApiLibPath = cannPathIn + "/compiler/ascendc/include/highlevel_api/lib";
        highLevelApiLibMatmulPath = cannPathIn + "/compiler/ascendc/include/highlevel_api/lib/matmul";
        highLevelApiImplPath = cannPathIn + "/compiler/ascendc/include/highlevel_api/impl";
        // tikcfw
        tikcfwPath = cannPathIn + "/compiler/tikcpp/tikcfw";
        tikcfwInterfacePath = cannPathIn + "/compiler/tikcpp/tikcfw/interface";
        tikcfwLibPath = cannPathIn + "/compiler/tikcpp/tikcfw/lib";
        tikcfwLibMatmulPath = cannPathIn + "/compiler/tikcpp/tikcfw/lib/matmul";
        tikcfwImplPath = cannPathIn + "/compiler/tikcpp/tikcfw/impl";
        // hostapi
        hostApiPath = cannPathIn + "/include/ascendc/host_api";
        // cann version
        cannVersionHeader = cannPathIn + "/include/version/cann_version.h";

        ascendClangIncludePath = cannPathIn + "/compiler/ccec_compiler/lib/clang/15.0.5/include";
        bishengPath = cannPathIn + "/compiler/ccec_compiler/bin/bisheng";
        objdumpPath = cannPathIn + "/compiler/ccec_compiler/bin/llvm-objdump";
    }

    std::string cannPath;
    std::string cannIncludePath;
    std::string highLevelApiPath;
    std::string highLevelApiLibPath;
    std::string highLevelApiLibMatmulPath;
    std::string highLevelApiImplPath;
    std::string tikcfwPath;
    std::string tikcfwInterfacePath;
    std::string tikcfwLibPath;
    std::string tikcfwLibMatmulPath;
    std::string tikcfwImplPath;
    std::string hostApiPath;
    std::string cannVersionHeader;
    std::string ascendClangIncludePath;
    std::string bishengPath;
    std::string objdumpPath;
};

class InfoManager {
public:
    inline static InfoManager& GetInstance() {
        static InfoManager instance;
        return instance;
    }

    using GlobalFuncInfo = std::tuple<KernelMetaType, std::string, uint32_t, uint32_t, KfcScene>;
    void SetSourceFile(const std::string& sourceFile);
    void UpdateDefinitions(bool hasHostStart, std::vector<std::string>::const_iterator& it);
    void SetCompileArgs(const std::vector<std::string>& compileArgs);
    void SetAclrtHeaderPath(const std::string& headerPath);
    void SetCannPath(const std::string& cannPath);        // update cannPath_ + pathInfo_
    void SetTempPath(const std::string& tempPath);
    void SetLogPath(const std::string& logPath);
    void SetSocVersion(const std::string& socVersion);
    void SetShortSocVersion(const AscPlugin::ShortSocVersion socVersion);
    void SetOptimizeLevel(const std::string& optLevel);
    void SetSaveTempRequested(const bool saveTemp);
    void SetUserDumpStatus(const bool dumpStatus);
    void SetHasPrintf(const bool hasPrintf);
    void SetHasAssert(const bool hasAssert);
    void SetOpSystemCfg(const bool hasOpSystemCfg);
    void AddGlobalSymbolInfo(const std::string &mangling, const KernelMetaType &type, const std::string &fileName,
        const uint32_t lineNo, const uint32_t colNo, const KfcScene kfcScene);
    void UpdateOneCoreDumpSize();                       // must be called after hasPrintf_ and hasAssert_ is updated
    void SetFirstKernel (const bool isFirstKernel);
    void SetAscendMetaFlag(const uint32_t& flag);
    size_t SetAndGetMetaFlagCounter();
    void ReportCompileArgs();

    const PathInfo& GetPathInfo() const;
    const CompileArgs& GetCompileArgs() const;
    ShortSocVersion GetShortSocVersion() const;
    const std::string& GetAclrtHeaderPath() const;
    const std::string& GetCannPath() const;
    const std::string& GetLogPath() const;
    const std::string& GetTempPath() const;             // path for saving temp files
    const std::string& GetSocVersion() const;
    const std::string& GetOptimizeLevel() const;
    const std::string& GetSourceFile() const;
    const std::unordered_map<std::string, GlobalFuncInfo>& GetGlobalSymbolInfo() const;
    uint32_t GetAscendMetaFlag() const;
    size_t GetMetaFlagCounter() const;
    bool SaveTempRequested() const;
    bool UserDumpRequested() const;
    bool HasTimeStamp() const;
    bool HasWorkspace() const;
    bool HasTiling() const;
    bool HasPrintf() const;
    bool HasAssert() const;
    bool IsDumpOn() const;   // when user not pass -DASCENDC_DUMP=0, and uses printf/ assert
    uint32_t GetOneCoreDumpSize() const;                // for -DONE_CORE_DUMP_SIZE=xxx
    bool IsL2CacheEnabled() const;
    bool HasOpSystemCfg() const;
    bool IsFirstKernel() const;
    bool IsAutoSyncOn() const;

private:
    InfoManager() = default;
    ~InfoManager() = default;
    InfoManager(const InfoManager&) = delete;
    InfoManager& operator=(const InfoManager&) = delete;
    InfoManager(InfoManager&&) = delete;
    InfoManager& operator=(InfoManager&&) = delete;

    PathInfo pathInfo_;
    CompileArgs compileArgs_;
    ShortSocVersion shortSocVersion_ = ShortSocVersion::ASCEND910B;   // example: Ascend910B / Ascend310P
    std::string aclrtLaunchHeaderPath_;                               // for ACLRT_LAUNCH_KERNEL
    std::string cannPath_;
    std::string tmpPath_;
    std::string logPath_;
    std::string socVersion_;                                          // example: Ascend910B1 / Ascend310P1
    std::string optimizeLevel_ = "-O3";
    std::string sourceFile_;
    bool saveTempRequested_ = false;
    bool userDumpStatus_ = true;                // if user passed -DASCENDC_DUMP, then update. True means = 1
    bool hasTimeStamp_ = false;                 // for -DASCENDC_TIME_STAMP_ON
    bool hasWorkspace_ = false;                 // for -DHAVE_WORKSPACE in KernelLaunch
    bool hasTiling_ = false;                    // for -DHAVE_TILING in KernelLaunch
    bool hasPrintf_ = false;
    bool hasAssert_ = false;
    bool enableL2Cache_ = true;                 // default enable
    bool hasOpSystemCfg_ =false;
    uint32_t oneCoreDumpSize_ = 1048576;        // 1024 K
    bool isFirstKernel_ = false;
    bool isAutoSyncOn_ = true;
    uint32_t ascendMetaFlag_ = 0;
    size_t metaFlagCounter_ = 0;

    // global func mangling name to tuple < ktype, filename, lineNo, colNo >
    std::unordered_map<std::string, GlobalFuncInfo> kernelFuncSymbolToFuncInfo_;
};

} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_INTERFACE_H__