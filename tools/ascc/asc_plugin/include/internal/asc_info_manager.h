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
 * \file asc_info_manager.h
 * \brief
 */


#ifndef __INCLUDE_INTERNAL_ASC_INFO_MANAGER_H__
#define __INCLUDE_INTERNAL_ASC_INFO_MANAGER_H__

#include <tuple>
#include <sys/utsname.h>

#include "asc_ast_utils.h"
#include "asc_utils.h"

namespace AscPlugin {

struct PathInfo {
    PathInfo() = default;

    PathInfo(const std::string& cannPathIn)
    {
        struct utsname info;
        std::string prefix;
        if (uname(&info) < 0) {
            prefix = "/compiler";
        } else {
            std::string machine = info.machine;
            if (machine == "x86_64") {
                prefix = "/x86_64-linux";
            } else if (machine == "aarch64" || machine == "arm64" || machine == "arm") {
                prefix = "/aarch64-linux";
            } else {
                prefix = "/compiler";
            }
        }
        cannPath = cannPathIn + prefix;

        cannIncludePath = {
            cannPath + "/include",

            cannPath + "/include/ascendc/host_api",
            cannPath + "/ascendc/include/highlevel_api",
            cannPath + "/tikcpp/tikcfw",
            cannPath + "/tikcpp/tikcfw/lib",
            cannPath + "/tikcpp/tikcfw/lib/matmul",
            cannPath + "/tikcpp/tikcfw/impl",
            cannPath + "/tikcpp/tikcfw/interface",

            cannPath + "/asc/impl/adv_api",
            cannPath + "/asc/impl/basic_api",
            cannPath + "/asc/impl/c_api",
            cannPath + "/asc/impl/micro_api",
            cannPath + "/asc/impl/simt_api",
            cannPath + "/asc/impl/utils",
            cannPath + "/asc",

            cannPath + "/asc/include",
            cannPath + "/asc/include/adv_api",
            cannPath + "/asc/include/adv_api/matmul",
            cannPath + "/asc/include/aicpu_api",
            cannPath + "/asc/include/basic_api",
            cannPath + "/asc/include/c_api",
            cannPath + "/asc/include/interface",
            cannPath + "/asc/include/micro_api",
            cannPath + "/asc/include/simt_api",
            cannPath + "/asc/include/tiling",
            cannPath + "/asc/include/utils"
        };

        std::string expectedCannVersionHeader = cannPath + "/include/version/asc_devkit_version.h";
        if (PathCheck(expectedCannVersionHeader.c_str(), false) != PathStatus::NOT_EXIST) {
            cannVersionHeader = cannPath + "/include/version/asc_devkit_version.h";
        }
        ascendClangIncludePath = cannPath + "/ccec_compiler/lib/clang/15.0.5/include";
        bishengPath = cannPath + "/ccec_compiler/bin/bisheng";
        objdumpPath = cannPath + "/ccec_compiler/bin/llvm-objdump";
    }

    std::string cannPath;
    std::vector<std::string> cannIncludePath;
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
    void SetOpSystemCfg(const bool hasOpSystemCfg);
    void AddGlobalSymbolInfo(const std::string &mangling, const KernelMetaType &type, const std::string &fileName,
        const uint32_t lineNo, const uint32_t colNo, const KfcScene kfcScene);
    void ReportCompileArgs();
    uint32_t SetKernelFuncFlag();

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
    bool SaveTempRequested() const;
    bool IsL2CacheEnabled() const;
    bool HasOpSystemCfg() const;
    bool IsAutoSyncOn() const;
    bool HasKernelFunc() const;

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
    bool enableL2Cache_ = true;                 // default enable
    bool hasOpSystemCfg_ =false;
    bool isAutoSyncOn_ = true;
    bool hasKernelFunc_ = false;

    // global func mangling name to tuple < ktype, filename, lineNo, colNo >
    std::unordered_map<std::string, GlobalFuncInfo> kernelFuncSymbolToFuncInfo_;
};

} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_INTERFACE_H__