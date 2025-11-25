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
 * \file asc_info_manager.cpp
 * \brief
 */

#include "asc_info_manager.h"

#include <cctype>
#include <cinttypes>
#include <iostream>
#include <string>

#include "asc_log.h"

namespace AscPlugin {

namespace {
inline uint32_t GetMaxCoreNumImpl(const ShortSocVersion& socVersion)
{
    constexpr uint32_t coreNumNormal = 75;
    constexpr uint32_t coreNum91095 = 108;
    if (socVersion == ShortSocVersion::ASCEND910_95) {
        return coreNum91095;
    }
    if (socVersion == ShortSocVersion::ASCEND910B || socVersion == ShortSocVersion::ASCEND310P ||
        socVersion == ShortSocVersion::ASCEND910 || socVersion == ShortSocVersion::ASCEND310B) {
        return coreNumNormal;
    }
    return coreNumNormal;
}
// return index of first char that does not fit c++ variable naming rule.  return -1 means all chars are valid
// Example: string: AA-XXX return 2, means "-" is invalid.
int32_t GetFirstInvalidChar(const std::string& str, bool isUndef) {
    if (str.empty()) {
        return -1;
    }

    char first = str[0];  // first char must be alpha / digit
    if (!(isalpha(static_cast<unsigned char>(first)) || first == '_')) {
        return 0;
    }

    // valid ones must be: digit / alpha / _
    for (size_t i = 1; i < str.size(); ++i) {
        char c = str[i];
        if (!(isalpha(static_cast<unsigned char>(c)) || isdigit(static_cast<unsigned char>(c)) || c == '_')) {
            if (!isUndef && c != '=') {
                return static_cast<int>(i);
            } else if (isUndef) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

void UpdateCompileArgArray(std::vector<std::string>& argsArray, std::vector<std::string>::const_iterator& it)
{
    const std::string compileArg = *it;
    if (compileArg.size() > 2) { // Example: -lascendcl    macro -I / -L/ -l length is 2
        argsArray.emplace_back(compileArg);
    } else { // -D AAAA
        argsArray.emplace_back(compileArg);
        ++it;
        argsArray.emplace_back(*it);
    }
}
} // namespace

void InfoManager::ReportCompileArgs()
{
    for (const auto& def : compileArgs_.definitions) {
        ASC_LOGD("Compile args has definition: %s.", def.c_str());
    }
    for (const auto& hostDef : compileArgs_.hostDefinitions) {
        ASC_LOGD("Compile args has host definition: %s.", hostDef.c_str());
    }
    for (const auto& incPath : compileArgs_.includePaths) {
        ASC_LOGD("Compile args has include path: %s.", incPath.c_str());
    }
    for (const auto& opt : compileArgs_.options) {
        ASC_LOGD("Compile args has compile option: %s.", opt.c_str());
    }
    for (const auto& incFile : compileArgs_.includeFiles) {
        ASC_LOGD("Compile args has include file: %s.", incFile.c_str());
    }
    for (const auto& linkFile : compileArgs_.linkFiles) {
        ASC_LOGD("Compile args has link file: %s.", linkFile.c_str());
    }
    for (const auto& linkPath : compileArgs_.linkPaths) {
        ASC_LOGD("Compile args has link path: %s.", linkPath.c_str());
    }
}

void InfoManager::UpdateDefinitions(bool hasHostStart, std::vector<std::string>::const_iterator& it)
{
    constexpr uint32_t macroLen = 2;  // -D / -U
    const std::string compileArg = *it;
    std::string prefix, content;  // -D AAAA, -D is prefix, AAA is content
    if (compileArg.size() > macroLen) { // Example: -DAAAA
        prefix = compileArg.substr(0, macroLen);
        content = compileArg.substr(macroLen);
    } else { // -D AAAA
        prefix = compileArg;
        ++it;
        content = *it;
    }

    if (StartsWith(content, "GEN_ACLRT")) {
        SetAclrtHeaderPath(content.substr(10)); // GEN_ACLRT= length is 10.  GEN_ACLRT={header path}
    }

    // extract the macro part
    std::string macroContent = content;
    const int32_t invalidIndex = GetFirstInvalidChar(content, prefix == "-U");
    if (invalidIndex != -1) {
        macroContent = content.substr(0, invalidIndex);
    }

    if (!hasHostStart) {
        compileArgs_.definitions.emplace_back(prefix + content);
        if (prefix != "-D") { // when prefix is -U
            if (macroContent == "ASCENDC_DUMP") {
                userDumpStatus_ = true;
            } else if (macroContent == "HAVE_WORKSPACE") {
                hasWorkspace_ = false;
            } else if (macroContent == "HAVE_TILING") {
                hasTiling_ = false;
            } else if (macroContent == "ASCENDC_TIME_STAMP_ON") {
                hasTimeStamp_ = false;
            } else if (macroContent == "ASCENDC_DEBUG") {
                enableL2Cache_ = true;
            }
        } else {
            if (macroContent == "ASCENDC_DUMP=0") {
                userDumpStatus_ = false;
            } else if (macroContent == "ASCENDC_DUMP=1") {
                userDumpStatus_ = true;
            } else if (macroContent == "HAVE_WORKSPACE") {
                hasWorkspace_ = true;
            } else if (macroContent == "HAVE_TILING") {
                hasTiling_ = true;
            } else if (macroContent == "ASCENDC_TIME_STAMP_ON") {
                hasTimeStamp_ = true;
            } else if (macroContent == "ASCENDC_DEBUG") {
                enableL2Cache_ = false;
            }
        }
    } else {
        compileArgs_.hostDefinitions.emplace_back(prefix + content);
    }
}

void InfoManager::SetCompileArgs(const std::vector<std::string>& compileArgs)
{
    compileArgs_.includeFiles = {"-include", pathInfo_.cannVersionHeader};
    bool hasHostStart = false;   // for -Xhost-start
    for (auto it = compileArgs.begin(); it != compileArgs.end(); ++it) {
        std::string compileArg = *it;
        if (compileArg == "-Xhost-start") {
            hasHostStart = true;
        } else if (compileArg == "-Xhost-end") {
            hasHostStart = false;
        }

        // -D, -U, -I, -l, -L all supports -X XXX such as -D AAA
        if (StartsWith(compileArg, "-D") || StartsWith(compileArg, "-U")) {
            UpdateDefinitions(hasHostStart, it);
        } else if (StartsWith(compileArg, "-I")) {
            UpdateCompileArgArray(compileArgs_.includePaths, it);
        } else if (StartsWith(compileArg, "-l")) {
            UpdateCompileArgArray(compileArgs_.linkFiles, it);
        } else if (StartsWith(compileArg, "-L")) {
            UpdateCompileArgArray(compileArgs_.linkPaths, it);
        } else if (compileArg == "-O0" || compileArg == "-O1" || compileArg == "-O2" || compileArg == "-O3") {
            if (compileArg == "-O1") {
                optimizeLevel_ = "-O2";          // bisheng not support -O1
            } else {
                optimizeLevel_ = compileArg;
            }
        } else if (compileArg == "-include") {
            compileArgs_.includeFiles.emplace_back("-include");
            ++it;
            compileArgs_.includeFiles.emplace_back(*it);   // Note: -include xx, xx validity is checked by compiler
        } else if (compileArg == "-sanitizer") {
            enableL2Cache_ = false;
        } else if (compileArg == "--cce-auto-sync=off") {
            isAutoSyncOn_ = false;
        }

        if (StartsWith(compileArg, "--cce-aicore-input-parameter-size=")) {
            compileArgs_.options.emplace_back(compileArg);
        }
    }

    ReportCompileArgs();
}

void InfoManager::SetAclrtHeaderPath(const std::string& headerPath)
{
    aclrtLaunchHeaderPath_ = headerPath;
    ASC_LOGD("ACLRT_LAUNCH_KERNEL header path is set up as %s.", headerPath.c_str());
}

void InfoManager::SetCannPath(const std::string& cannPath)
{
    cannPath_ = cannPath;
    ASC_LOGD("CANN package path is set up as %s.", cannPath.c_str());
    pathInfo_ = PathInfo(cannPath);
}

void InfoManager::SetLogPath(const std::string& logPath)
{
    logPath_ = logPath;
}

void InfoManager::SetTempPath(const std::string& tempPath)
{
    tmpPath_ = tempPath;
}

void InfoManager::SetSourceFile(const std::string& sourceFile)
{
    sourceFile_ = sourceFile;
}

void InfoManager::SetSocVersion(const std::string& socVersion)
{
    if (!socVersion.empty()) {
        socVersion_ = socVersion;
    }
}

void InfoManager::SetShortSocVersion(const AscPlugin::ShortSocVersion socVersion)
{
    shortSocVersion_ = socVersion;
}

void InfoManager::SetOptimizeLevel(const std::string& optLevel)
{
    optimizeLevel_ = optLevel;
}

void InfoManager::SetSaveTempRequested(const bool saveTemp)
{
    saveTempRequested_ = saveTemp;
}

void InfoManager::SetUserDumpStatus(const bool dumpStatus)
{
    userDumpStatus_ = dumpStatus;
}

void InfoManager::SetHasPrintf(const bool hasPrintf)
{
    hasPrintf_ = hasPrintf;
}

void InfoManager::SetHasAssert(const bool hasAssert)
{
    hasAssert_ = hasAssert;
}

void InfoManager::SetOpSystemCfg(const bool hasOpSystemCfg)
{
    hasOpSystemCfg_ = hasOpSystemCfg;
}

void InfoManager::AddGlobalSymbolInfo(const std::string &mangling, const KernelMetaType &type,
    const std::string &fileName, const uint32_t lineNo, const uint32_t colNo, const KfcScene kfcScene)
{
    kernelFuncSymbolToFuncInfo_.emplace(mangling, std::make_tuple(type, fileName, lineNo, colNo, kfcScene));
}

void InfoManager::SetAscendMetaFlag(const uint32_t& flag)
{
    ascendMetaFlag_ |= flag;
    ASC_LOGI("Set meta section add flag 0x%02" PRIX32 ".", flag);
}

size_t InfoManager::SetAndGetMetaFlagCounter()
{
    metaFlagCounter_ += 1;
    return metaFlagCounter_;
}

void InfoManager::UpdateOneCoreDumpSize()
{
    constexpr uint32_t PRINTF_SIZE = 1048576;   // 1024M
    constexpr uint32_t ASSERT_SIZE = 1024;
    if (hasPrintf_) {
        oneCoreDumpSize_ = PRINTF_SIZE;
        return;
    }
    if (hasAssert_) {
        oneCoreDumpSize_ = ASSERT_SIZE;
    }
}

const PathInfo& InfoManager::GetPathInfo() const
{
    return pathInfo_;
}

const CompileArgs& InfoManager::GetCompileArgs() const
{
    return compileArgs_;
}

ShortSocVersion InfoManager::GetShortSocVersion() const
{
    return shortSocVersion_;
}

const std::string& InfoManager::GetAclrtHeaderPath() const
{
    return aclrtLaunchHeaderPath_;
}

const std::string& InfoManager::GetCannPath() const
{
    return cannPath_;
}

const std::string& InfoManager::GetLogPath() const
{
    return logPath_;
}

const std::string& InfoManager::GetTempPath() const
{
    return tmpPath_;
}

const std::string& InfoManager::GetSocVersion() const
{
    return socVersion_;
}

const std::string& InfoManager::GetOptimizeLevel() const
{
    return optimizeLevel_;
}

const std::string& InfoManager::GetSourceFile() const
{
    return sourceFile_;
}

const std::unordered_map<std::string, InfoManager::GlobalFuncInfo>& InfoManager::GetGlobalSymbolInfo() const
{
    return kernelFuncSymbolToFuncInfo_;
}

uint32_t InfoManager::GetAscendMetaFlag() const
{
    return ascendMetaFlag_;
}

uint32_t InfoManager::GetMaxCoreNum(const ShortSocVersion& socVersion) const
{
    return GetMaxCoreNumImpl(socVersion);
}

uint32_t InfoManager::GetMaxCoreNum() const
{
    return GetMaxCoreNumImpl(shortSocVersion_);
}

size_t InfoManager::GetMetaFlagCounter() const
{
    return metaFlagCounter_;
}
bool InfoManager::SaveTempRequested() const
{
    return saveTempRequested_;
}

bool InfoManager::UserDumpRequested() const
{
    return userDumpStatus_;
}

bool InfoManager::HasTimeStamp() const
{
    return hasTimeStamp_;
}

bool InfoManager::HasWorkspace() const
{
    return hasWorkspace_;
}

bool InfoManager::HasTiling() const
{
    return hasTiling_;
}

bool InfoManager::HasPrintf() const
{
    return hasPrintf_;
}

bool InfoManager::HasAssert() const
{
    return hasAssert_;
}

bool InfoManager::IsDumpOn() const
{
    return userDumpStatus_ && (hasPrintf_ || hasAssert_);
}

bool InfoManager::IsFifoDumpOn() const
{
    return IsSupportFifoDump() && (IsDumpOn() || hasTimeStamp_);
}

uint32_t InfoManager::GetOneCoreDumpSize() const
{
    return oneCoreDumpSize_;
}

bool InfoManager::IsL2CacheEnabled() const
{
    return enableL2Cache_;
}

bool InfoManager::HasOpSystemCfg() const
{
    return hasOpSystemCfg_;
}

void InfoManager::SetFirstKernel(const bool isFirstKernel)
{
    isFirstKernel_ = isFirstKernel;
}

bool InfoManager::IsFirstKernel() const
{
    return isFirstKernel_;
}

bool InfoManager::IsAutoSyncOn() const
{
    return isAutoSyncOn_;
}

bool InfoManager::IsSupportFifoDump() const
{
    return shortSocVersion_ == ShortSocVersion::ASCEND910B;
}

} // namespace AscPlugin