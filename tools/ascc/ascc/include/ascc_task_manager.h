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
 * \file ascc_task_manager.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_TASK_MANAGER_H__
#define __INCLUDE_ASCC_TASK_MANAGER_H__
#include <string>
#include <vector>
#include "ascc_common_types.h"
#include "ascc_utils.h"
#include "task.h"

namespace Ascc {

// path for executable / storage
struct PathInfo {
    PathInfo() {}

    PathInfo(const std::string& ascendFePathIn, const std::string& bishengPathIn, const std::string& gppPathIn,
        const std::string& cannPathIn, const std::string& tmpFilePathIn, const std::vector<std::string>& linkPathIn)
        : ascendFePath(ascendFePathIn), bishengPath(bishengPathIn), gppPath(gppPathIn), cannPath(cannPathIn),
        tmpFilePath(tmpFilePathIn), linkPath(linkPathIn)
    {}

    std::string ascendFePath;
    std::string bishengPath;
    std::string gppPath;
    std::string ldlldPath;                  // ld.lld
    std::string cannPath;
    std::string ascPackKernelPath;          // ascendc_pack_kernel
    std::string tmpFilePath;                // final directory to store generated files
    std::string includePath;                // files to replace __aicore__ and GM_ADDR
    std::string objectFilePath;             // store .o files
    std::string preprocessDevicePath;       // store xxx_vec/cube_ii.cpp by -E
    std::string preprocessHostPath;         // store xxx_host_ii.cpp by -E
    std::string deviceStubPath;             // path to store stub_xx.cpp + device_stub_xx.cpp
    std::string hostStubPath;               // path to store host_stub.cpp
    std::vector<std::string> linkPath;
};

struct ArgInfo {
    ArgInfo() {}

    std::string socVersion;
    std::string optLevel;                   // users optimization level, should not be used directly by bisheng
    std::string bishengOptLevel;            // bisheng not support O1, thus use O2 instead
    std::string outputFileName;
    std::string tmpPath;                    // tmp path passed by users
    bool needSaveTemps = false;
    bool verboseStatus = false;
    bool timeStatus = false;
    bool hasUsersDump = false;              // whether has users defined ASCENDC_DUMP
    OutputFileType outputMode = OutputFileType::FILE_EXECUTABLE;
};

const std::string FindExecPath(std::string execName);
const std::string GetSocForSimulator(const std::string& socVersion);
const std::string GetCoreTypeStr(const CoreType coreType);
AsccStatus GetJsonInfo(std::unordered_map<uint8_t, std::vector<std::string>>& kernelType,
    std::unordered_map<uint8_t, uint32_t>& mixNumLineMap, std::unordered_map<uint8_t, uint32_t>& dumpUintLineMap,
    std::unordered_map<uint8_t, uint32_t>& dumpWorkspaceLineMap, std::unordered_map<uint8_t, uint32_t>& dumpSizeMap,
    std::unordered_map<uint8_t, std::string>& stubFileMap, const std::string& jsonName);
AsccStatus InitArgInfo(ArgInfo& argInfo);
AsccStatus InitPathInfo(PathInfo& pathInfo, const ArgInfo& argInfo);
AsccStatus CreateTmpDirectory(const std::vector<std::string>& files, const PathInfo& pathInfo);
AsccStatus CheckFiles(const std::vector<std::string> files, const ArgInfo& argInfo);
AsccStatus ProcessFiles(const std::vector<std::string>& files, const PathInfo& pathInfo, const ArgInfo& argInfo);

int64_t GetDeviceFileSize(const std::string& basePath, const std::string& suffix);
const std::string GetReplaceCmd(const std::string& filename, const std::string& replaceContent, const uint32_t lineNo);
AsccStatus ReplaceStubFile(const std::string& fileName, CodeMode kernelTypeEnum, const uint32_t mixNumLineNo,
    const uint32_t dumpUintLineNo, const uint32_t dumpWorkspaceLineNo, const uint32_t dumpSize);
std::string GetDeviceStubMacro(const CoreType coreType, const std::string& mangleName, const CodeMode codeMode);
std::string GetInputFileStubDirName(const std::string& file);

const std::vector<std::string> GetCompileOptionContent(const std::vector<std::string>& compileOptions);
const std::vector<std::string> GetDependencyCompileOptionContent(const std::string& filename);
const std::vector<std::string> GetLinkCompileOptionContent();
std::vector<std::string> GetDeviceDefaultCompileOption(const ArgInfo& argInfo);
std::vector<std::string> GetHostDefaultIncludePath(const PathInfo& pathInfo);
std::vector<std::string> GetDeviceDefaultIncludePath(const PathInfo& pathInfo, const std::string& file);

const std::string GetPreprocessFileName(const PathInfo& pathInfo, const std::string& file, const std::string& coreType);
const std::string GetStubFileDirectory(const PathInfo& pathInfo, const std::string& file, const std::string& coreType);
std::string GetHostStubFileName(const std::string& fileName);
std::string GetDeviceStubFileName(CodeMode kernelTypeEnum, const CoreType coreType);
const std::string GetDeviceJsonName(const std::string& fileName, const std::string& coreType);
const std::string GetObjFileDirectory(const PathInfo& pathInfo, const std::string& srcFile);
bool HasJsonInStubDir(const std::string& file, const PathInfo& pathInfo);

Task TaskPreprocessDevice(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo,
    const CoreType coreType);
Task TaskPreprocessHost(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo);
Task TaskDeviceStub(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo, const CoreType coreType);
Task TaskHostStub(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo);
AsccStatus TaskPreprocessLaunch(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo);
AsccStatus TaskHostStubCompileO(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo);
AsccStatus TaskPureHostCompileO(std::vector<std::string>& objFile, const std::string& file, const PathInfo& pathInfo,
    const ArgInfo& argInfo);
AsccStatus TaskDeviceStubCompileO(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo,
    const CoreType coreType);
AsccStatus TaskMergeDeviceObjFiles(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo);
AsccStatus TaskPackKernel(std::vector<std::string>& objFile, const std::string& file, const PathInfo& pathInfo,
    const ArgInfo& argInfo);
AsccStatus TaskSharedLibrary(const std::vector<std::string>& objFile, const PathInfo& pathInfo, const ArgInfo& argInfo);
AsccStatus TaskExecutable(const std::vector<std::string>& objFile, const PathInfo& pathInfo, const ArgInfo& argInfo);

void UpdateSanitizerProcess(Task& task, const PathInfo& pathInfo);
void AddLdLldTask(TaskGroup& taskGroup, const PathInfo& pathInfo, const std::vector<std::string> files,
    const std::string& objFilePath, const std::string& coreType);
}
#endif // __INCLUDE_ASCC_TASK_MANAGER_H__