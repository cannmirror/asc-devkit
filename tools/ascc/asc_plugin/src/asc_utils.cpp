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
 * \file asc_utils.cpp
 * \brief
 */

#include "asc_utils.h"
#include "asc_log.h"

#include <climits>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <ctime>
#include <string>
#include <utility>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <memory>
#include <sys/wait.h>

namespace AscPlugin {

namespace {
std::string GetCurrentDirectory(void)
{
    char buffer[PATH_MAX];
    if (getcwd(buffer, sizeof(buffer)) == nullptr) {
        return std::string();
    }
    return std::string(buffer);
}
} // namespace

PathStatus PathCheck(const char* path, bool needCheck)
{
    if (access(path, W_OK) == 0) {
        return PathStatus::WRITE;
    }
    if (access(path, R_OK) == 0) {
        return PathStatus::READ;
    }
    if (access(path, F_OK) == 0) {
        return PathStatus::EXIST;
    }
    if (needCheck) {
        ASC_LOGE("Path [%s]:  no such file or directory.", path);
    } else {
        ASC_LOGI("Path [%s]:  no such file or directory.", path);
    }
    return PathStatus::NOT_EXIST;
}

std::string CheckAndGetFullPath(const char* path)
{
    if (PathCheck(path, true) == PathStatus::NOT_EXIST) {
        ASC_LOGE("Path [%s]: path not exist.", path);
        return std::string();
    }
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(path, resolvedPath) == nullptr) {
        ASC_LOGE("Path [%s]:  realpath failed.", path);
        return std::string();
    }
    return resolvedPath;
}

std::string CheckAndGetFullPath(const std::string& path)
{
    return CheckAndGetFullPath(path.c_str());
}

std::string GetFilePath(const std::string& filePath)
{
    size_t pos = filePath.find_last_of("/\\");
    if (pos == std::string::npos) {
        return GetCurrentDirectory();
    }
    return CheckAndGetFullPath(filePath.substr(0, pos).c_str());
}

bool StartsWith(const std::string& srcStr, const std::string& prefix)
{
    if (prefix.size() > srcStr.size()) {
        return false;
    }
    return srcStr.compare(0, prefix.size(), prefix) == 0;
}

bool EndsWith(const std::string& srcStr, const std::string& suffix)
{
    if (suffix.size() > srcStr.size()) {
        return false;
    }
    return srcStr.compare(srcStr.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string GenerateTimestamp()
{
    std::time_t now = std::time(nullptr);
    struct tm tmBuf;
    if (!localtime_r(&now, &tmBuf)) {
        ASC_LOGE("Timestamp: convert to local time failed.");
        return "";
    }

    char timestamp[80];  // set up length 80 to store timestamp info
    if (strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &tmBuf) == 0) {
        ASC_LOGE("Timestamp: strftime format failed.");
        return "";
    }
    return std::string(timestamp);
}

// Assume userFilePath is A, srcFilePath is test/add_custom.cpp
// Expected path will be : A/add_custom_[timestamp]_[pid]_[tid]/[folderTypeName]
std::string GetTempFolder(const std::string& userFilePath, const std::string& srcFilePath, const std::string& timeTag,
    const std::string& folderTypeName)
{
    std::string baseFolderPath = userFilePath.empty() ? "/tmp/asc_plugin" : userFilePath;

    // get source file prefix
    size_t pos = srcFilePath.find_last_of("/\\");
    std::string fileName = srcFilePath;
    std::string fileNamePrefix = "temp_";
    if (pos != std::string::npos) {
        fileName = srcFilePath.substr(pos + 1);
    }
    size_t lastdot = fileName.find_last_of(".");
    if (lastdot != std::string::npos) {
        fileNamePrefix = fileName.substr(0, lastdot);
    } else {
        fileNamePrefix = fileName;
    }
    std::stringstream ss;
    ss << baseFolderPath << "/" << fileNamePrefix << "_" << timeTag << "_" << getpid() << "_" << syscall(SYS_gettid)
        << "/" << folderTypeName;
    return ss.str();
}

int32_t ExecMkdir(struct stat& fileStatus, const char* dirPath)
{
    if (stat(dirPath, &fileStatus) != 0) {
        // need execute permission for directories
        int mkdirRes = mkdir(dirPath, S_IRWXU | S_IRWXG );
        if (mkdirRes != 0 ) {
            if (errno == EEXIST) {
                return ASC_SUCCESS;
            } else {
                ASC_LOGE("Directory: create directory failed.");
                return ASC_FAILURE;
            }
        }
    }
    return ASC_SUCCESS;
}

// Example dir:  /tmp/bishengcc/tmp_log
int32_t CreateDirectory(const std::string& dirPath)
{
    if (dirPath.empty()) {
        ASC_LOGE("Directory: dirPath is empty in CreateDirectory.");
        return ASC_FAILURE;
    }

    size_t pos = 0;
    std::string currentDir = dirPath;
    std::string currentFullDir;

    // deal with absolute path
    if (dirPath[0] == '/') {
        currentFullDir = "/";
        currentDir = dirPath.substr(1);
        pos = 0;
    }

    struct stat fileStatus;
    // for loop to search "/": in Example dir above, deal with dir [tmp], dir [bishengcc]
    while ((pos = currentDir.find('/', pos)) != std::string::npos) {
        std::string subDir = currentDir.substr(0, pos);
        if (subDir.empty()) {
            pos++;
            continue;
        }

        currentFullDir += subDir + "/";
        if (ExecMkdir(fileStatus, currentFullDir.c_str()) != ASC_SUCCESS) {
            ASC_LOGE("Directory: create directory failed.");
            return ASC_FAILURE;
        }
        // update currentdir to the rest part, and reset pos
        currentDir = currentDir.substr(pos + 1);
        pos = 0;
    }

    // in Example dir above, deal with dir [tmp_log]
    currentFullDir += currentDir;
    if (ExecMkdir(fileStatus, currentFullDir.c_str()) != ASC_SUCCESS) {
        ASC_LOGE("Directory: create directory failed.");
        return ASC_FAILURE;
    }
    return ASC_SUCCESS;
}

std::pair<std::string, int32_t> ExecuteCommand(const char *command)
{
    if (!command) {
        ASC_LOGE("Command is nullptr");
        return {"", -1};
    }

    FILE *pipe = popen(command, "r");
    if (!pipe) {
        ASC_LOGE("popen failed for command '%s': %s", command, strerror(errno));
        return {"", -1};
    }
    std::string output;
    char buffer[2048];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output.append(buffer);
    }
    if (ferror(pipe) != 0) {
        ASC_LOGE("Error reading output for command '%s': %s", command, strerror(errno));
        pclose(pipe);
        return {"", -1};
    }
    int status = pclose(pipe);
    if (status == -1) {
        ASC_LOGE("pclose failed for command '%s': %s", command, strerror(errno));
        return {"", -1};
    }
    unsigned uStatus = static_cast<unsigned>(status);
    if (WIFEXITED(uStatus)) {
        return {output, WEXITSTATUS(uStatus)};
    } else if (WIFSIGNALED(uStatus)) {
        ASC_LOGE("Command '%s' terminated by signal: %d", command, static_cast<int>(WTERMSIG(uStatus)));
        return {"", -1};
    } else {
        ASC_LOGE("Command '%s' did not terminate normally", command);
        return {"", -1};
    }
}

int32_t ExecuteCompile(const std::string &cmd)
{
    ASC_LOGI("Compile cmd: [%s].", cmd.c_str());
    std::string output = "DEFAULT";
    int returnCode = 0;
    std::tie(output, returnCode) = ExecuteCommand((cmd + " 2>&1").c_str());
    if (returnCode != 0) {
        ASC_LOGE("Function %s at line %d: Command: [%s] execution failed, returnCode is %d!",
            __FUNCTION__, __LINE__, cmd.c_str(), returnCode);
        ASC_LOGD("Output of ascendc_compiler:%s", output.c_str());
        return ASC_FAILURE;
    }
    return ASC_SUCCESS;
}

std::vector<std::string> SplitLines(const std::string& str) {
    std::vector<std::string> lines;
    std::stringstream ss(str);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}


KernelMetaType GetBishengKTypeByCoreRatio(const CoreRatio& ratio, const KernelMetaType& defaultKType)
{
    if (ratio.cubeNum == 1) {
        if (ratio.vecNum == 0) {
            return KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0;
        } else if (ratio.vecNum == 1) {
            return KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1;
        } else if (ratio.vecNum == 2) {  // aic num 1, aiv num 2
            return KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2;
        }
    }
    if (ratio.vecNum == 1 && ratio.cubeNum == 0) {
        return KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0;
    }
    if (ratio.isCoreRatio) {
        ASC_LOGE("Invalid core ratio: cubeNum %u, vecNum %u", ratio.cubeNum, ratio.vecNum);
    }
    return defaultKType;
}

KernelMetaType ExtractKernelType(const std::unordered_set<KernelMetaType> kTypeSet)
{
    // If kTypeSet has multiple kernel type, it means involving core ratio(cube, vec)
    // It will need template instance's ratio to confirm the exact kernel type.
    // Otherwise, it will always has one kernel type or return Max when empty
    if (kTypeSet.size() == 0) {
        return KernelMetaType::KERNEL_TYPE_MAX;
    }
    auto it = kTypeSet.begin();
    return *it;
}

std::string ToUpper(const std::string& str)
{
    std::string tmpStr(str);
    std::transform(tmpStr.begin(), tmpStr.end(), tmpStr.begin(), ::toupper);
    return tmpStr;
}

} // namespace AscPlugin