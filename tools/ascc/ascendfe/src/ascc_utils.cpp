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
 * \file ascc_utils.cpp
 * \brief
 */
#include "ascc_utils.h"

#include <fstream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstring>
#include <climits>
#include <sys/wait.h>
#include <regex>
#include <sys/stat.h>
#include <unistd.h>

#include "ascc_global_env_manager.h"
#include "ascc_log.h"

namespace Ascc {
bool IsPathLegal(const std::string& path) {
    std::regex illegal(R"([\x00-\x1F*?\"<>|]|\.\./|\.\.\\)");
    return !std::regex_search(path, illegal);
}

bool IsParentDirValid(const std::string& path) {
    size_t pos = path.find_last_of("/");
    std::string parent = (pos != std::string::npos) ? path.substr(0, pos) : ".";
    struct stat info;
    if (stat(parent.c_str(), &info) != 0) {
        return false;
    }
    return S_ISDIR(info.st_mode);
}

PathStatus PathCheck(const char* path, bool needCheck)
{
    if (access(path, W_OK) == 0) {
        return PathStatus::WRITE;
    }
    ASC_LOG_ASC_WARN(UTILS, "Path [%s] : write permission denied!", path);
    if (access(path, R_OK) == 0) {
        return PathStatus::READ;
    }
    ASC_LOG_ASC_WARN(UTILS, "Path [%s] : read permission denied!", path);
    if (access(path, F_OK) == 0) {
        return PathStatus::EXIST;
    }
    if (needCheck) {
        ASC_LOG_ASC_ERROR(UTILS, "Path [%s] : no such file or directory!", path);
    } else {
        ASC_LOG_ASC_WARN(UTILS, "Path [%s] : no such file or directory!", path);
    }
    return PathStatus::NOT_EXIST;
}

std::string CheckAndGetFullPath(const char* path)
{
    if (PathCheck(path, true) == PathStatus::NOT_EXIST) {
        return std::string();
    }
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(path, resolvedPath) == nullptr) {
        ASC_LOG_ASC_ERROR(UTILS, "Path [%s] realpath fail!", path);
        return std::string();
    }
    return resolvedPath;
}

std::string CheckAndGetFullPath(const std::string& path)
{
    return CheckAndGetFullPath(path.c_str());
}

std::string GetCurrentDirectory(void)
{
    char buffer[1024];
    if (getcwd(buffer, sizeof(buffer)) == nullptr) {
        ASC_LOG_ASC_ERROR(UTILS, "Get current directory failed!");
        return std::string();
    }
    return std::string(buffer);
}

std::string GetFilePath(const std::string &filePath)
{
    size_t pos = filePath.find_last_of("/\\");
    if (pos == std::string::npos) {
        return Ascc::AsccGlobalEnvManager::GetInstance().currentPath;
    }
    return CheckAndGetFullPath(filePath.substr(0, pos));
}

std::string ToUpper(const std::string &str)
{
    std::string tmpStr(str);
    std::transform(tmpStr.begin(), tmpStr.end(), tmpStr.begin(), ::toupper);
    return tmpStr;
}

std::string ToLower(const std::string &str)
{
    std::string tmpStr(str);
    std::transform(tmpStr.begin(), tmpStr.end(), tmpStr.begin(), ::tolower);
    return tmpStr;
}

void SaveCompileLogFile(const std::string &note, const std::string &content)
{
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    std::string filePath = envVar.asccTmpPath + "/compile_log";
    filePath = CheckAndGetFullPath(filePath) + "/compile.log";
    if (!Ascc::IsPathLegal(filePath) || !Ascc::IsParentDirValid(filePath)) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "filePath [%s] does not exist!", filePath.c_str());
        return;
    }
    std::ofstream file(filePath, std::ios::app);
    std::string noteRes = "# " + note + "\n";
    std::string contentRes = content + "\n\n";
    file.write(noteRes.c_str(), noteRes.size());
    file.write(contentRes.c_str(), contentRes.size());
    file.close();
}

const char **ConvertStringVecToCStringVec(std::vector<const char*>& charVec, const std::vector<std::string>& strVec)
{
    charVec.resize(strVec.size());
    for (size_t i = 0; i < strVec.size(); ++i) {
        charVec[i] = strVec[i].c_str();
    }
    return charVec.data();
}
}