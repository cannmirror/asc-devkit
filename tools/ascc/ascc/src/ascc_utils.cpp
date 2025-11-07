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
#include <climits>
#include <ctime>
#include "ascc_mlog.h"

namespace Ascc {

PathStatus PathCheck(const char* path, bool needCheck)
{
    if (access(path, W_OK) == 0) {
        return PathStatus::WRITE;
    }
    ASCC_LOGW("Path [%s] : write permission denied!", path);
    if (access(path, R_OK) == 0) {
        return PathStatus::READ;
    }
    ASCC_LOGW("Path [%s] : read permission denied!", path);
    if (access(path, F_OK) == 0) {
        return PathStatus::EXIST;
    }
    if (needCheck) {
        ASCC_LOGE("Path [%s] : no such file or directory!", path);
    } else {
        ASCC_LOGW("Path [%s] : no such file or directory!", path);
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
        ASCC_LOGE("Path [%s] realpath fail!", path);
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

std::string GetCurrentDirectory(void)
{
    char buffer[1024];
    if (getcwd(buffer, sizeof(buffer)) == nullptr) {
        ASCC_LOGE("Get current directory failed!");
        return std::string();
    }
    return std::string(buffer);
}

std::string GenerateTimestamp()
{
    std::time_t now = std::time(nullptr);
    struct tm tmBuf;
    if (!localtime_r(&now, &tmBuf)) {
        ASCC_LOGE("Failed to get local time in GenerateTimestamp!");
    }

    char timestamp[80];    // set up length 80 to store timestamp info
    if (strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &tmBuf) == 0) {
        ASCC_LOGE("Timestamp formatting failed in GenerateTimestamp!");
    }
    return std::string(timestamp);
}

// execute mkdir to create directories
AsccStatus ExecMkdir(struct stat& fileStatus, const char* dirPath)
{
    if (stat(dirPath, &fileStatus) != 0) {
        // need execute permission for directories
        int mkdirRes = mkdir(dirPath, S_IRWXU | S_IRWXG );
        if (mkdirRes != 0 ) {
            if (errno == EEXIST) {
                ASCC_LOGD("Directory already exists: [%s].", dirPath);
                return AsccStatus::SUCCESS;
            } else {
                ASCC_LOGE("Failed to create directory: [%s].", dirPath);
                return AsccStatus::FAILURE;
            }
        }
        ASCC_LOGD("Create directory successfully: [%s].", dirPath);
    }
    return AsccStatus::SUCCESS;
}

// Example dir:  /tmp/bishengcc/tmp_log
AsccStatus CreateDirectory(const std::string& dirPath)
{
    ASCC_CHECK((!dirPath.empty()), {ASCC_LOGE("dirPath is empty in CreateDirectory!");});

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
        ASCC_CHECK((ExecMkdir(fileStatus, currentFullDir.c_str()) == AsccStatus::SUCCESS), {});
        // update currentdir to the rest part, and reset pos
        currentDir = currentDir.substr(pos + 1);
        pos = 0;
    }

    // in Example dir above, deal with dir [tmp_log]
    currentFullDir += currentDir;
    ASCC_CHECK((ExecMkdir(fileStatus, currentFullDir.c_str()) == AsccStatus::SUCCESS), {});
    return AsccStatus::SUCCESS;
}

}