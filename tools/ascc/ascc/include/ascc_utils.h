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
 * \file ascc_utils.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_UTILS_H__
#define __INCLUDE_ASCC_UTILS_H__

#include <string>
#include <sys/stat.h>
#include "ascc_common_types.h"
#include "ascc_common_utils.h"

namespace Ascc {

/**
 * @brief Validates path and accessibility
 * @param path Null-terminated C-style path string
 * @param needCheck true means will show error message, false means only check realpath, no error message
 * @return PathStatus enum (EXIST/READ/WRITE/NOT_EXIST)
 */
PathStatus PathCheck(const char* path, bool needCheck);

/**
 * @brief Resolves relative paths and returns absolute path
 * @param path Input path
 * @return Canonical absolute path
 */
std::string CheckAndGetFullPath(const char* path);
std::string CheckAndGetFullPath(const std::string& path);

/**
 * @brief Extracts directory path from full path
 * @param filePath Input path (e.g., "/home/user/doc.txt")
 * @return Directory component (e.g., "/home/user/")
 */
std::string GetFilePath(const std::string& filePath);

/**
 * @brief Retrieves current working directory
 * @return Platform-independent path string
 * @note Uses getcwd internally
 */
std::string GetCurrentDirectory(void);

/**
 * @brief Generate timestamp
 * @return string that represents timestamp
 */
std::string GenerateTimestamp();

/**
 * @brief Create directory by mkdir(c++)
 * @param fileStatus status of file
 * @param dirPath directory path (e.g., "/tmp/bishengcc")
 * @return string that represents timestamp
 */
AsccStatus ExecMkdir(struct stat& fileStatus, const char* dirPath);

/**
 * @brief Create directory recursively
 * @param dirPath directory path (e.g., "/tmp/bishengcc")
 * @return return SUCCESS if create directory successfully
 */
AsccStatus CreateDirectory(const std::string& dirPath);
}

#endif // __INCLUDE_ASCC_UTILS_H__