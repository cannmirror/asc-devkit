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
 * \file ascc_utils.h
 * \brief
 */


#ifndef __INCLUDE_ASCC_UTILS_H__
#define __INCLUDE_ASCC_UTILS_H__
#include <string>
#include "ascc_common_utils.h"
#include "ascc_info_function.h"
#include "ascc_info_storage.h"

namespace Ascc {

/**
 * @brief Resolves relative paths and returns absolute path
 * @param path Input path (string)
 * @return Canonical absolute path
 */
bool IsPathLegal(const std::string& path);

/**
 * @brief Resolves relative paths and returns absolute path
 * @param path Input path (string)
 * @return Canonical absolute path
 */
bool IsParentDirValid(const std::string& path);

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
 * @brief Converts string to uppercase
 * @param str Input string
 * @return Uppercase converted string
 * @note Supports ASCII
 */
std::string ToUpper(const std::string &str);

/**
 * @brief Converts string to lowercase
 * @param str Input string
 * @return Lowercase converted string
 * @note Supports ASCII
 */
std::string ToLower(const std::string &str);

/**
 * @brief Save compile log
 * @param note Log header/tag
 * @param content Log message body
 * @note Save to <tmpdirectory>/compile_log/compile.log
 */
void SaveCompileLogFile(const std::string &note, const std::string &content);

/**
 * @brief Get storage code info after AST parse
 * @tparam T Metadata type (must inherit from AsccInfoBase)
 * @param fileInfo Compilation file registry container
 * @return shared_ptr to requested file metadata
 *
 * @usage
 * auto funcInfo = GetFileInfo<AsccInfoFunction>(fileInfo);
 */
template <typename T>
std::shared_ptr<T> GetFileInfo(const AsccInfoStorage::FileInfos& fileInfo)
{
    AscCursorTypes cursorType = AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION;
    if constexpr (std::is_same<T, AsccInfoFunction>::value) {
        cursorType = AscCursorTypes::ASC_CURSOR_NORMAL_FUNCTION;
    }
    return fileInfo.find(cursorType) != fileInfo.end() ? std::dynamic_pointer_cast<T>(fileInfo.at(cursorType))
                                                    : nullptr;
}

/**
 * @brief Converts vector<string> to C-style string array.
 *
 * @param[out] charVec Output vector to store C-string pointers.
 * @param[in] strVec Input string vector. Must outlive charVec usage.
 * @return const char** Pointer to C-string array.
 *
 * @note Returned pointers reference strVec's internal buffers.
 * @warning strVec must remain valid while charVec is used.
 */
const char **ConvertStringVecToCStringVec(std::vector<const char*>& charVec, const std::vector<std::string>& strVec);
} // Ascc
#endif // __INCLUDE_ASCC_UTILS_H__