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
 * \file ascc_common_utils.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_COMMON_UTILS_H__
#define __INCLUDE_ASCC_COMMON_UTILS_H__

#include <string>
#include <iostream>

#define ASCC_CHECK(cond, behavior)             \
    do {                                       \
        if (!(cond)) {                         \
            behavior;                          \
            return Ascc::AsccStatus::FAILURE;  \
        }                                      \
    } while (0)

namespace Ascc {
/**
 * @brief Handles fatal errors
 * @param message Error description
 * @note Writes to stderr
 */
void HandleError(const std::string& message);
void HandleErrorAndCheckLog(const std::string& message);    // with please check log in the end

/**
 * @brief Executes system command and captures output
 * @param command Shell command to execute
 * @return Pair containing <command_output, exit_status>
 * @warning Blocks until command completes
 */
std::pair<std::string, int> ExecuteCommand(const char *command);

/**
 * @brief Removes file extension from filename
 * @param fileName Input name (e.g., "image.jpg")
 * @return Name without last extension (e.g., "image")
 * @note Returns original name if no extension exists
 */
std::string RemoveSuffix(const std::string& fileName);

/**
 * @brief Extracts filename with extension from full path
 * @param filePath Input path (e.g., "/home/user/doc.txt")
 * @return Filename component (e.g., "doc.txt")
 * @note Handles both '/' and '\\' separators
 */
std::string GetFileName(const std::string& filePath);

/**
 * @brief Get file extension from filename
 * @param fileName Input name (e.g., "add_custom.cpp")
 * @return file extension (e.g., ".cpp")
 * @note Returns "" if no suffix exists
 */
std::string GetSuffix(const std::string& fileName);

} // namespace Ascc

#endif // __INCLUDE_ASCC_COMMON_UTILS_H__