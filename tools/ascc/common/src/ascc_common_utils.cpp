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
 * \file ascc_common_utils.cpp
 * \brief
 */
#include <climits>
#include <sstream>
#include <string.h>
#include <mutex>
#include "ascc_common_utils.h"

namespace Ascc {
std::mutex g_cout_mutex;    // due to multi thread in bishengcc, must use lock to make thread safe for std::xx

void HandleError(const std::string& message)
{
    std::lock_guard<std::mutex> lock(g_cout_mutex);
    std::cerr << "[ERROR] " << message << std::endl;
}

void HandleErrorAndCheckLog(const std::string& message)
{
    std::cerr << "[ERROR] " << message << " Please check log." << std::endl;
}

std::pair<std::string, int> ExecuteCommand(const char *command)
{
    std::ostringstream oss;
    FILE *pipe = popen(command, "r");
    if (!pipe) {
        HandleError(std::string("popen() failed: ") + std::string(strerror(errno)));
        return {"", -1};
    }

    char buffer[2048];    //  set up as 2048 to store
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        oss << buffer;
    }

    int status = pclose(pipe);
    if (status == -1) {
        HandleError(std::string("pclose() failed: ") + std::string(strerror(errno)));
        return {"", -1};
    }

    unsigned uStatus = static_cast<unsigned>(status);
    if (WIFEXITED(uStatus)) {
        return {oss.str(), WEXITSTATUS(uStatus)};
    } else if (WIFSIGNALED(uStatus)) {
        HandleError(std::string("Command terminated by signal: ") + std::to_string(WTERMSIG(uStatus)));
        return {"", -1};
    } else {
        HandleError("Command did not terminate normally.");
        return {"", -1};
    }
}

std::string RemoveSuffix(const std::string& fileName)
{
    size_t dotPos = fileName.find_last_of('.');
    if (dotPos != std::string::npos && dotPos > 0) {
        return fileName.substr(0, dotPos);
    }
    return fileName;
}

std::string GetFileName(const std::string &filePath)
{
    size_t pos = filePath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filePath.substr(pos + 1);
    }
    return filePath;
}

std::string GetSuffix(const std::string &fileName)
{
    size_t lastdot = fileName.find_last_of(".");
    if (lastdot != std::string::npos) {
        return fileName.substr(lastdot, fileName.size());  // add.cpp -> .cpp
    }
    return "";
}

}