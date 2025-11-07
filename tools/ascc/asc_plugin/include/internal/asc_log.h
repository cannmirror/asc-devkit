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
 * \file asc_log.h
 * \brief
 */

#ifndef __INCLUDE_ASC_LOG_H__
#define __INCLUDE_ASC_LOG_H__
#include <cstdio>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>
#include <string>
#include "mmpa/mmpa_api.h"
#include "asc_info_manager.h"

namespace AscPlugin {

enum class LogLevel : uint32_t {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

class LogManager {
public:
    LogManager(const std::string& fullPath) :  logFile_(fopen((fullPath).c_str(), "a")) {
        if (logFile_ == nullptr) {
            std::cerr << "Failed to open log file: " << fullPath << std::endl;
        }
    };
    static const char* GetOutEnv();
    static const char* GetLevelEnv();
    FILE* GetFileLog();
    ~LogManager()
    {
        if (logFile_ != nullptr) {
            fclose(logFile_);
            logFile_ = nullptr;
        }
    }
private:
    FILE* logFile_;
};

void LogToFile(const char* logLevel, const char* file, int line, const char* format, ...)
    __attribute__((format(printf, 4, 5)));
bool AscCheckLogLevel(const LogLevel &logLevel);

#define ASC_LOGD(fmt, ...)                                                                           \
    do {                                                                                             \
        if (AscPlugin::AscCheckLogLevel(AscPlugin::LogLevel::DEBUG)) {                               \
            fprintf(stderr, "[DEBUG] ASCPLUGIN [pid:%u, tid:%u] [%s:%d] " fmt "\n",                  \
                static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)),         \
                __FILE__, __LINE__, ##__VA_ARGS__);                                                  \
            AscPlugin::LogToFile("DEBUG", __FILE__, __LINE__, fmt, ##__VA_ARGS__);                   \
        }                                                                                            \
    } while(0)

#define ASC_LOGI(fmt, ...)                                                                           \
    do {                                                                                             \
        if (AscPlugin::AscCheckLogLevel(AscPlugin::LogLevel::INFO)) {                                \
            fprintf(stderr, "[INFO] ASCPLUGIN [pid:%u, tid:%u] [%s:%d] " fmt "\n",                   \
                static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)),         \
                __FILE__, __LINE__, ##__VA_ARGS__);                                                  \
            AscPlugin::LogToFile("INFO", __FILE__, __LINE__, fmt, ##__VA_ARGS__);                    \
        }                                                                                            \
    } while(0)

#define ASC_LOGW(fmt, ...)                                                                           \
    do {                                                                                             \
        fprintf(stderr, "[WARN] ASCPLUGIN [pid:%u, tid:%u] [%s:%d] " fmt "\n",                       \
            static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)),             \
            __FILE__, __LINE__, ##__VA_ARGS__);                                                      \
        if (AscPlugin::AscCheckLogLevel(AscPlugin::LogLevel::WARN)) {                                \
            AscPlugin::LogToFile("WARN", __FILE__, __LINE__, fmt, ##__VA_ARGS__);                    \
        }                                                                                            \
    } while(0)

#define ASC_LOGE(fmt, ...)                                                                           \
    do {                                                                                             \
        fprintf(stderr, "[ERROR] ASCPLUGIN [pid:%u, tid:%u] [%s:%d] " fmt "\n",                      \
            static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)),             \
            __FILE__, __LINE__, ##__VA_ARGS__);                                                      \
        if (AscPlugin::AscCheckLogLevel(AscPlugin::LogLevel::ERROR)) {                               \
            AscPlugin::LogToFile("ERROR", __FILE__, __LINE__, fmt, ##__VA_ARGS__);                   \
        }                                                                                            \
    } while(0)

#define ASC_CHECK_NULLPTR(inputPtr, funcName)                               \
    do {                                                                    \
        if ((inputPtr) == nullptr) {                                        \
            ASC_LOGE("%s in %s cannot be nullptr.", #inputPtr, funcName);   \
            return AscPlugin::ASC_NULLPTR;                                  \
        }                                                                   \
    } while (0)

#define ASC_CHECK_EMPTY_STR(inputStr, argName, structName)                         \
    do {                                                                           \
        if ((inputStr).empty()) {                                                  \
            ASC_LOGE("%s in struct %s cannot be empty.", argName, structName);     \
            return AscPlugin::ASC_FAILURE;                                         \
        }                                                                          \
    } while (0)

} // namespace AscPlugin
#endif