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
 * \file asc_log.cpp
 * \brief
 */

#include "asc_log.h"

namespace AscPlugin {

FILE* AscPlugin::LogManager::GetFileLog() {
    return logFile_;
}

const char* AscPlugin::LogManager::GetOutEnv() {
    static const char* outEnv = [] () {
        const char* env = nullptr;
        MM_SYS_GET_ENV(MM_ENV_ASCEND_SLOG_PRINT_TO_STDOUT, env);
        return env;
    }();
    return outEnv;
}

const char* AscPlugin::LogManager::GetLevelEnv() {
    static const char* levelEnv = [] () {
        const char* env = nullptr;
        MM_SYS_GET_ENV(MM_ENV_ASCEND_GLOBAL_LOG_LEVEL, env);
        return env;
    }();
    return levelEnv;
}

void LogToFile(const char* logLevel, const char* file, int line, const char* format, ...) {
    if (AscPlugin::InfoManager::GetInstance().SaveTempRequested()) {
        static AscPlugin::LogManager log(AscPlugin::InfoManager::GetInstance().GetLogPath() + "/AscPlugin.log");
        FILE* logFile = log.GetFileLog();
        fprintf(logFile,"[%s] ASCPLUGIN [pid:%u, tid:%u] [%s:%d] ", logLevel,
            static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)), file, line);
        va_list args;
        va_start(args, format);
        vfprintf(logFile, format, args);
        va_end(args);
        fputc('\n', logFile);
    }
    return;
}

bool AscCheckLogLevel(const LogLevel &logLevel) {
    const char * const outEnv = AscPlugin::LogManager::GetOutEnv();
    const char * const levelEnv = AscPlugin::LogManager::GetLevelEnv();
    if (logLevel == LogLevel::DEBUG) {
        return (outEnv != nullptr && outEnv[0] == '1' && levelEnv != nullptr && levelEnv[0] <= '0');
    } else if (logLevel == LogLevel::INFO) {
        return (outEnv != nullptr && outEnv[0] == '1' && levelEnv != nullptr && levelEnv[0] <= '1');
    } else if (logLevel == LogLevel::WARN) {
        return (outEnv != nullptr && outEnv[0] == '1' && levelEnv != nullptr && levelEnv[0] <= '2');
    } else if (logLevel == LogLevel::ERROR) {
        return (outEnv != nullptr && outEnv[0] == '1' && levelEnv != nullptr && levelEnv[0] <= '3');
    }
    return false;
}
} // namespace AscPlugin