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
 * \file ascc_log.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_LOG_H__
#define __INCLUDE_ASCC_LOG_H__
#include <sys/syscall.h>
#include <unistd.h>

#include "ascc_global_env_manager.h"

// Note: In ASCC log, use stage name below
enum class AsccStageType : uint8_t {
    INIT,
    OPTION,
    PREPROCESS,
    UTILS,
    DEVICE_STUB,
    HOST_STUB,
    COMPILE,
    LINK
};

enum class AsccLogLevel : uint8_t {
    ASC_DEBUG,    // 0
    ASC_INFO,     // 1
    ASC_WARN,     // 2
    ASC_ERROR     // 3
};

#define ASCC_CHECK_MAIN(cond, behavior) \
    do {                                \
        if (!(cond)) {                  \
            behavior;                   \
            return Ascc::ASCC_FAILURE;  \
        }                               \
    } while (0)

namespace Ascc {

#define ASC_LOG_ASC_DEBUG(stage, format, ...)                                                                         \
    do {                                                                                                              \
        if (Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout == 1 &&                                               \
            Ascc::AsccGlobalEnvManager::ascendGlobalLogLevel <= static_cast<uint8_t>(AsccLogLevel::ASC_DEBUG)) {      \
            printf("[DEBUG] ASCC [pid:%u, tid:%u] [stage: " #stage "] [%s:%d] " format "\n",                          \
                static_cast<uint32_t>(getpid()),                                                                      \
                static_cast<uint32_t>(syscall(SYS_gettid)),                                                           \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                ##__VA_ARGS__);                                                                                       \
        }                                                                                                             \
    } while (0)

#define ASC_LOG_ASC_INFO(stage, format, ...)                                                                          \
    do {                                                                                                              \
        if (Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout == 1 &&                                               \
            Ascc::AsccGlobalEnvManager::ascendGlobalLogLevel <= static_cast<uint8_t>(AsccLogLevel::ASC_INFO)) {       \
            printf("[INFO]  ASCC [pid:%u, tid:%u] [stage: " #stage "] [%s:%d] " format "\n",                          \
                static_cast<uint32_t>(getpid()),                                                                      \
                static_cast<uint32_t>(syscall(SYS_gettid)),                                                           \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                ##__VA_ARGS__);                                                                                       \
        }                                                                                                             \
    } while (0)

#define ASC_LOG_ASC_WARN(stage, format, ...)                                                                          \
    do {                                                                                                              \
        if (Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout == 1 &&                                               \
            Ascc::AsccGlobalEnvManager::ascendGlobalLogLevel <= static_cast<uint8_t>(AsccLogLevel::ASC_WARN)) {       \
            printf("[WARN]  ASCC [pid:%u, tid:%u] [stage: " #stage "] [%s:%d] " format "\n",                          \
                static_cast<uint32_t>(getpid()),                                                                      \
                static_cast<uint32_t>(syscall(SYS_gettid)),                                                           \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                ##__VA_ARGS__);                                                                                       \
        }                                                                                                             \
    } while (0)

#define ASC_LOG_ASC_ERROR(stage, format, ...)                                                                         \
    do {                                                                                                              \
        if (Ascc::AsccGlobalEnvManager::ascendSlogPrintToStdout == 1 &&                                               \
            Ascc::AsccGlobalEnvManager::ascendGlobalLogLevel <= static_cast<uint8_t>(AsccLogLevel::ASC_ERROR)) {      \
            printf("[ERROR] ASCC [pid:%u, tid:%u] [stage: " #stage "] [%s:%d] " format "\n",                          \
                static_cast<uint32_t>(getpid()),                                                                      \
                static_cast<uint32_t>(syscall(SYS_gettid)),                                                           \
                __FILE__,                                                                                             \
                __LINE__,                                                                                             \
                ##__VA_ARGS__);                                                                                       \
        }                                                                                                             \
    } while (0)

} // Ascc
#endif