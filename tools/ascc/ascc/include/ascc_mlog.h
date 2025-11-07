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
 * \file ascc_mlog.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_MLOG_H__
#define __INCLUDE_ASCC_MLOG_H__
#include <cstdio>
#include <sys/syscall.h>
#include <unistd.h>
#include "ascc_utils.h"

namespace Ascc {
const char * const g_outEnv = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
const char * const g_levelEnv = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");

#define ASCC_LOGD(fmt, ...)                                                                          \
    do {                                                                                             \
        if (Ascc::g_outEnv != nullptr && Ascc::g_outEnv[0] == '1' &&                                 \
            Ascc::g_levelEnv != nullptr && Ascc::g_levelEnv[0] <= '0') {                             \
            printf("[DEBUG] BISHENGCC [pid:%u, tid:%u] [%s:%d] " fmt "\n",                           \
                static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)),         \
                __FILE__, __LINE__, ##__VA_ARGS__);                                                  \
        }                                                                                            \
    } while(0)

#define ASCC_LOGI(fmt, ...)                                                                          \
    do {                                                                                             \
        if (Ascc::g_outEnv != nullptr && Ascc::g_outEnv[0] == '1' &&                                 \
            Ascc::g_levelEnv != nullptr && Ascc::g_levelEnv[0] <= '1') {                             \
            printf("[INFO] BISHENGCC [pid:%u, tid:%u] [%s:%d] " fmt "\n",                            \
                static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)),         \
                __FILE__, __LINE__, ##__VA_ARGS__);                                                  \
        }                                                                                            \
    } while(0)

#define ASCC_LOGW(fmt, ...)                                                                          \
    do {                                                                                             \
        if (Ascc::g_outEnv != nullptr && Ascc::g_outEnv[0] == '1' &&                                 \
            Ascc::g_levelEnv != nullptr && Ascc::g_levelEnv[0] <= '2') {                             \
            printf("[WARN] BISHENGCC [pid:%u, tid:%u] [%s:%d] " fmt "\n",                            \
                static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)),         \
                __FILE__, __LINE__, ##__VA_ARGS__);                                                  \
        }                                                                                            \
    } while(0)

#define ASCC_LOGE(fmt, ...)                                                                          \
    do {                                                                                             \
        if (Ascc::g_outEnv != nullptr && Ascc::g_outEnv[0] == '1' &&                                 \
            Ascc::g_levelEnv != nullptr && Ascc::g_levelEnv[0] <= '3') {                             \
            printf("[ERROR] BISHENGCC [pid:%u, tid:%u] [%s:%d] " fmt "\n",                           \
                static_cast<uint32_t>(getpid()), static_cast<uint32_t>(syscall(SYS_gettid)),         \
                __FILE__, __LINE__, ##__VA_ARGS__);                                                  \
        }                                                                                            \
    } while(0)
}
#endif