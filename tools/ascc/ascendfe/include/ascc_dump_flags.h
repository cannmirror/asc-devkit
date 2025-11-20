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
 * \file ascc_dump_flags.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_DUMP_FLAGS_H__
#define __INCLUDE_ASCC_DUMP_FLAGS_H__

#include "ascc_log.h"
namespace Ascc {
class AsccDumpFlags {
public:
    inline static AsccDumpFlags& GetInstance()
    {
        static AsccDumpFlags instance;
        return instance;
    }
    bool GetIsDumpCloseManual() const
    {
        return isDumpCloseManual;
    }
    bool GetPrintfFlag() const
    {
        return printfFlag;
    }
    int GetPrintfBufSize() const
    {
        return printfBufSize;
    }
    bool GetAssertFlag() const
    {
        return assertFlag;
    }
    int GetAssertBufSize() const
    {
        return assertBufSize;
    }

    void SetIsDumpCloseManual()
    {
        this->isDumpCloseManual = true;
        ASC_LOG_ASC_DEBUG(PREPROCESS, "DumpFlags: isDumpCloseManual is set to true");
    }

    void SetPrintfFlag()
    {
        this->printfFlag = true;
        ASC_LOG_ASC_DEBUG(PREPROCESS, "DumpFlags: printfFlag is set to true");
    }

    void SetAssertFlag()
    {
        this->assertFlag = true;
        ASC_LOG_ASC_DEBUG(PREPROCESS, "DumpFlags: assertFlag is set to true");
    }

    // when user not pass -DASCENDC_DUMP=0, either printf or assert means need dump
    bool IsDumpOn()
    {
        return (!(this->isDumpCloseManual) && (this->printfFlag || this->assertFlag));
    }

    // only when print = false, assert = true => dump size = 1024. Otherwise always 1024 * 1024
    uint32_t GetDumpSize() const
    {
        if (!isDumpCloseManual && printfFlag) {
            return printfBufSize;
        } else if (!isDumpCloseManual && assertFlag) {
            return assertBufSize;
        }
        return printfBufSize;
    }

private:
    bool isDumpCloseManual = false;  // user pass -DASCENDC_DUMP=xx
    bool printfFlag = false;
    bool assertFlag = false;
    int printfBufSize = 1048576;     // 1024 * 1024
    int assertBufSize = 1024;
    AsccDumpFlags() = default;
    ~AsccDumpFlags() = default;
    AsccDumpFlags(const AsccDumpFlags&) = delete;
    AsccDumpFlags& operator=(const AsccDumpFlags&) = delete;
};
}
#endif // Ascc