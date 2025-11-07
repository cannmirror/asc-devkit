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
 * \file kernel_operator_sys_var_impl.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H

#include <cstdint>
#include "kernel_log.h"
#include "kernel_macros.h"

namespace AscendC {
__aicore__ inline void GetArchVersionImpl(uint32_t& coreVersion)
{
    (void)(coreVersion);
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetArchVersion");
}

__aicore__ inline int64_t GetSubBlockNumImpl()
{
    return 1;
}

__aicore__ inline int64_t GetProgramCounterImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetProgramCounter");
    return 0;
}

__aicore__ inline int64_t GetSystemCycleImpl()
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    const int32_t timeExp = 1000000000;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    int64_t timeStamp = static_cast<int64_t>(ts.tv_sec) * timeExp + ts.tv_nsec;
    return timeStamp;
#else
    uint64_t sysCnt = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(sysCnt));
    return (int64_t)(sysCnt);
#endif
}

__aicore__ inline void SetPcieRDCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    (void)(isSetPcie);
    (void)(maxBurstLen);
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetPcieRDCtrl");
}

__aicore__ inline void SetPcieWRCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    (void)(isSetPcie);
    (void)(maxBurstLen);
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetPcieWRCtrl");
}

__aicore__ inline void TrapImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Trap");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H