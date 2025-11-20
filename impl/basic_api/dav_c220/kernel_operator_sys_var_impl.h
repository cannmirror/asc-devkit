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
 * \file kernel_operator_sys_var_impl.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H
#include "kernel_utils.h"
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <ctime>
#endif // ASCENDC_CPU_DEBUG

namespace AscendC {

__aicore__ inline int64_t GetSubBlockNumImpl()
{
    if ASCEND_IS_AIC {
        return 1;
    } else {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        return g_taskRation;
#else
        return get_subblockdim();
#endif
    }
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

__aicore__ inline void GetArchVersionImpl(uint32_t& coreVersion)
{
    const int32_t coreVersionOffset = 32;
    coreVersion = static_cast<uint32_t>((static_cast<uint64_t>(get_arch_ver()) >> coreVersionOffset) & 0xFFF);
}

__aicore__ inline int64_t GetProgramCounterImpl()
{
    int64_t pc = static_cast<int64_t>(static_cast<uint64_t>(get_pc()) & 0xFFFFFFFFFFFF);
    return pc;
}

__aicore__ inline void TrapImpl()
{
    trap();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H