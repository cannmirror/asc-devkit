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

namespace AscendC {
__aicore__ inline void GetArchVersionImpl(uint32_t& coreVersion)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on this version");
}

__aicore__ inline int64_t GetSubBlockNumImpl()
{
    return 1;
}

__aicore__ inline int64_t GetProgramCounterImpl()
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on this version");
    return 0;
}

__aicore__ inline int64_t GetSystemCycleImpl()
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on this version");
    return 0;
}

__aicore__ inline void SetPcieRDCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on this version");
}

__aicore__ inline void SetPcieWRCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on this version");
}

__aicore__ inline void TrapImpl()
{
    ASCENDC_ASSERT((false), "SetDeqScale is not supported on this version");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H