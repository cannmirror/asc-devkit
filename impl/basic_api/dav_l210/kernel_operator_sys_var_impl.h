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
 * \brief AscendC l210 support sys var api.
 */

#ifndef ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H

namespace AscendC {
__aicore__ inline void GetArchVersionImpl(uint32_t& coreVersion)
{
    const int32_t coreVersionOffset = 32;
    coreVersion = (uint32_t)((uint64_t)(get_arch_ver() >> coreVersionOffset) & 0xFFF);
}

__aicore__ inline int64_t GetSubBlockNumImpl()
{
    return 1;
}

__aicore__ inline int64_t GetPhyCoreIDImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline int64_t GetDataMainBaseImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline int64_t GetDataSizeImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline int64_t GetDataLocalBaseImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline int64_t GetL2VirtualAddressImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline int64_t GetParameterBaseAddrImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline int64_t GetProgramCounterImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline int64_t GetSystemCycleImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline int64_t GetSystemVirtualBaseImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
    return 0;
}

__aicore__ inline void SetPcieRDCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
}

__aicore__ inline void SetPcieWRCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
}
__aicore__ inline void TrapImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "NotifyEvent is not supported on this device!"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H