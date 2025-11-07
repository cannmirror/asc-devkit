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

namespace AscendC {
__aicore__ inline void GetArchVersionImpl(uint32_t& coreVersion)
{
    ASCENDC_ASSERT((false), "unsupported GetArchVersion!");
}

__aicore__ inline int64_t GetSubBlockNumImpl()
{
    return 1;
}

__aicore__ inline int64_t GetPhyCoreIDImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetPhyCoreID!");
    return 0;
}

__aicore__ inline int64_t GetDataMainBaseImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetDataMainBase!");
    return 0;
}

__aicore__ inline int64_t GetDataSizeImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetDataSize!");
    return 0;
}

__aicore__ inline int64_t GetDataLocalBaseImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetDataLocalBase!");
    return 0;
}

__aicore__ inline int64_t GetL2VirtualAddressImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetL2VirtualAddress!");
    return 0;
}

__aicore__ inline int64_t GetParameterBaseAddrImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetProgramCounter!");
    return 0;
}

__aicore__ inline int64_t GetProgramCounterImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetParameterBaseAddr!");
    return 0;
}

__aicore__ inline int64_t GetSystemCycleImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetSystemCycle!");
    return 0;
}

__aicore__ inline int64_t GetSystemVirtualBaseImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetSystemVirtualBase!");

    return 0;
}

__aicore__ inline void SetPcieRDCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    ASCENDC_ASSERT((false), "unsupported SetPcieRDCtrl!");
}

__aicore__ inline void SetPcieWRCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    ASCENDC_ASSERT((false), "unsupported SetPcieWRCtrl!");
}

__aicore__ inline void TrapImpl()
{
    trap();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H