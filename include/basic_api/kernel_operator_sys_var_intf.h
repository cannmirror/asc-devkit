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
 * \file kernel_operator_sys_var_intf.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_SYS_VAR_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_SYS_VAR_INTERFACE_H
#include "kernel_struct_mm.h"

namespace AscendC {

__aicore__ inline int64_t GetBlockNum();

__aicore__ inline int64_t GetBlockIdx();

__aicore__ inline int64_t GetSubBlockIdx();

__aicore__ inline int64_t GetTaskRatio();

__aicore__ inline constexpr int16_t GetDataBlockSizeInBytes()
{
    return ONE_BLK_SIZE;
}

__aicore__ inline void GetArchVersion(uint32_t& coreVersion);

__aicore__ inline int64_t GetSubBlockNum();

__aicore__ inline int64_t GetProgramCounter();

__aicore__ inline void Trap();

__aicore__ inline int64_t GetSystemCycle();

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
template <SpecialPurposeReg spr>
__aicore__ inline int64_t GetSpr();

template <SpecialPurposeReg spr>
__aicore__ inline void ClearSpr();
#endif

#if defined(__NPU_ARCH__) &&                                                                                    \
    ((__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 2103) || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) ||    \
     (__NPU_ARCH__ == 3113) || (__NPU_ARCH__ == 3101))
__aicore__ inline constexpr uint32_t GetUBSizeInBytes()
{
    return TOTAL_UB_SIZE;
}

__aicore__ inline constexpr uint32_t GetVecLen()
{
    return VECTOR_REG_WIDTH;
}
#endif
} // namespace AscendC
#include "../../impl/basic_api/kernel_operator_sys_var_intf_impl.h"
#endif // ASCENDC_MODULE_OPERATOR_SYS_VAR_INTERFACE_H