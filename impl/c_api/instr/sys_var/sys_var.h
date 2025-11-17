/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
#ifndef C_API_INSTR_SYS_VAR_H
#define C_API_INSTR_SYS_VAR_H
 
#include "get_core_id/asc_2201/get_core_id_impl.h"
#include "get_ctrl/asc_2201/get_ctrl_impl.h"
#include "get_imm/asc_2201/get_imm_impl.h"
#include "get_overflow_status/asc_2201/get_overflow_status_impl.h"
#include "get_sub_block_dim/asc_2201/get_sub_block_dim_impl.h"
#include "get_sub_block_id/asc_2201/get_sub_block_id_impl.h"
#include "get_sys_cnt/asc_2201/get_sys_cnt_impl.h"
#include "set_ctrl/asc_2201/set_ctrl_impl.h"
#include "get_ffts_base_addr/asc_2201/get_ffts_base_addr_impl.h"
#include "set_ffts_base_addr/asc_2201/set_ffts_base_addr_impl.h"

__aicore__ inline int64_t asc_GetCoreId()
{
    return CApiInternal::asc_GetCoreId();
}

__aicore__ inline int64_t asc_GetCtrl()
{
    return CApiInternal::asc_GetCtrl();
}

__aicore__ inline int64_t asc_GetFftsBaseAddr()
{
    return CApiInternal::asc_GetFftsBaseAddr();
}

__aicore__ inline uint64_t asc_GetPhyBufAddr(uint64_t offset)
{
    return CApiInternal::asc_GetPhyBufAddr(offset);
}

__aicore__ inline uint64_t asc_GetOverflowStatus()
{
    return CApiInternal::asc_GetOverflowStatus();
}

__aicore__ inline int64_t asc_GetSubBlockDim()
{
    return CApiInternal::asc_GetSubBlockDim();
}

__aicore__ inline int64_t asc_GetSubBlockId()
{
    return CApiInternal::asc_GetSubBlockId();
}

__aicore__ inline int64_t asc_GetSystemCycle()
{
    return CApiInternal::asc_GetSystemCycle();
}

__aicore__ inline void asc_SetCtrl(uint64_t config)
{
    CApiInternal::asc_SetCtrl(config);
}

__aicore__ inline void asc_SetFftsBaseAddr(uint64_t config)
{
    CApiInternal::asc_SetFftsBaseAddr(config);
}

#endif