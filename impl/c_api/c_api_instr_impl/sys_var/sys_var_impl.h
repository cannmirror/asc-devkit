/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
#ifndef IMPL_C_API_INSTR_SYS_VAR_H
#define IMPL_C_API_INSTR_SYS_VAR_H
 
#include "get_core_id/asc_2201/get_core_id_impl.h"
#include "get_ctrl/asc_2201/get_ctrl_impl.h"
#include "get_imm/asc_2201/get_imm_impl.h"
#include "get_overflow_status/asc_2201/get_overflow_status_impl.h"
#include "get_sub_block_dim/asc_2201/get_sub_block_dim_impl.h"
#include "get_sub_block_id/asc_2201/get_sub_block_id_impl.h"
#include "get_sys_cnt/asc_2201/get_sys_cnt_impl.h"
#include "set_ctrl/asc_2201/set_ctrl_impl.h"
#include "get_block_idx/asc_2201/get_block_idx_impl.h"
#include "get_block_num/asc_2201/get_block_num_impl.h"

__aicore__ inline int64_t asc_get_core_id()
{
    return CApiInternal::get_core_id_impl();
}

__aicore__ inline int64_t asc_get_block_idx()
{
    return CApiInternal::get_block_idx_impl();
}

__aicore__ inline int64_t asc_get_block_num()
{
    return CApiInternal::get_block_num_impl();
}

__aicore__ inline int64_t asc_get_ctrl()
{
    return CApiInternal::get_ctrl_impl();
}

__aicore__ inline uint64_t asc_get_phy_buf_addr(uint64_t offset)
{
    return CApiInternal::get_phy_buf_addr_impl(offset);
}

__aicore__ inline uint64_t asc_get_overflow_status()
{
    return CApiInternal::get_overflow_status_impl();
}

__aicore__ inline int64_t asc_get_sub_block_dim()
{
    return CApiInternal::get_sub_block_dim_impl();
}

__aicore__ inline int64_t asc_get_sub_block_id()
{
    return CApiInternal::get_sub_block_id_impl();
}

__aicore__ inline int64_t asc_get_system_cycle()
{
    return CApiInternal::get_system_cycle_impl();
}

__aicore__ inline void asc_set_ctrl(uint64_t config)
{
    CApiInternal::set_ctrl_impl(config);
}

#endif