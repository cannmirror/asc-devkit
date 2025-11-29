/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
#ifndef IMPL_C_API_INSTR_MISC_H
#define IMPL_C_API_INSTR_MISC_H

#include "init_soc_state_impl.h"
#include "get_block_idx_impl.h" 
#include "get_block_num_impl.h"

__aicore__ inline void asc_init()
{
    CApiInternal::init_soc_state_impl();
}

__aicore__ inline int64_t asc_get_block_idx()
{
    return CApiInternal::get_block_idx_impl();
}

__aicore__ inline int64_t asc_get_block_num()
{
    return CApiInternal::get_block_num_impl();
}

#endif