/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
/**
 * \file kernel_operator_swap_mem_intf.h
 * \brief Interface for memory swap and workspace management
 */
 
#ifndef ASCENDC_MODULE_SWAP_MEM_INTF_H
#define ASCENDC_MODULE_SWAP_MEM_INTF_H

#include "kernel_reg.h"
#include "kernel_process_lock.h"
#include "kernel_operator_tensor_trait.h"

#ifndef WORKSPACE_PARAM_OFFSET
#define WORKSPACE_PARAM_OFFSET 0xffffffff
#endif

__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_sysWorkspaceReserved;

#if defined(ASCENDC_CPU_DEBUG)
__aicore__ __gm__ uint8_t* __gm__ GetSysWorkSpacePtr();
#else
__aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()
{
#if (WORKSPACE_PARAM_OFFSET != 0xffffffff)
    return ((GM_ADDR *)get_para_base())[WORKSPACE_PARAM_OFFSET];
#else
    return g_sysWorkspaceReserved;
#endif
}
#endif

#endif // ASCENDC_KERNEL_SWAP_MEM_INTF_H