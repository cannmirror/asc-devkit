/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef C_API_SYS_VAR_H
#define C_API_SYS_VAR_H

#include "impl/c_api/instr/sys_var/sys_var.h"

__aicore__ inline void asc_SetCtrl(uint64_t config);

__aicore__ inline int64_t asc_GetCoreId();

__aicore__ inline int64_t asc_GetCtrl();

__aicore__ inline uint64_t asc_GetOverflowStatus();

__aicore__ inline uint64_t asc_GetPhyBufAddr(uint64_t offset);

__aicore__ inline int64_t asc_GetSubBlockDim();

__aicore__ inline int64_t asc_GetSubBlockId();

__aicore__ inline int64_t asc_GetSystemCycle();

#endif