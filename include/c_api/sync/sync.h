/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INCLUDE_C_API_SYNC_H
#define INCLUDE_C_API_SYNC_H

#include "c_api_instr_impl/sync/sync_impl.h"
#include "c_api_interf_util.h"

template<typename Pipe, typename TPipe>
__aicore__ inline void asc_sync_notify(Pipe pipe, TPipe tpipe, int id);

template<typename Pipe, typename TPipe>
__aicore__ inline void asc_sync_wait(Pipe pipe, TPipe tpipe, int id);

template<typename Pipe>
__aicore__ inline void asc_sync(Pipe pipe);

__aicore__ inline void asc_sync_vec();

__aicore__ inline void asc_sync_mte3(int id);

__aicore__ inline void asc_sync_mte2(int id);

__aicore__ inline void asc_sync_mte1(int id);

__aicore__ inline void asc_sync_matrix(int id);

__aicore__ inline void asc_sync_fixpipe(int id);

__aicore__ inline void asc_sync();

#endif