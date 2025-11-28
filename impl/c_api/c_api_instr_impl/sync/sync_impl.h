
/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
#ifndef IMPL_C_API_INSTR_SYNC_H
#define IMPL_C_API_INSTR_SYNC_H

#include "c_api_interf_util.h"
#include "set_flag/asc_2201/set_flag_impl.h"
#include "wait_flag/asc_2201/wait_flag_impl.h"
#include "pipe_barrier/asc_2201/pipe_barrier_impl.h"

template<typename Pipe, typename TPipe>
__aicore__ inline void asc_sync_notify(Pipe pipe, TPipe tpipe, int id)
{
    CApiInternal::sync_notify_impl<Pipe::value, TPipe::value>(id);
}

template<typename Pipe, typename TPipe>
__aicore__ inline void asc_sync_wait(Pipe pipe, TPipe tpipe, int id)
{
    CApiInternal::sync_wait_impl<Pipe::value, TPipe::value>(id);
}

template<typename Pipe>
__aicore__ inline void asc_sync(Pipe pipe)
{
    CApiInternal::sync_impl<Pipe::value>();
}

__aicore__ inline void asc_sync_vec()
{
    CApiInternal::sync_vec_impl();
}

__aicore__ inline void asc_sync_mte3(int id)
{
    CApiInternal::sync_mte3_impl(id);
}

__aicore__ inline void asc_sync_mte2(int id)
{
    CApiInternal::sync_mte2_impl(id);
}

__aicore__ inline void asc_sync_mte1(int id)
{
    CApiInternal::sync_mte1_impl(id);
}

__aicore__ inline void asc_sync_matrix(int id)
{
    CApiInternal::sync_matrix_impl(id);
}

__aicore__ inline void asc_sync_fixpipe(int id)
{
    CApiInternal::sync_fixpipe_impl(id);
}

__aicore__ inline void asc_sync()
{
    CApiInternal::sync_impl();
}

#endif