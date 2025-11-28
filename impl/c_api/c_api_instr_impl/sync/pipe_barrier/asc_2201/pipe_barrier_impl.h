/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file pipe_barrier.h
 * \brief
 */
#ifndef IMPL_INSTR_SYNC_PIPE_BARRIER_ASC_2201_PIPE_BARRIER_IMPL_H
#define IMPL_INSTR_SYNC_PIPE_BARRIER_ASC_2201_PIPE_BARRIER_IMPL_H

namespace CApiInternal {

template<pipe_t pipe>
__aicore__ inline void sync_impl()
{
    pipe_barrier(pipe);
}

__aicore__ inline void sync_vec_impl()
{
    pipe_barrier(pipe_t::PIPE_ALL);
}

__aicore__ inline void sync_mte1_impl(int id) { }

__aicore__ inline void sync_mte2_impl(int id)
{
    set_flag(pipe_t::PIPE_MTE2, pipe_t::PIPE_MTE3, id);
    set_flag(pipe_t::PIPE_MTE2, pipe_t::PIPE_V, id);
    wait_flag(pipe_t::PIPE_MTE2, pipe_t::PIPE_MTE3, id);
    wait_flag(pipe_t::PIPE_MTE2, pipe_t::PIPE_V, id);
    pipe_barrier(pipe_t::PIPE_MTE2);
}

__aicore__ inline void sync_mte3_impl(int id)
{
    set_flag(pipe_t::PIPE_MTE3, pipe_t::PIPE_MTE2, id);
    set_flag(pipe_t::PIPE_MTE3, pipe_t::PIPE_V, id);
    wait_flag(pipe_t::PIPE_MTE3, pipe_t::PIPE_MTE2, id);
    wait_flag(pipe_t::PIPE_MTE3, pipe_t::PIPE_V, id);
    pipe_barrier(pipe_t::PIPE_MTE3);
}

__aicore__ inline void sync_matrix_impl(int id) { }

__aicore__ inline void sync_fixpipe_impl(int id) { }

__aicore__ inline void sync_impl()
{
    pipe_barrier(pipe_t::PIPE_ALL);
}

}

#endif