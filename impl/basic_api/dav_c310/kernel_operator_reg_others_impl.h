/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
/*!
 * \file kernel_operator_reg_others_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_REG_OTHERS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_REG_OTHERS_IMPL_H
 
#ifndef ASCENDC_CPU_DEBUG
namespace AscendC {
template <typename T> __aicore__ inline void SetFlag(pipe_t pipe, pipe_t tpipe, T pipeID)
{
    set_flag(pipe, tpipe, pipeID);
}
 
template <typename T> __aicore__ inline void WaitFlag(pipe_t pipe, pipe_t tpipe, T pipeID)
{
    wait_flag(pipe, tpipe, pipeID);
}
}
#endif
#endif // ASCENDC_MODULE_OPERATOR_REG_OTHERS_IMPL_H