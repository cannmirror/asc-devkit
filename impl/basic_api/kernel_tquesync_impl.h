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
 * \file kernel_tquesync_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_TQUESYNC_IMPL_H
#define ASCENDC_MODULE_TQUESYNC_IMPL_H

namespace AscendC {
template <pipe_t src, pipe_t dst>
__aicore__ inline void TQueSync<src, dst>::SetFlag(TEventID id)
{
    static_assert((src != dst), "src/dst pipe cannot be same.");
    static_assert(IsSupportedPipe(src), "src pipe not supported");
    static_assert(IsSupportedPipe(dst), "dst pipe not supported");
    ASCENDC_ASSERT((id < QUE_MAX_EVENT), {
        KERNEL_LOG(KERNEL_ERROR, "event id input is %d, which should be less than %d", id, QUE_MAX_EVENT);
    });
    set_flag(src, dst, id);
}

template <pipe_t src, pipe_t dst>
__aicore__ inline void TQueSync<src, dst>::WaitFlag(TEventID id)
{
    static_assert((src != dst), "src/dst pipe cannot be same.");
    static_assert(IsSupportedPipe(src), "src pipe not supported");
    static_assert(IsSupportedPipe(dst), "dst pipe not supported");
    ASCENDC_ASSERT((id < QUE_MAX_EVENT), {
        KERNEL_LOG(KERNEL_ERROR, "event id input is %d, which should be less than %d", id, QUE_MAX_EVENT);
    });
    wait_flag(src, dst, id);
}
}
#endif // ASCENDC_MODULE_TQUESYNC_IMPL_H