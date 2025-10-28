/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_common.h
 * \brief
 */
#ifndef IMPL_HCCL_COMMON_H
#define IMPL_HCCL_COMMON_H

namespace AscendC {
__aicore__ inline void FlushDataCache(GlobalTensor<int64_t> &globalHcclMsgArea, __gm__ void *gmAddr)
{
    AscendC::Barrier();
    globalHcclMsgArea.SetGlobalBuffer((__gm__ int64_t *)gmAddr);
    __asm__("NOP");
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(globalHcclMsgArea);
    DataSyncBarrier<MemDsbT::ALL>();
}

__aicore__ inline void FlushDataCache(__gm__ void *gmAddr)
{
    GlobalTensor<int64_t> globalHcclMsgArea;
    FlushDataCache(globalHcclMsgArea, gmAddr);
}

}

#endif