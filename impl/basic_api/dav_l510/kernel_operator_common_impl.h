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
 * \file kernel_operator_common_impl.h
 * \brief AscendC l510 support common api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#define ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#include "kernel_struct_mm.h"
namespace AscendC {
__aicore__ inline int64_t GetSubBlockIdxImpl()
{
    return 0;
}

__aicore__ inline int64_t GetTaskRationImpl()
{
    return 1;
}

__aicore__ inline int64_t GetBlockIdxImpl()
{
    return block_idx;
}

[[deprecated(
    "NOTICE: SetSysWorkSpace has been deprecated and will be removed in the next version.")]]
__aicore__ inline void SetSysWorkspace(GM_ADDR workspace)
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    ASCENDC_ASSERT((workspace != nullptr), { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
#else
    if (g_sysWorkspaceReserved == nullptr) {
        g_sysWorkspaceReserved = workspace;
    }
#endif
}

__aicore__ inline void SetSysWorkspaceForce(GM_ADDR workspace)
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    ASCENDC_ASSERT((workspace != nullptr), { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
#else
    g_sysWorkspaceReserved = workspace;
#endif
}

__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
{
#if defined(__CCE_KT_TEST__) && __CCE_KT_TEST__ == 1
    ASCENDC_ASSERT((workspace != nullptr), { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
    return workspace;
#else
    (void)(workspace);
    // reserved 0 Bytes
    return g_sysWorkspaceReserved;
#endif
}

template<bool isAIVOnly = true>
__aicore__ inline void SoftSyncAllImpl(__gm__ int32_t* gmWorkspaceAddr, __ubuf__ int32_t* ubWorkspaceAddr,
    const int usedCores)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported software syncAll on this version"); });
}

template<bool isAIVOnly = true>
__aicore__ inline void SyncAllImpl()
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported hardware syncAll on this version"); });
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void NotifyEventImpl(uint16_t flagId)
{
    ASCENDC_ASSERT((false), "CrossCoreSetFlag is not supported on this version");
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void WaitEventImpl(uint16_t flagId)
{
    ASCENDC_ASSERT((false), "CrossCoreWaitFlag is not supported on this version");
}

template <atomic_type_t type, atomic_op_t op>
__aicore__ inline void SetStoreAtomicConfigImpl()
{
    ASCENDC_ASSERT((false), "SetStoreAtomicConfig is not supported on this version");
}

__aicore__ inline int64_t GetStoreAtomicConfigImpl()
{
    ASCENDC_ASSERT((false), "GetStoreAtomicConfigImpl is not supported on this version");
    return 0;
}

__aicore__ inline void GetStoreAtomicConfigImpl(uint16_t &atomicType, uint16_t &atomicOp)
{
    ASCENDC_ASSERT((false), "GetStoreAtomicConfig is not supported on this version");
}

__aicore__ inline void SetSyncBaseAddrImpl(uint64_t config)
{
    ASCENDC_ASSERT((false), "SetSyncBaseAddrImpl is not supported on this version");
}

template <typename T>
__aicore__ inline void DataCachePreloadImpl(const GlobalTensor<uint64_t> &src, const T cacheOffset)
{
    ASCENDC_ASSERT((false), "DataCachePreloadImpl is not supported on this version");
}

__aicore__ inline int64_t GetICachePreloadStatusImpl()
{
    return get_icache_prl_st();
}

__aicore__ inline void PreLoadImpl(void *pc, const int64_t prefetchLen)
{
    ASCENDC_ASSERT((false), "PreLoadImpl is not supported on this version");
}

__aicore__ inline void CheckLocalMemoryIAImpl(const CheckLocalMemoryIAParam& checkParams)
{
    ASCENDC_ASSERT((false), "CheckLocalMemoryIAImpl is not supported on this version");
}

__aicore__ inline void PreLoad(const int64_t preFetchLen)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported software syncAll on this version"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
