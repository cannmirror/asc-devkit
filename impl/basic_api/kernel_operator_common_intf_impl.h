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
 * \file kernel_operator_common_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_IMPL_H
#include "kernel_tensor.h"
#include "kernel_struct_mm.h"

/*
 * ingroup：SetAtomicAdd
 * brief：Set the next data from UB to the outside of AI Core whether the move write Tensor operation performs
 * atomic accumulation.
 */
#if __NPU_ARCH__ == 1001
#include "dav_c100/kernel_operator_set_atomic_impl.h"
#include "dav_c100/kernel_operator_common_impl.h"
#include "dav_c100/kernel_operator_vec_duplicate_impl.h"
#include "dav_c100/kernel_operator_sync_impl.h"
#elif __NPU_ARCH__ == 2002
#include "dav_m200/kernel_operator_set_atomic_impl.h"
#include "dav_m200/kernel_operator_common_impl.h"
#include "dav_m200/kernel_operator_vec_duplicate_impl.h"
#include "dav_m200/kernel_operator_sync_impl.h"
#elif __NPU_ARCH__ == 2201
#include "dav_c220/kernel_operator_set_atomic_impl.h"
#include "dav_c220/kernel_operator_common_impl.h"
#include "dav_c220/kernel_operator_sync_impl.h"
#include "dav_c220/kernel_operator_vec_duplicate_impl.h"
#include "dav_c220/kfc/kfc_comm_client.h"
#include "dav_c220/kfc/kfc_comm_server.h"
#include "dav_c220/core_mng/roc/kernel_operator_cube_group_handle_impl.h"
#include "dav_c220/core_mng/roc/kernel_operator_group_barrier_impl.h"
#elif __NPU_ARCH__ == 3002
#include "dav_m300/kernel_operator_set_atomic_impl.h"
#elif __NPU_ARCH__ == 3102
#include "dav_m310/kernel_operator_set_atomic_impl.h"
#elif __NPU_ARCH__ == 3101
#include "dav_c310/kernel_operator_set_atomic_impl.h"
#include "dav_c310/kernel_operator_common_impl.h"
#include "dav_c310/kernel_operator_sync_impl.h"
#if KFC_C310_SSBUF == 1
#include "dav_c310/kfc/kfc_comm_client.h"
#include "dav_c310/kfc/kfc_comm_server.h"
#else
#include "dav_c310/kfc/kfc_comm_client_gm.h"
#include "dav_c310/kfc/kfc_comm_server_gm.h"
#endif
#if __NPU_ARCH__ == 3101
#include "dav_c310/core_mng/roc/kernel_operator_cube_group_handle_impl.h"
#include "dav_c310/core_mng/roc/kernel_operator_group_barrier_impl.h"
#endif
#elif (__NPU_ARCH__ == 5102)
#include "dav_m510/kernel_operator_set_atomic_impl.h"
#include "dav_m510/kernel_operator_common_impl.h"
#include "dav_m510/kernel_operator_sync_impl.h"
#elif (__NPU_ARCH__ == 3113)
#include "dav_l311/kernel_operator_set_atomic_impl.h"
#endif
#include "kernel_pop_stack_buffer.h"

namespace AscendC {
/*
 * @ingroup：IBSet, IBWait
 * @brief：Set the flag bit of a core
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 * @param [in] blockIdx the idx number waiting for the core
 * @param [in] eventID Set and wait events
 */
__aicore__ inline int64_t GetBlockNum();
template <bool isAIVOnly>
__aicore__ inline void IBSet(const GlobalTensor<int32_t> &gmWorkspace,
    const LocalTensor<int32_t> &ubWorkspace, int32_t blockIdx, int32_t eventID)
{
    int32_t blockNum = GetBlockNum();
#if (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
#if (__NPU_ARCH__ != 5102)
    if ASCEND_IS_AIC {
        return;
    }
#endif
    if (!isAIVOnly) {
        blockNum = GetBlockNum() * 2;
    }
#endif
#if __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002
    __ib_set_stub(blockIdx, eventID, isAIVOnly);
#endif
    auto localSyncGM = gmWorkspace[blockNum * 8 * eventID + blockIdx * 8];
    pipe_barrier(PIPE_ALL);

    while (true) {
        DataCopy(ubWorkspace, localSyncGM, ONE_BLK_SIZE / sizeof(int32_t));
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        if (ubWorkspace.GetValue(0) == 0) {
            ubWorkspace.SetValue(0, 1);
            event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            DataCopy(localSyncGM, ubWorkspace, ONE_BLK_SIZE / sizeof(int32_t));
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
#if __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002
    __ib_set_stub(blockIdx, eventID, isAIVOnly);
#endif
}

template <bool isAIVOnly>
__aicore__ inline void IBWait(const GlobalTensor<int32_t> &gmWorkspace,
    const LocalTensor<int32_t> &ubWorkspace, int32_t blockIdx, int32_t eventID)
{
    int32_t blockNum = GetBlockNum();
#if (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
#if (__NPU_ARCH__ != 5102)
    if ASCEND_IS_AIC {
        return;
    }
#endif
    if (!isAIVOnly) {
        blockNum = GetBlockNum() * 2;
    }
#endif
#if __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002
    __ib_wait_stub(blockIdx, eventID, isAIVOnly);
#endif
    auto localSyncGM = gmWorkspace[blockNum * 8 * eventID + blockIdx * 8];
    pipe_barrier(PIPE_ALL);

    while (true) {
        DataCopy(ubWorkspace, localSyncGM, ONE_BLK_SIZE / sizeof(int32_t));
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        if (ubWorkspace.GetValue(0) == 1) {
            ubWorkspace.SetValue(0, 0);
            event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            DataCopy(localSyncGM, ubWorkspace, ONE_BLK_SIZE / sizeof(int32_t));
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
#if __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002
    __ib_wait_stub(blockIdx, eventID, isAIVOnly);
#endif
}

/*
 * @ingroup：SetNextTaskStart, WaitPreTaskEnd
 * @brief：In SuperKernel fusion mode, set wait flag between two operators
 */
template<pipe_t AIV_PIPE, pipe_t AIC_PIPE>
__aicore__ inline void SetNextTaskStart()
{
#ifdef __ASCENDC_ENABLE_SET_NEXT_TASK_START
    SetNextTaskStartImpl<AIV_PIPE, AIC_PIPE>();
#endif
}

__aicore__ inline void WaitPreTaskEnd()
{
#ifdef __ASCENDC_ENABLE_WAIT_PRE_TASK_END
    WaitPreTaskEndImpl();
#endif
}

/*
 * @ingroup：SyncALL
 * @brief：Set flag bits of all cores
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 */
template <bool isAIVOnly>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t> &gmWorkspace,
    const LocalTensor<int32_t> &ubWorkspace, const int usedCores)
{
#if ASCENDC_CPU_DEBUG
    SoftSyncAllImpl<false>((__gm__ int32_t*)gmWorkspace.GetPhyAddr(),
        (__ubuf__ int32_t*)ubWorkspace.GetPhyAddr(), usedCores);
#else
    SoftSyncAllImpl<isAIVOnly>((__gm__ int32_t*)gmWorkspace.GetPhyAddr(),
        (__ubuf__ int32_t*)ubWorkspace.GetPhyAddr(), usedCores);
#endif
}

__aicore__ inline void InitSocState()
{
    AscendCUtils::InitSocStateImpl();
}

__aicore__ inline int64_t GetBlockIdx()
{
    return GetBlockIdxImpl();
}

__aicore__ inline int64_t GetBlockNum()
{
#ifdef __SUPER_KERNEL_STATIC_BLOCK_NUM__
    return __SUPER_KERNEL_STATIC_BLOCK_NUM__;
#elif defined(__SUPER_KERNEL_DYNAMIC_BLOCK_NUM__)
    return g_super_kernel_dynamic_block_num;
#else
    return get_block_num();
#endif
}

__aicore__ inline int64_t GetSubBlockIdx()
{
    return GetSubBlockIdxImpl();
}

__aicore__ inline int64_t GetTaskRatio()
{
    return GetTaskRationImpl();
}

__aicore__ inline int64_t GetTaskRation()
{
    return GetTaskRatio();
}

template <typename T>
__aicore__ inline __in_pipe__(V)
    __out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value)
{
#if (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
#if (__NPU_ARCH__ != 5102)
    if ASCEND_IS_AIC {
        return;
    }
#endif
    LocalTensor<T> popBuffer;
    bool ret = PopStackBuffer<T, TPosition::LCM>(popBuffer);
    uint32_t maxBurstSize = (MAX_REPEAT_TIMES * ONE_BLK_SIZE) / sizeof(T);
    uint32_t popSize = popBuffer.GetSize() >= maxBurstSize ? maxBurstSize : popBuffer.GetSize();
    uint32_t round = size / popSize;
    uint32_t tail = size % popSize;
    uint32_t roundSize = round != 0 ? popSize : 0;
    DuplicateImpl<T>((__ubuf__ T*)popBuffer.GetPhyAddr(), value, popSize);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    struct DataCopyExtParams repeatParams;
    uint32_t comOffset = 0;
    // compute the main block
    repeatParams = { 1, static_cast<uint32_t>(roundSize * sizeof(T)), 0, 0, 0 };
    for (int index = 0; index < round; ++index) {
        DataCopyPadUB2GMImpl((__gm__ T*)gmWorkspaceAddr.GetPhyAddr() + comOffset,
            (__ubuf__ T*)popBuffer.GetPhyAddr(),
            repeatParams);
        comOffset += roundSize;
    }
    // compute the tail block
    repeatParams = {1, static_cast<uint32_t>(tail * sizeof(T)), 0, 0, 0};
    if (tail != 0) {
        comOffset = round * roundSize;
        DataCopyPadUB2GMImpl((__gm__ T*)gmWorkspaceAddr.GetPhyAddr() + comOffset,
            (__ubuf__ T*)popBuffer.GetPhyAddr(),
            repeatParams);
    }
#endif
}

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
template<bool isAIVOnly, const SyncAllConfig& config>
__aicore__ inline void SyncAll()
{
    SyncAllImpl<isAIVOnly, config.triggerPipe, config.waitPipe>();
}
#else
template<bool isAIVOnly>
__aicore__ inline void SyncAll()
{
    SyncAllImpl<isAIVOnly>();
}
#endif

template <AtomicDtype type, AtomicOp op>
__aicore__ inline void SetStoreAtomicConfig()
{
    SetStoreAtomicConfigImpl<static_cast<atomic_type_t>(type), static_cast<atomic_op_t>(op)>();
}

__aicore__ inline int64_t GetStoreAtomicConfig()
{
    return GetStoreAtomicConfigImpl();
}

__aicore__ inline void GetStoreAtomicConfig(uint16_t &atomicType, uint16_t &atomicOp)
{
    GetStoreAtomicConfigImpl(atomicType, atomicOp);
}

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
template <pipe_t pipe, uint8_t subBlockSyncMode = 2>
__aicore__ inline void NotifyEvent(uint16_t flagId)
{
    NotifyEventImpl<subBlockSyncMode, pipe>(flagId);
}

template <pipe_t pipe=PIPE_S, uint8_t mode = 0>
__aicore__ inline void WaitEvent(uint16_t flagId)
{
    WaitEventImpl<mode, pipe>(flagId);
}
#else
template <pipe_t pipe>
__aicore__ inline void NotifyEvent(uint16_t flagId)
{
    constexpr uint8_t subBlockSyncMode = 0x02;
    NotifyEventImpl<subBlockSyncMode, pipe>(flagId);
}

template <pipe_t pipe=PIPE_S>
__aicore__ inline void WaitEvent(uint16_t flagId)
{
    constexpr uint8_t mode = 0;
    WaitEventImpl<mode, pipe>(flagId);
}
#endif

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
template <int count>
__aicore__ inline void Nop()
{
    if (count <= 0) {
        return;
    }
    PipeBarrier<PIPE_ALL>();
    #pragma unroll
    for (int i = 0; i < count; i++) {
        asm volatile("nop");
    }
    PipeBarrier<PIPE_ALL>();
}
#endif

template<uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
{
    NotifyEventImpl<modeId, pipe>(flagId);    
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
{
    WaitEventImpl<modeId, pipe>(flagId);
}

template <typename T>
__aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t> &src, const T cacheOffset)
{
#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
    DataCachePreloadImpl((__gm__ uint64_t*)src.GetPhyAddr(), cacheOffset);
#else
    DataCachePreloadImpl(src, cacheOffset);
#endif
}

__aicore__ inline void ICachePreLoad(const int64_t preFetchLen)
{
    PreLoad(preFetchLen);
}

__aicore__ inline int64_t GetICachePreloadStatus()
{
    return GetICachePreloadStatusImpl();
}

__aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams)
{
    CheckLocalMemoryIAImpl(checkParams);
}

#if (__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
template <HardEvent event, MemoryT memT, bool isVirtual> __aicore__ inline void HSetFlag(int32_t eventID)
{
    if (g_coreType == AIV) {
        return;
    }
    HSetFlagImpl<event, memT, isVirtual>(eventID);
}

template <HardEvent event, MemoryT memT, bool isVirtual> __aicore__ inline void HWaitFlag(int32_t eventID)
{
    if (g_coreType == AIV) {
        return;
    }
    HWaitFlagImpl<event, memT, isVirtual>(eventID);
}
#endif

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)
template <SpecialPurposeReg spr>
__aicore__ inline int64_t GetSpr(){
 
    return GetSprImpl<spr>();
}
 
template <SpecialPurposeReg spr>
__aicore__ inline void ClearSpr(){
    ClearSprImpl<spr>();
}

template <int8_t startBit, int8_t endBit>
__aicore__ static inline void SetCtrlSpr(int64_t value){
    SetCtrlSprImpl<startBit, endBit>(value);
}

template <int8_t startBit, int8_t endBit>
__aicore__ static inline int64_t GetCtrlSpr(){
    return GetCtrlSprImpl<startBit, endBit>();
}

template <int8_t startBit, int8_t endBit>
__aicore__ static inline void ResetCtrlSpr(){
    ResetCtrlSprImpl<startBit, endBit>();
}
#endif
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_IMPL_H
