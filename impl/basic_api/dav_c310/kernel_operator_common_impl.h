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

#ifndef ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#define ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#include "kernel_common.h"
#include "kernel_utils.h"
#include "kernel_struct_mm.h"
namespace AscendC {

__aicore__ inline int64_t GetSubBlockIdxImpl()
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if ASCEND_IS_AIV {
        return sub_block_idx;
    }
    return 0;
#else
    return get_subblockid();
#endif
}

__aicore__ inline int64_t GetTaskRationImpl()
{
    if ASCEND_IS_AIC {
        return 1;
    } else {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        return g_taskRation;
#else
        return get_subblockdim();
#endif
    }
}

__aicore__ inline int64_t TscmGetTaskRation()
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        return g_taskRation;
#else
        return get_subblockdim();
#endif
}

__aicore__ inline int64_t GetBlockIdxImpl()
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if ASCEND_IS_AIV {
        return block_idx * g_taskRation + sub_block_idx;
    }
    return block_idx;
#else
    if ASCEND_IS_AIV {
        return get_block_idx() * get_subblockdim() + get_subblockid();
    } else {
        return get_block_idx();
    }
#endif
}


__aicore__ inline void SetSysWorkspace(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((workspace != nullptr),
        { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
#else
    if (g_sysWorkspaceReserved == nullptr) {
        g_sysWorkspaceReserved = workspace;
    }
#endif
}

__aicore__ inline void SetSysWorkspaceForce(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((workspace != nullptr),
        { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
#else
#if (WORKSPACE_PARAM_OFFSET == 0xffffffff)
    g_sysWorkspaceReserved = workspace;
#endif
#endif
}

__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
{
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ASCENDC_ASSERT((workspace != nullptr),
        { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
    return workspace;
#else
    (void)(workspace);
    return GetSysWorkSpacePtr() + RESERVED_WORKSPACE;
#endif
}

__aicore__ inline int64_t GetStoreAtomicConfigImpl()
{
    return get_st_atomic_cfg();
}

__aicore__ inline void GetStoreAtomicConfigImpl(uint16_t &atomicType, uint16_t &atomicOp)
{
    int64_t stAtomic = get_st_atomic_cfg();
    constexpr uint64_t typeMask = 0x7;
    constexpr uint64_t opBit = 4;
    constexpr uint64_t opMask = 0x3;
    atomicType = (static_cast<uint64_t>(stAtomic) & typeMask);
    atomicOp = ((static_cast<uint64_t>(stAtomic) >> opBit) & opMask);
}

template <typename T>
__aicore__ inline void DataCachePreloadImpl(__gm__ uint64_t *src, const T cacheOffset)
{
    static_assert(SupportType<T, int16_t, int64_t>(),
        "Failed to check dtype in DataCachePreload, current api support dtype is int16_t / int64_t");
    dc_preload(src, cacheOffset);
}

__aicore__ inline void PreLoadImpl(void *pc, const int64_t preFetchLen)
{
    preload(pc, preFetchLen);
}

__aicore__ inline int64_t GetICachePreloadStatusImpl()
{
    return get_icache_prl_st();
}

__aicore__ inline void PreLoad(const int64_t preFetchLen)
{
    int64_t pc = get_pc() & 0xFFFFFFFFFFFF;
    PreLoadImpl(reinterpret_cast<void *>(pc), preFetchLen);
}

__aicore__ inline void CheckLocalMemoryIAImpl(const CheckLocalMemoryIAParam& checkParams)
{
    (void)(checkParams);
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CheckLocalMemoryIA is not supported on current device"); });
}

template <atomic_type_t type, atomic_op_t op>
__aicore__ inline void SetStoreAtomicConfigImpl()
{
    set_st_atomic_cfg(type, op);
}

template <SpecialPurposeReg spr>
__aicore__ inline int64_t GetSprImpl()
{
    static_assert(SupportEnum<spr, SpecialPurposeReg::AR>(),
        "current GetSpr api only support SpecialPurposeReg AR on current device!");
    return get_ar();
}
 
__simd_vf__ inline void ClearARImpl()
{
    constexpr uint8_t SPR_AR_VALUE = 74;
    constexpr auto sprValue = std::integral_constant<::Spr, static_cast<::Spr>(SPR_AR_VALUE)>();
    sprclr(sprValue);
}
 
template <SpecialPurposeReg spr>
__aicore__ inline void ClearSprImpl()
{
    static_assert(SupportEnum<spr, SpecialPurposeReg::AR>(),
        "current ClearSpr api only support SpecialPurposeReg AR on current device!");
    
    if constexpr (spr == SpecialPurposeReg::AR) {
        ClearARImpl();
    }
}

template <int8_t startBit, int8_t endBit>
__aicore__ static inline void SetCtrlSprImpl(int64_t value)
{
    static_assert((startBit <= endBit && startBit >= 0 && endBit < 64), "Invalid bit range on current device!");
    static_assert((6 <= startBit && startBit <= 10 && 6 <= endBit && endBit <= 10) ||
                      (startBit == endBit && (startBit == 45 || startBit == 48 || startBit == 50 || startBit == 53 ||
                                              startBit == 59 || startBit == 60)),
                  "Invalid startBit/endBit on current device!");
    if (endBit - startBit == 63) {
        set_ctrl(value);
        return;
    }
    uint64_t mask = ((uint64_t(1) << (endBit - startBit + 1)) - 1) << startBit;
    mask = ~mask;
    int64_t setValue = get_ctrl() & mask;
    setValue |= (value << startBit);
    set_ctrl(setValue);
}

template <int8_t startBit, int8_t endBit>
__aicore__ static inline int64_t GetCtrlSprImpl()
{
    static_assert((startBit <= endBit && startBit >= 0 && endBit < 64), "Invalid bit range on current device!");
    int64_t value = get_ctrl();
    if (endBit - startBit == 63) {
        return value;
    }
    value = value >> startBit;
    value &= ((uint64_t(1) << (endBit - startBit + 1)) - 1);
    return value;
}

template <int8_t startBit, int8_t endBit>
__aicore__ static inline void ResetCtrlSprImpl()
{
    static_assert((startBit <= endBit && startBit >= 0 && endBit < 64), "Invalid bit range on current device!");
    static_assert((6 <= startBit && startBit <= 10 && 6 <= endBit && endBit <= 10) ||
                      (startBit == endBit && (startBit == 45 || startBit == 48 || startBit == 50 || startBit == 53 ||
                                              startBit == 59 || startBit == 60)),
                  "Invalid startBit/endBit on current device!");
    int64_t defaultCtrl = 0x1000000000000008; // default value of ctrl
    if (endBit - startBit == 63) {
        set_ctrl(defaultCtrl);
        return;
    }
    uint64_t mask = ((uint64_t(1) << (endBit - startBit + 1)) - 1) << startBit;
    defaultCtrl = defaultCtrl & mask;
    mask = ~mask;
    int64_t value = get_ctrl() & mask;
    value = value | defaultCtrl;
    set_ctrl(value);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
