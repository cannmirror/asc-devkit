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
 * \file kernel_operator_common_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#define ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
#include "kernel_common.h"
#include "kernel_utils.h"
#include "kernel_tensor.h"
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

__aicore__ inline int64_t GetBlockIdxImpl()
{
    // get_block_idx --- return 0 ~ 23 in aic/aiv
    // get_subblockdim --- return 1 in aic, return 2 in aiv
    // get_subblockid --- return 0 in aic, return 0 ~ 1 in aiv
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if ASCEND_IS_AIV {
        return block_idx * g_taskRation + sub_block_idx;
    }
    return block_idx;
#else
    if ASCEND_IS_AIV {
        return get_block_idx() * GetTaskRationImpl() + get_subblockid();
    } else {
        return get_block_idx();
    }
#endif
}

[[deprecated(
    "NOTICE: SetSysWorkSpace has been deprecated and will be removed in the next version.")]]
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
    g_sysWorkspaceReserved = workspace;
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
    return g_sysWorkspaceReserved + RESERVED_WORKSPACE;
#endif
}

template <atomic_type_t type, atomic_op_t op>
__aicore__ inline void SetStoreAtomicConfigImpl()
{
    set_st_atomic_cfg(type, op);
}

__aicore__ inline int64_t GetStoreAtomicConfigImpl()
{
    return get_st_atomic_cfg();
}

__aicore__ inline void GetStoreAtomicConfigImpl(uint16_t &atomicType, uint16_t &atomicOp)
{
    int64_t stAtomic = get_st_atomic_cfg();
    constexpr uint64_t typeMask = 0x7;
    constexpr uint64_t opBit = 3;
    constexpr uint64_t opMask = 0x3;
    atomicType = (static_cast<uint64_t>(stAtomic) & typeMask);
    atomicOp = ((static_cast<uint64_t>(stAtomic) >> opBit) & opMask);
}

template <typename T>
__aicore__ inline void DataCachePreloadImpl(const GlobalTensor<uint64_t> &src, const T cacheOffset)
{
    if constexpr ((IsSameType<T, int16_t>::value) || (IsSameType<T, int64_t>::value)) {
        dc_preload((__gm__ uint64_t *)src.GetPhyAddr(), cacheOffset);
    } else {
        ASCENDC_REPORT_NOT_SUPPORT(false, "current data type");
    }
}

__aicore__ inline void PreLoadImpl(void *pc, const int64_t preFetchLen)
{
    preload(pc, preFetchLen);
}

__aicore__ inline int64_t GetICachePreloadStatusImpl()
{
    return get_icache_prl_st();
}

__aicore__ inline void CheckLocalMemoryIAImpl(const CheckLocalMemoryIAParam& checkParams)
{
    uint64_t config = 0;
    config = config | (static_cast<uint64_t>(checkParams.startAddr) << 48);     // start address, DEC[63:48]
    config = config | (static_cast<uint64_t>(checkParams.endAddr) << 32);       // end address, DEC[47:32]
    config = config | (static_cast<uint64_t>(checkParams.isScalarRead) << 31);  // scalar read access, DEC[31]
    config = config | (static_cast<uint64_t>(checkParams.isScalarWrite) << 30); // scalar write access, DEC[30]
    config = config | (static_cast<uint64_t>(checkParams.isVectorRead) << 29);  // vector read access, DEC[29]
    config = config | (static_cast<uint64_t>(checkParams.isVectorWrite) << 28); // vector write access, DEC[28]
    config = config | (static_cast<uint64_t>(checkParams.isMteRead) << 27);     // vector mte read access, DEC[27]
    config = config | (static_cast<uint64_t>(checkParams.isMteWrite) << 26);    // vector mte write access, DEC[26]
    config = config | (checkParams.reserved << 1);                              // reserved, DEC[25:1]
    config = config | (static_cast<uint8_t>(checkParams.isEnable));             // enable bit, DEC[0]
    if (checkParams.enableBit == SET_DATA_EXP_ZERO) {
        set_data_exp_0(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_ONE) {
        set_data_exp_1(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_TWO) {
        set_data_exp_2(config);
    } else if (checkParams.enableBit == SET_DATA_EXP_THREE) {
        set_data_exp_3(config);
    } else {
        ASCENDC_REPORT_NOT_SUPPORT(false, "enableBit");
    }
}

__aicore__ inline void PreLoad(const int64_t preFetchLen)
{
    int64_t pc = get_pc() & 0xFFFFFFFFFFFF;
    PreLoadImpl(reinterpret_cast<void *>(pc), preFetchLen);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H
