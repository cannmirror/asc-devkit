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
 * \file matmul_server_impl_base.h
 * \brief
 */
#ifndef IMPL_MATMUL_KFC_MATMUL_SERVER_IMPL_BASE_H
#define IMPL_MATMUL_KFC_MATMUL_SERVER_IMPL_BASE_H

#include "matmul_server.h"
namespace AscendC {
#if !defined(__DAV_C310__) && !defined(__DAV_310R6__)
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetOffsetSize(
    MsgTmpPos MatmulConfigParams* body, KFC_Enum funID, uint32_t sync, uint64_t &offsetSize,
    uint32_t &enSequentialWrite, bool hasSetWorkspace)
{
    if constexpr ((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_NORMAL) == 0) {
        ASSERT(body->cAddr != 0); // The output address must be configured.
        if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
            offsetSize = enSequentialWrite ? ToMatmulConfig(MM_CFG).baseMN : 0;
        } else {
            offsetSize = enSequentialWrite ? (tiling_.GetBaseM() * tiling_.GetBaseN()) : 0;
        }
    } else {
        if (funID == KFC_Enum::MMFUN_ITERATE_ALL) {
            ASSERT(body->cAddr != 0); // The output address must be configured.
            if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
                offsetSize = enSequentialWrite ? ToMatmulConfig(MM_CFG).baseMN : 0;
            } else {
                offsetSize = enSequentialWrite ? (tiling_.GetBaseM() * tiling_.GetBaseN()) : 0;
            }
        } else if (sync == 0) {
            // For asynchronous Iterate, the offset must be used for address calculation and
            // the size is baseM x baseN.
            if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
                offsetSize = ToMatmulConfig(MM_CFG).baseMN;
            } else {
                offsetSize = tiling_.GetBaseM() * tiling_.GetBaseN();
            }
            enSequentialWrite = 1;
        }
    }
}


template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::StartIterate(
    MsgTmpPos MatmulConfigParams* body, KFC_Enum funID, uint32_t sync, uint32_t &cntIterator)
{
    uint64_t size;
    if constexpr (ToMatmulConfig(MM_CFG).singleCoreMN != 0) {
        size = ToMatmulConfig(MM_CFG).singleCoreMN;
    } else {
        size = tiling_.GetSingleCoreM() * tiling_.GetSingleCoreN();
    }

    GlobalTensor<DstT> cGlobal;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(body->cAddr), size);
    LocalTensor<DstT> cLocal;
    if constexpr (PhyPosIsL1(C_TYPE::pos)) {
        cLocal = GetLocalTensor<typename C_TYPE::T, C_TYPE::pos>(body->cAddr, size);
    }
    uint64_t offset = 0;
    uint64_t offsetSize = 0;
    auto enSequentialWrite = body->enSequentialWrite;
    auto enAtomic = body->enAtomic;
    auto enPartialSum = body->enPartialSum;
    GetOffsetSize(body, funID, sync, offsetSize, enSequentialWrite);
    TRACE_START(TraceId::MatMul_CALC);
    // Asynchronous and configure the workspace
    while (mul.Iterate(enPartialSum)) {
        if constexpr ((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_NORMAL) != 0) {
            if (unlikely(cntIterator == 0) && unlikely(funID == KFC_Enum::MMFUN_ITERATE && sync == 1)) {
                TRACE_STOP(TraceId::MatMul_CALC);
                return false; // The queue is not switched, and no message needs to be returned.
            }
        }
        if constexpr (PhyPosIsL1(C_TYPE::pos)) {
            mul.GetTensorC(cLocal[offset], (uint8_t)(enAtomic), enSequentialWrite);
        } else {
            mul.GetTensorC(cGlobal[offset], (uint8_t)(enAtomic), enSequentialWrite);
        }
        cntIterator++;
        if constexpr ((ToMatmulConfig(MM_CFG).iterateMode & IterateMode::ITERATE_MODE_NORMAL) != 0) {
            if (cntIterator < INC_PROCESS_CHECK && funID == KFC_Enum::MMFUN_ITERATE) {
                uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
                NotifyEvent<PIPE_FIX>(eventID);
            }
        }
        offset += offsetSize;
    }
    return true;
}
#endif
}  // namespace AscendC
#endif // IMPL_MATMUL_KFC_MATMUL_SERVER_IMPL_BASE_H