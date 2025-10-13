/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_server_impl.h
 * \brief
 */
#ifndef IMPL_MATMUL_KFC_MATMUL_SERVER_IMPL_H
#define IMPL_MATMUL_KFC_MATMUL_SERVER_IMPL_H

#include "matmul_server_impl_base.h"

namespace AscendC {
#if !defined(__DAV_C310__) && !defined(__DAV_310R6__)
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::Init(MSG_POS KfcMsg* msg)
{
    if constexpr (!ToMatmulConfig(MM_CFG).enableInit) {
        return;
    } else {
        ASSERT(msg != nullptr && "msg cannot be nullptr when init matmul server");
        ASSERT(msg->tilingInfo.tilingAddr != nullptr && "tiling cannot be nullptr when init matmul server");
        auto temp1 = ((__gm__ uint32_t*)(msg->tilingInfo.tilingAddr));
        tiling_.SetTiling(&tmpTiling_);
        auto temp2 = (uint32_t*)(tiling_.GetTiling());

        constexpr uint32_t tCubeTilingSize = ConstCeil(sizeof(TCubeTiling), CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
        GlobalTensor<int64_t> tilingGlobal;
        for (int i = 0; i < tCubeTilingSize; i += CACHE_LINE_SIZE) {
            Barrier();
            tilingGlobal.SetGlobalBuffer((__gm__ int64_t *)(msg->tilingInfo.tilingAddr + i));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tilingGlobal);
        }

        for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.Init(this->tiling_.GetTiling(), nullptr);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetOrgShape(MSG_POS KfcMsg* msg)
{
    if constexpr (!ToMatmulConfig(MM_CFG).enableInit) {
        if (mul.GetSubBlockIdx() == 0) {
            msgAux.msg0.orgM = msg->orgShape.orgM;
            msgAux.msg0.orgN = msg->orgShape.orgN;
            msgAux.msg0.orgKa = msg->orgShape.orgKa;
            msgAux.msg0.orgKb = msg->orgShape.orgKb;
            msgAux.msg0.orgKc = msg->orgShape.orgKc;
            msgAux.msg0.setOrgShape = true;
        } else {
            msgAux.msg1.orgM = msg->orgShape.orgM;
            msgAux.msg1.orgN = msg->orgShape.orgN;
            msgAux.msg1.orgKa = msg->orgShape.orgKa;
            msgAux.msg1.orgKb = msg->orgShape.orgKb;
            msgAux.msg1.orgKc = msg->orgShape.orgKc;
            msgAux.msg1.setOrgShape = true;
        }
    } else {
        mul.SetOrgShape(msg->orgShape.orgM, msg->orgShape.orgN, msg->orgShape.orgKa, msg->orgShape.orgKb,
            msg->orgShape.orgKc);
    }
}
#endif

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorA(MsgTmpPos MatmulConfigParams* body)
{
#if !defined(__DAV_C310__)
    if (!body->setTensorA)
        return;
#endif
    if constexpr (A_TYPE::format == CubeFormat::SCALAR) {
        SrcAT scalar;
        auto temp1 = reinterpret_cast<MsgTmpPos uint8_t*>(&(body->aAddr));
        auto temp2 = (uint8_t*)&scalar;

        for (int i = 0; i < sizeof(SrcAT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.SetTensorA(scalar);
        return;
    }
    uint64_t size = 0;
#if !defined(__DAV_C310__) && !defined(__DAV_310R6__)
    size = (uint64_t)(body->sizeAmatrix); // not defined in C310
#endif
    if constexpr (NeedTransitByGm(A_TYPE::pos)) {
        GlobalTensor<SrcAT> aGlobal;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcAT*>(body->aAddr), size);
        mul.SetTensorA(aGlobal, body->isTransA);
    } else {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        // use addr to send intraId in C310
        if constexpr (PhyPosIsL1(A_TYPE::pos)) {
            KfcSetIntraAId(mul, body->aAddr >> VALID_ADDR_BITS_NUM);
        }
#endif
        const auto& aLocal = GetLocalTensor<typename A_TYPE::T, A_TYPE::pos>(body->aAddr, size);
        mul.SetTensorA(aLocal, body->isTransA);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorA(
    MsgTmpPos MatmulConfigParams* body, const uint64_t size, const uint64_t offset)
{
    if (!body->setTensorA) {
        return;
    }
    if constexpr (A_TYPE::format == CubeFormat::SCALAR) {
        SrcAT scalar;
        auto temp1 = reinterpret_cast<MsgTmpPos uint8_t*>(&(body->aAddr) + offset);
        auto temp2 = (uint8_t*)&scalar;

        for (int i = 0; i < sizeof(SrcAT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.SetTensorA(scalar);
        return;
    }
    if constexpr (PhyPosIsL1(A_TYPE::pos)) {
        const auto& aLocal = GetLocalTensor<typename A_TYPE::T, A_TYPE::pos>(body->aAddr + offset, size);
        mul.SetTensorA(aLocal, body->isTransA);
    } else {
        GlobalTensor<SrcAT> aGlobal;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcAT*>(body->aAddr + offset), size);
        mul.SetTensorA(aGlobal, body->isTransA);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorB(MsgTmpPos MatmulConfigParams* body)
{
#if !defined(__DAV_C310__)
    if (!body->setTensorB)
        return;
#endif
    if constexpr (B_TYPE::format == CubeFormat::SCALAR) {
        SrcBT scalar;
        auto temp1 = reinterpret_cast<MsgTmpPos uint8_t*>(&(body->bAddr));
        auto temp2 = (uint8_t*)&scalar;

        for (int i = 0; i < sizeof(SrcBT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.SetTensorB(scalar);
        return;
    }
    uint64_t size = 0;
#if !defined(__DAV_C310__) && !defined(__DAV_310R6__)
    size = (uint64_t)(body->sizeBmatrix); // not defined in C310
#endif
    if constexpr (NeedTransitByGm(B_TYPE::pos)) {
        GlobalTensor<SrcBT> bGlobal;
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcBT*>(body->bAddr), size);
        mul.SetTensorB(bGlobal, body->isTransB);
    } else {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        // use addr to send intraId in C310
        if constexpr (PhyPosIsL1(B_TYPE::pos)) {
            KfcSetIntraBId(mul, body->bAddr >> VALID_ADDR_BITS_NUM);
        }
#endif
        const auto& bLocal = GetLocalTensor<typename B_TYPE::T, B_TYPE::pos>(body->bAddr, size);
        mul.SetTensorB(bLocal, body->isTransB);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetTensorB(
    MsgTmpPos MatmulConfigParams* body, const uint64_t size, const uint64_t offset)
{
    if (!body->setTensorB) {
        return;
    }
    if constexpr (B_TYPE::format == CubeFormat::SCALAR) {
        SrcBT scalar;
        auto temp1 = reinterpret_cast<MsgTmpPos uint8_t*>(&(body->bAddr) + offset);
        auto temp2 = (uint8_t*)&scalar;

        for (int i = 0; i < sizeof(SrcBT); i++, temp1++, temp2++) {
            *temp2 = *temp1;
        }
        mul.SetTensorB(scalar);
        return;
    }
    if constexpr (PhyPosIsL1(B_TYPE::pos)) {
        const auto& bLocal = GetLocalTensor<typename B_TYPE::T, B_TYPE::pos>(body->bAddr + offset, size);
        mul.SetTensorB(bLocal, body->isTransB);
    } else {
        GlobalTensor<SrcBT> bGlobal;
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ SrcBT*>(body->bAddr + offset), size);
        mul.SetTensorB(bGlobal, body->isTransB);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetBias(MsgTmpPos MatmulConfigParams* body)
{
    if (body->setTensorBias) {
        const uint64_t size = (uint64_t)tiling_.GetSingleCoreN();
        if constexpr (NeedTransitByGm(BIAS_TYPE::pos)) {
            GlobalTensor<typename BIAS_TYPE::T> biasGlobal;
            biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ typename BIAS_TYPE::T*>(body->biasAddr), size);
            mul.SetBias(biasGlobal);
        } else {
            const auto& biasLocal = GetLocalTensor<typename BIAS_TYPE::T, BIAS_TYPE::pos>(body->biasAddr, size);
            mul.SetBias(biasLocal);
        }
    } else {
        mul.DisableBias();
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::SetBias(
    MsgTmpPos MatmulConfigParams* body, const uint64_t offset)
{
    if (body->setTensorBias) {
        const uint64_t size = (uint64_t)tiling_.GetSingleCoreN();
        if constexpr (PhyPosIsL1(BIAS_TYPE::pos)) {
            const auto& biasLocal = GetLocalTensor<typename BIAS_TYPE::T, BIAS_TYPE::pos>(body->biasAddr + offset, size);
            mul.SetBias(biasLocal);
        } else {
            GlobalTensor<typename BIAS_TYPE::T> biasGlobal;
            biasGlobal.SetGlobalBuffer(
                reinterpret_cast<__gm__ typename BIAS_TYPE::T*>(body->biasAddr + offset), size);
            mul.SetBias(biasGlobal);
        }
    } else {
        mul.DisableBias();
    }
}

#if !defined(__DAV_C310__) && !defined(__DAV_310R6__)
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::GetTensorC(MSG_POS KfcMsg* msg)
{
    if constexpr (A_TYPE::layout != LayoutMode::NONE) {
        return true;
    }
    uint64_t size;
    if constexpr (ToMatmulConfig(MM_CFG).baseMN != 0) {
        size = ToMatmulConfig(MM_CFG).baseMN;
    } else {
        size = tiling_.GetBaseM() * tiling_.GetBaseN();
    }
    if constexpr (PhyPosIsL1(C_TYPE::pos)) {
        const auto& cLocal = GetLocalTensor<typename C_TYPE::T, C_TYPE::pos>(msg->body.cAddr, size);
        mul.GetTensorC(cLocal, (uint8_t)(msg->body.enAtomic), msg->body.enSequentialWrite);
    } else {
        GlobalTensor<DstT> cGlobal;
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(msg->body.cAddr), size);
        mul.GetTensorC(cGlobal, (uint8_t)(msg->body.enAtomic), msg->body.enSequentialWrite);
    }
    // Now release UB
    if constexpr (PhyPosIsUB(C_TYPE::pos)) {
        if (unlikely(msg->ubAddr >= 0)) {
            kfcCommSrv->FreeUB(msg->ubAddr);
        }
    }
    if (msg->body.sync == 1) { // Synchronize
        uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
        NotifyEvent<PIPE_FIX>(eventID);
    }
    return false;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::IterateBatch(MSG_POS KfcMsg* msg)
{
    if constexpr (A_TYPE::layout == LayoutMode::NONE) {
        return true;
    }
    // In the batch scenario, messages occupy 128 bytes. After the update, messages occupy 64 bytes.
    GlobalTensor<int64_t> msgGlobalTensor;
    msgGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(msg) + sizeof(int64_t));
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobalTensor);
    __gm__ auto *body = &(msg->body);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if (body->setQuant == 1) {
        ASSERT(body->quantMode != 1);  // scalar mode is not supported for quantization parameters in
        // Batch MM
    }
#endif
    IterateSetMessage(msg, body);
    uint64_t size = tiling_.GetSingleCoreM() * tiling_.GetSingleCoreN();

    GlobalTensor<DstT> cGlobal;
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(body->cAddr), size);
    mul.IterateBatch(cGlobal, body->enPartialSum, (uint8_t)(body->enAtomic),body->enSequentialWrite,
                    body->matrixStrideA, body->matrixStrideB, body->matrixStrideC);
    mul.End();

    // Now release UB
    if constexpr (PhyPosIsUB(A_TYPE::pos) || PhyPosIsUB(B_TYPE::pos) ||
        PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsUB(C_TYPE::pos)) {
        if (unlikely(msg->ubAddr >= 0)) {
            kfcCommSrv->FreeUB(msg->ubAddr);
        }
    }
    if (body->sync || body->waitIterateBatch) {
        uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
        NotifyEvent<PIPE_FIX>(eventID);
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline void MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::StartIterateNBatch(
    MsgTmpPos MatmulConfigParams* body, uint32_t &cntIterator)
{
    const uint64_t size = tiling_.GetSingleCoreM() * tiling_.GetSingleCoreN();
    uint64_t singleBatchASize = (uint64_t)(body->sizeAmatrix) / body->batchLoop;
    uint64_t singleBatchBSize = (uint64_t)(body->sizeBmatrix) / body->batchLoop;
    uint32_t batchC = body->batchA > body->batchB ? body->batchA : body->batchB;
    bool layoutGCondition = tiling_.GetCLayoutInfoG() == 1 &&
                            (tiling_.GetBLayoutInfoG() != 1 || tiling_.GetALayoutInfoG() != 1);
    if (layoutGCondition) {
        int32_t layoutG = tiling_.GetBLayoutInfoG() > tiling_.GetALayoutInfoG() ? tiling_.GetBLayoutInfoG() : tiling_.GetALayoutInfoG();
        batchC = batchC / layoutG;
    }
    uint64_t batchOffsetBias = tiling_.GetCLayoutInfoS2() * batchC * sizeof(typename BIAS_TYPE::T);
    
    BmmOffset batchLoopOffset;
    for (uint32_t loopIdx = 0U; loopIdx < body->batchLoop; loopIdx++) {
        const uint64_t biasOffset = batchOffsetBias * loopIdx;
        batchLoopOffset.offA = CalcNBatchoffset<A_TYPE>(body->batchA, loopIdx, tiling_.GetALayoutInfoN(), tiling_.GetALayoutInfoG(), tiling_.GetALayoutInfoD(), tiling_.GetALayoutInfoS());
        batchLoopOffset.offB = CalcNBatchoffset<B_TYPE>(body->batchB, loopIdx, tiling_.GetBLayoutInfoN(), tiling_.GetBLayoutInfoG(), tiling_.GetBLayoutInfoD(), tiling_.GetBLayoutInfoS());
        batchLoopOffset.offC = CalcNBatchoffset<C_TYPE>(batchC, loopIdx, tiling_.GetCLayoutInfoN(), tiling_.GetCLayoutInfoG(), tiling_.GetCLayoutInfoS2(), tiling_.GetCLayoutInfoS1());

        IterateSetMessage(body, singleBatchASize, singleBatchBSize, batchLoopOffset.offA, batchLoopOffset.offB, biasOffset);
        GlobalTensor<DstT> cGlobal;
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(body->cAddr + batchLoopOffset.offC), size);
        mul.IterateBatch(cGlobal, body->enPartialSum, (uint8_t)(body->enAtomic),
            body->enSequentialWrite, body->matrixStrideA,
            body->matrixStrideB, body->matrixStrideC);
        cntIterator++;
        if (cntIterator < INC_PROCESS_CHECK && (!body->sync && !body->waitIterateBatch)) {
            uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
            NotifyEvent<PIPE_FIX>(eventID);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::IterateNBatch(MSG_POS KfcMsg* msg)
{
    if constexpr (!ToMatmulConfig(MM_CFG).isNBatch) {
        return true;
    }
    GlobalTensor<int64_t> msgGlobal;
    msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(msg) + sizeof(int64_t));
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobal);
    __gm__ auto *body = &(msg->body);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if (msg->body.setQuant == 1) {
        ASSERT(msg->body.quantMode != 1);  // scalar mode is not supported for quantization parameters in
        // Batch MM
    }
#endif
    uint32_t cntIterator = 0;
    StartIterateNBatch(body, cntIterator);
    // Now release UB
    if constexpr (PhyPosIsUB(A_TYPE::pos) || PhyPosIsUB(B_TYPE::pos) ||
        PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsUB(C_TYPE::pos)) {
        if (unlikely(msg->ubAddr >= 0)) {
            kfcCommSrv->FreeUB(msg->ubAddr);
        }
    }
    uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
    if (msg->body.sync || msg->body.waitIterateBatch) {
        NotifyEvent<PIPE_FIX>(eventID);
    } else if (cntIterator >= INC_PROCESS_CHECK) {
        NotifyEvent<PIPE_FIX>(eventID);
    }
    return true;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG, class MM_CB,
    MATMUL_POLICY_TEMPLATE_OF(MATMUL_POLICY)>
__aicore__ inline bool MatmulService<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>::Iterate(
    MSG_POS KfcMsg* msg, KFC_Enum funID)
{
    if constexpr (A_TYPE::layout != LayoutMode::NONE) {
        return true;
    }
    if constexpr ((A_TYPE::ibShare == true) || (B_TYPE::ibShare == true)) {
        if (msg->body.iterateFakeMsg) {
            if (funID == KFC_Enum::MMFUN_ITERATE_ALL) { // fake msg
                uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + kfcCommSrv->subBlockID);
                NotifyEvent<PIPE_FIX>(eventID);
                return true;
            }
        }
    } else {
        ASSERT(!msg->body.iterateFakeMsg &&"Only Ib share mode support fake msg.");
    }
    SyncCubeWithVec<A_TYPE::ibShare, B_TYPE::ibShare>();
    QuantCacheRefresh(msg);
    IterateSetMessage(msg, &(msg->body));
    uint32_t cntIterator = 0;
    auto sync = msg->body.sync;
    if(!StartIterate(&(msg->body), funID, sync, cntIterator)) {
        return false;
    }
    // Now release UB
    if constexpr (PhyPosIsUB(A_TYPE::pos) || PhyPosIsUB(B_TYPE::pos) ||
        PhyPosIsUB(BIAS_TYPE::pos) || PhyPosIsUB(C_TYPE::pos)) {
        if (unlikely(msg->ubAddr >= 0)) {
            kfcCommSrv->FreeUB(msg->ubAddr);
        }
    }

    uint16_t eventID = static_cast<uint16_t>(this->devEvtID * 2 + mul.GetSubBlockIdx());
    if (sync || msg->body.waitIterateAll) {
        ASSERT(funID == KFC_Enum::MMFUN_ITERATE_ALL);
        NotifyEvent<PIPE_FIX>(eventID);
    } else if (cntIterator >= INC_PROCESS_CHECK && funID == KFC_Enum::MMFUN_ITERATE) {
        NotifyEvent<PIPE_FIX>(eventID);
    }
    mul.End();
    TRACE_STOP(TraceId::MatMul_CALC);
    return true;
}
#endif
} // namespace AscendC
#endif // __MATMUL_SERVER_IMPL_H__
