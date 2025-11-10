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

/* !
 * \file softmax_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_IMPL_H

#include "softmax_common_impl.h"

namespace AscendC {
template <typename T1, typename T2, bool isLog = false>
__no_simd_vf_fusion__ __simd_vf__ inline void SoftMaxGenericNZVFImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, __local_mem__ float* expUb, __local_mem__ T2* tmpUb,
    __local_mem__ float* workUb, const LastAxisShapeND originalSrcShape, uint16_t srcM, uint16_t srcK,
    uint16_t VcgFoldRepeat, uint16_t e2bRep, uint16_t dtypeRepStride, uint16_t dtypeBlkStride)
{
    uint16_t originK = originalSrcShape.k;
    uint16_t originM = originalSrcShape.m;
    uint16_t dataBlock = srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint16_t mRepeatTimes = static_cast<uint16_t>(CeilDivision(dataBlock, FLOAT_REPEAT_SIZE));
    uint16_t kRepeatTimes = originK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    MicroAPI::MaskReg pregCnt;
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::RegTensor<T2> castReg;

    // reducemax
    for (uint16_t i = 0; i < mRepeatTimes; ++i) {
        Duplicate(maxVreg, notNum.f);
        for (uint16_t j = 0; j < kRepeatTimes; ++j) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, pregFull);
            MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregFull);
        }
        MicroAPI::ReduceMaxWithDataBlock(maxVreg, maxVreg, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(workUb + i * DEFAULT_BLK_NUM, maxVreg, pregOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < VcgFoldRepeat; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            maxVreg, tmpVreg, workUb + i * HALF_REPEAT_SIZE);
        MicroAPI::Max(maxVreg, maxVreg, tmpVreg, pregFull);
        StoreIfNeedCast<T2>(tmpUb + i * FLOAT_REPEAT_SIZE, maxVreg, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
            workUb + i * HALF_REPEAT_SIZE, maxVreg, maxVreg, pregFull);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    uint32_t sreg = originM * dtypeBlkStride;
    for (uint16_t i = 0; i < e2bRep; ++i) {
        pregCnt = MicroAPI::UpdateMask<T2>(sreg);
        LoadE2B<T2>(castReg, tmpUb + i * DEFAULT_BLK_NUM);
        MicroAPI::DataCopy(maxUb + i * dtypeRepStride, castReg, pregCnt);
    }

    // reducesum
    for (uint16_t i = 0; i < mRepeatTimes; ++i) {
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        Duplicate(sumVreg, 0);
        LoadE2B<float>(maxVreg, workUb + i * DEFAULT_BLK_NUM);
        for (uint16_t j = 0; j < kRepeatTimes; ++j) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, pregFull);
            MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregFull);
            MicroAPI::Exp(tmpVreg, dstVreg, pregFull);
            MicroAPI::DataCopy(expUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, tmpVreg, pregFull);
            MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        }
        MicroAPI::ReduceSumWithDataBlock(sumVreg, sumVreg, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(workUb + i * DEFAULT_BLK_NUM, sumVreg, pregOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < VcgFoldRepeat; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            sumVreg, tmpVreg, workUb + i * HALF_REPEAT_SIZE);
        MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        StoreIfNeedCast<T2>(tmpUb + i * FLOAT_REPEAT_SIZE, sumVreg, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
            workUb + i * HALF_REPEAT_SIZE, sumVreg, sumVreg, pregFull);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    sreg = originM * dtypeBlkStride;
    for (uint16_t i = 0; i < e2bRep; ++i) {
        pregCnt = MicroAPI::UpdateMask<T2>(sreg);
        LoadE2B<T2>(castReg, tmpUb + i * DEFAULT_BLK_NUM);
        MicroAPI::DataCopy(sumUb + i * dtypeRepStride, castReg, pregCnt);
    }

    for (uint16_t j = 0; j < kRepeatTimes; ++j) {
        sreg = originM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        for (uint16_t i = 0; i < mRepeatTimes; ++i) {
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy(tmpVreg, expUb + i * FLOAT_REPEAT_SIZE + j * dataBlock);
            LoadE2B<float>(sumVreg, workUb + i * DEFAULT_BLK_NUM);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregCnt);
            if constexpr (isLog) {
                MicroAPI::Log10(dstVreg, dstVreg, pregCnt);
            }
            StoreIfNeedCast<T1>(dstUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, dstVreg, pregCnt);
        }
    }
}

template <typename T1, typename T2, bool isLog = false>
__aicore__ inline void SoftMaxGenericNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = tiling.srcM;
    uint16_t srcK = tiling.srcK;

    uint16_t VcgFoldRepeat = static_cast<uint16_t>(CeilDivision(srcM, FLOAT_REPEAT_SIZE));
    uint16_t e2bRep = srcM / DEFAULT_BLK_NUM;
    uint16_t dtypeRepStride = IsSameType<T2, half>::value ? HALF_REPEAT_SIZE : FLOAT_REPEAT_SIZE;
    uint16_t dtypeBlkStride = dtypeRepStride / DEFAULT_BLK_NUM;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* expUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ T2* tmpUb = (__local_mem__ T2*)workLocal.GetPhyAddr(srcM * srcK);
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK + VcgFoldRepeat * FLOAT_REPEAT_SIZE);

    SoftMaxGenericNZVFImpl<T1, T2, isLog>(dstUb, sumUb, maxUb, srcUb, expUb, tmpUb,
        workUb, originalSrcShape, srcM, srcK, VcgFoldRepeat, e2bRep, dtypeRepStride, dtypeBlkStride);
}

template <typename T1, typename T2, bool isLog = false>
__no_simd_vf_fusion__ __simd_vf__ inline void SoftMaxGenericNZWithTailVFImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, __local_mem__ float* expUb, __local_mem__ T2* tmpUb,
    __local_mem__ float* workUb, uint16_t srcM, uint16_t srcK, uint16_t originM, uint16_t VcgFoldRepeat,
    uint16_t kRepeatTimes, uint16_t e2bRep, uint16_t dtypeRepStride, uint16_t dtypeBlkStride)
{
    uint16_t dataBlock = srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint16_t mRepeatTimes = static_cast<uint16_t>(CeilDivision(dataBlock, FLOAT_REPEAT_SIZE));
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    MicroAPI::MaskReg pregkTail = MicroAPI::MoveMask<uint32_t>();
    MicroAPI::MaskReg pregCnt;
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> minVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::RegTensor<T2> castReg;

    // reducemax
    Duplicate(minVreg, notNum.f);
    for (uint16_t i = 0; i < mRepeatTimes; ++i) {
        Duplicate(maxVreg, notNum.f);
        for (uint16_t j = 0; j < kRepeatTimes; ++j) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, pregFull);
            MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregFull);
        }
        LoadIfNeedCast<T1>(srcVreg, srcUb + i * FLOAT_REPEAT_SIZE + kRepeatTimes * dataBlock, pregFull);
        MicroAPI::Select(srcVreg, srcVreg, minVreg, pregkTail);
        MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregFull);

        MicroAPI::ReduceMaxWithDataBlock(maxVreg, maxVreg, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(workUb + i * DEFAULT_BLK_NUM, maxVreg, pregOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < VcgFoldRepeat; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            maxVreg, tmpVreg, workUb + i * HALF_REPEAT_SIZE);
        MicroAPI::Max(maxVreg, maxVreg, tmpVreg, pregFull);
        StoreIfNeedCast<T2>(tmpUb + i * FLOAT_REPEAT_SIZE, maxVreg, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
            workUb + i * HALF_REPEAT_SIZE, maxVreg, maxVreg, pregFull);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    uint32_t sreg = originM * dtypeBlkStride;
    for (uint16_t i = 0; i < e2bRep; ++i) {
        pregCnt = MicroAPI::UpdateMask<T2>(sreg);
        LoadE2B<T2>(castReg, tmpUb + i * DEFAULT_BLK_NUM);
        MicroAPI::DataCopy(maxUb + i * dtypeRepStride, castReg, pregCnt);
    }

    // reducesum
    for (uint16_t i = 0; i < mRepeatTimes; ++i) {
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        Duplicate(sumVreg, 0);
        LoadE2B<float>(maxVreg, workUb + i * DEFAULT_BLK_NUM);
        for (uint16_t j = 0; j < kRepeatTimes; ++j) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, pregFull);
            MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregFull);
            MicroAPI::Exp(tmpVreg, dstVreg, pregFull);
            MicroAPI::DataCopy(expUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, tmpVreg, pregFull);
            MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        }
        LoadIfNeedCast<T1>(srcVreg, srcUb + i * FLOAT_REPEAT_SIZE + kRepeatTimes * dataBlock, pregFull);
        MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregkTail);
        MicroAPI::Exp(tmpVreg, dstVreg, pregkTail);
        MicroAPI::DataCopy(expUb + i * FLOAT_REPEAT_SIZE + kRepeatTimes * dataBlock, tmpVreg, pregkTail);
        MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);

        MicroAPI::ReduceSumWithDataBlock(sumVreg, sumVreg, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(workUb + i * DEFAULT_BLK_NUM, sumVreg, pregOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < VcgFoldRepeat; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            sumVreg, tmpVreg, workUb + i * HALF_REPEAT_SIZE);
        MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        StoreIfNeedCast<T2>(tmpUb + i * FLOAT_REPEAT_SIZE, sumVreg, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_INTLV_B32>(
            workUb + i * HALF_REPEAT_SIZE, sumVreg, sumVreg, pregFull);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    sreg = originM * dtypeBlkStride;
    for (uint16_t i = 0; i < e2bRep; ++i) {
        pregCnt = MicroAPI::UpdateMask<T2>(sreg);
        LoadE2B<T2>(castReg, tmpUb + i * DEFAULT_BLK_NUM);
        MicroAPI::DataCopy(sumUb + i * dtypeRepStride, castReg, pregCnt);
    }

    for (uint16_t j = 0; j < kRepeatTimes; ++j) {
        sreg = originM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        for (uint16_t i = 0; i < mRepeatTimes; ++i) {
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy(tmpVreg, expUb + i * FLOAT_REPEAT_SIZE + j * dataBlock);
            LoadE2B<float>(sumVreg, workUb + i * DEFAULT_BLK_NUM);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregCnt);
            if constexpr (isLog) {
                MicroAPI::Log10(dstVreg, dstVreg, pregCnt);
            }
            StoreIfNeedCast<T1>(dstUb + i * FLOAT_REPEAT_SIZE + j * dataBlock, dstVreg, pregCnt);
        }
    }
    sreg = originM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    for (uint16_t i = 0; i < mRepeatTimes; ++i) {
        pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
        MicroAPI::DataCopy(tmpVreg, expUb + i * FLOAT_REPEAT_SIZE + kRepeatTimes * dataBlock);
        LoadE2B<float>(sumVreg, workUb + i * DEFAULT_BLK_NUM);
        MicroAPI::MaskAnd(pregOneBlk, pregkTail, pregCnt, pregFull);
        MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregOneBlk);
        if constexpr (isLog) {
            MicroAPI::Log10(dstVreg, dstVreg, pregOneBlk);
        }
        StoreIfNeedCast<T1>(dstUb + i * FLOAT_REPEAT_SIZE + kRepeatTimes * dataBlock, dstVreg, pregOneBlk);
    }
}

template <typename T1, typename T2, bool isLog = false>
__aicore__ inline void SoftMaxGenericNZWithTailImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = tiling.srcM;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t originM = originalSrcShape.m;
    uint16_t kRepeatTimes = originK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint16_t VcgFoldRepeat = static_cast<uint16_t>(CeilDivision(srcM, FLOAT_REPEAT_SIZE));
    uint16_t e2bRep = srcM / DEFAULT_BLK_NUM;
    uint16_t dtypeRepStride = IsSameType<T2, half>::value ? HALF_REPEAT_SIZE : FLOAT_REPEAT_SIZE;
    uint16_t dtypeBlkStride = dtypeRepStride / DEFAULT_BLK_NUM;
    uint64_t mask[2] = {0, 0};
    CreateSpecialFormatMask(mask[0], originK % SOFTMAX_SHAPE_NZ_BASIC_COUNT, FLOAT_REPEAT_SIZE / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
    SetVectorMask<uint32_t>(mask[1], mask[0]);

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* expUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ T2* tmpUb = (__local_mem__ T2*)workLocal.GetPhyAddr(srcM * srcK);
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK + VcgFoldRepeat * FLOAT_REPEAT_SIZE);

    SoftMaxGenericNZWithTailVFImpl<T1, T2, isLog>(dstUb, sumUb, maxUb, srcUb, expUb, tmpUb,
        workUb, srcM, srcK, originM, VcgFoldRepeat, kRepeatTimes, e2bRep, dtypeRepStride, dtypeBlkStride);
}

template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftMaxNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    static_assert((SupportType<T1, float>() && SupportType<T2, float>()) ||
                  (SupportType<T1, half>() && SupportType<T2, half>()) ||
                  (SupportType<T1, half>() && SupportType<T2, float>()),
                  "SoftMax api only support half/float on current device");
    if (tiling.srcK != originalSrcShape.k) {
        SoftMaxGenericNZWithTailImpl<T1, T2>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
    } else {
        SoftMaxGenericNZImpl<T1, T2>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
    }
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__no_simd_vf_fusion__ __simd_vf__ inline void SoftMaxGenericNDVFImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, __local_mem__ float* tmpUb, __local_mem__ float* workUb,
    uint16_t srcM, uint16_t srcK, uint16_t repeatTimes, uint16_t blockStride)
{
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOneBlk;
    MicroAPI::MaskReg pregOnePt = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL1>();
    if constexpr (IsSameType<T2, half>::value) {
        pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL16>();
    } else {
        pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    }
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;

    for (uint16_t i = 0; i < srcM; ++i) {
        Duplicate(maxVreg, notNum.f);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregFull);
        }
        MicroAPI::ReduceMax(maxVreg, maxVreg, pregFull);
        if constexpr (outputBrc) {
            Duplicate(maxVreg, maxVreg, pregOneBlk);
            StoreIfNeedCast<T2>(maxUb + i * blockStride, maxVreg, pregOneBlk);
        } else {
            StoreIfNeedCastM1<T2>(maxUb + i, maxVreg, pregOnePt);
        }

        Duplicate(sumVreg, 0);
        Duplicate(maxVreg, maxVreg, pregFull);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregFull);
            MicroAPI::Exp(tmpVreg, dstVreg, pregFull);
            if constexpr (!isFlashV2) {
                MicroAPI::DataCopy(workUb + i * srcK + j * FLOAT_REPEAT_SIZE, tmpVreg, pregFull);
            } else {
                StoreIfNeedCast<T1>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, tmpVreg, pregFull);
            }
            MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        }
        MicroAPI::ReduceSum(sumVreg, sumVreg, pregFull);
        if constexpr (outputBrc) {
            Duplicate(sumVreg, sumVreg, pregOneBlk);
            StoreIfNeedCast<T2>(sumUb + i * blockStride, sumVreg, pregOneBlk);
        } else {
            StoreIfNeedCastM1<T2>(sumUb + i, sumVreg, pregOnePt);
        }
        if constexpr (!isFlashV2 && sizeof(T2) == sizeof(half)) {
            MicroAPI::DataCopy(tmpUb + i * blockStride, sumVreg, pregOneBlk);
        }
    }

    if constexpr (!isFlashV2) {
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < srcM; ++i) {
            if constexpr (sizeof(T2) == sizeof(half)) {
                MicroAPI::DataCopy(sumVreg, tmpUb + i * blockStride);
            } else {
                MicroAPI::DataCopy(sumVreg, sumUb + i * blockStride);
            }
            Duplicate(sumVreg, sumVreg, pregFull);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                MicroAPI::DataCopy(tmpVreg, workUb + i * srcK + j * FLOAT_REPEAT_SIZE);
                MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregFull);
                if constexpr (isLog) {
                    MicroAPI::Log10(dstVreg, dstVreg, pregFull);
                }
                StoreIfNeedCast<T1>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, dstVreg, pregFull);
            }
        }
    }
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__aicore__ inline void SoftMaxGenericNDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(originK, FLOAT_REPEAT_SIZE));
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK
                                                                 : B32_DATA_NUM_PER_BLOCK;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * blockStride);

    SoftMaxGenericNDVFImpl<T1, T2, isFlashV2, isLog, outputBrc>(dstUb, sumUb,
         maxUb, srcUb, tmpUb, workUb, srcM, srcK, repeatTimes, blockStride);
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__no_simd_vf_fusion__ __simd_vf__ inline void SoftMaxGenericNDWithTailVFImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, __local_mem__ float* tmpUb, __local_mem__ float* workUb,
    uint16_t srcM, uint16_t srcK, uint16_t originK, uint16_t repeatTimes, uint16_t blockStride)
{
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    MicroAPI::MaskReg pregCnt;
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOnePt = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg pregOneBlk;
    if constexpr (IsSameType<T2, half>::value) {
        pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL16>();
    } else {
        pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    }
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> minVreg;
    MicroAPI::RegTensor<float> dstVreg;

    Duplicate(minVreg, notNum.f);
    for (uint16_t i = 0; i < srcM; ++i) {
        uint32_t sreg = originK;
        Duplicate(maxVreg, notNum.f);
        for (uint16_t j = 0; j < static_cast<uint16_t>(repeatTimes - 1); ++j) {
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregCnt);
        }
        pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
        LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK + (repeatTimes - 1) * FLOAT_REPEAT_SIZE, pregFull);
        MicroAPI::Select(srcVreg, srcVreg, minVreg, pregCnt);
        MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregFull);

        MicroAPI::ReduceMax(maxVreg, maxVreg, pregFull);
        if constexpr (outputBrc) {
            Duplicate(maxVreg, maxVreg, pregOneBlk);
            StoreIfNeedCast<T2>(maxUb + i * blockStride, maxVreg, pregOneBlk);
        } else {
            StoreIfNeedCastM1<T2>(maxUb + i, maxVreg, pregOnePt);
        }

        Duplicate(sumVreg, 0);
        Duplicate(maxVreg, maxVreg, pregFull);
        sreg = originK;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregCnt);
            MicroAPI::Exp(tmpVreg, dstVreg, pregCnt);
            if constexpr (!isFlashV2) {
                MicroAPI::DataCopy(workUb + i * srcK + j * FLOAT_REPEAT_SIZE, tmpVreg, pregCnt);
            } else {
                StoreIfNeedCast<T1>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, tmpVreg, pregCnt);
            }
            MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        }
        MicroAPI::ReduceSum(sumVreg, sumVreg, pregFull);
        if constexpr (outputBrc) {
            Duplicate(sumVreg, sumVreg, pregOneBlk);
            StoreIfNeedCast<T2>(sumUb + i * blockStride, sumVreg, pregOneBlk);
        } else {
            StoreIfNeedCastM1<T2>(sumUb + i, sumVreg, pregOnePt);
        }
        if constexpr (!isFlashV2 && sizeof(T2) == sizeof(half)) {
            MicroAPI::DataCopy(tmpUb + i * blockStride, sumVreg, pregOneBlk);
        }
    }

    if constexpr (!isFlashV2) {
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < srcM; ++i) {
            if constexpr (sizeof(T2) == sizeof(half)) {
                MicroAPI::DataCopy(sumVreg, tmpUb + i * blockStride);
            } else {
                MicroAPI::DataCopy(sumVreg, sumUb + i * blockStride);
            }
            Duplicate(sumVreg, sumVreg, pregFull);
            uint32_t sreg = originK;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
                MicroAPI::DataCopy(tmpVreg, workUb + i * srcK + j * FLOAT_REPEAT_SIZE);
                MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregCnt);
                if constexpr (isLog) {
                    MicroAPI::Log10(dstVreg, dstVreg, pregCnt);
                }
                StoreIfNeedCast<T1>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, dstVreg, pregCnt);
            }
        }
    }
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__aicore__ inline void SoftMaxGenericNDWithTailImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(originK, FLOAT_REPEAT_SIZE));
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK
                                                                 : B32_DATA_NUM_PER_BLOCK;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * blockStride);

    SoftMaxGenericNDWithTailVFImpl<T1, T2, isFlashV2, isLog, outputBrc>(dstUb, sumUb,
        maxUb, srcUb, tmpUb, workUb, srcM, srcK, originK, repeatTimes, blockStride);
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__no_simd_vf_fusion__ __simd_vf__ inline void SingleSoftMaxGenericNDForBlkVFImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, __local_mem__ float* tmpUb0, __local_mem__ float* tmpUb1, __local_mem__ float* workUb,
    uint16_t srcM, uint16_t srcK, uint16_t factorRow, uint16_t factor, uint16_t blockStride)
{
    uint32_t sreg = srcK * srcM;
    MicroAPI::MaskReg pregDst;
    MicroAPI::MaskReg pregOut;
    MicroAPI::MaskReg pregCnt = MicroAPI::MoveMask<uint32_t>();
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::UnalignReg ureg0;

    for (uint16_t i = 0; i < factor; ++i) {
        LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK * factorRow, pregFull);

        MicroAPI::ReduceMaxWithDataBlock(maxVreg, srcVreg, pregCnt);
        MicroAPI::DataCopy(tmpUb0 + i * factorRow, maxVreg, pregOneBlk);
        if constexpr (!outputBrc) {
            if constexpr (SupportType<T2, half>()) {
                MicroAPI::RegTensor<T2> castVreg;
                MicroAPI::Cast<T2, float, Internal::castTraitB32ToB16>(castVreg, maxVreg, pregOneBlk);
                MicroAPI::Pack<uint16_t, uint32_t>(
                    (MicroAPI::RegTensor<uint16_t>&)castVreg, (MicroAPI::RegTensor<uint32_t>&)castVreg);
                MicroAPI::DataCopyUnAlign(maxUb, castVreg, ureg0, factorRow);
                MicroAPI::DataCopyUnAlignPost(maxUb, ureg0, 0);
            } else {
                MicroAPI::DataCopy<float>(maxUb + i * factorRow, maxVreg, pregOneBlk);
            }
        }
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < factor; ++i) {
        pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
        LoadE2B<float>(maxVreg, tmpUb0 + i * factorRow);
        if constexpr (outputBrc) {
            StoreIfNeedCast<T2>(maxUb + i * blockStride * factorRow, maxVreg, pregOut);
        }

        LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK * factorRow, pregFull);
        MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregFull);
        MicroAPI::Exp(tmpVreg, dstVreg, pregFull);
        if constexpr (!isFlashV2) {
            MicroAPI::DataCopy(workUb + i * srcK * factorRow, tmpVreg, pregOut);
        } else {
            MicroAPI::MaskAnd(pregDst, pregCnt, pregOut, pregFull);
            StoreIfNeedCast<T1>(dstUb + i * srcK * factorRow, tmpVreg, pregDst);
        }

        MicroAPI::ReduceSumWithDataBlock(sumVreg, tmpVreg, pregCnt);
        MicroAPI::DataCopy(tmpUb1 + i * factorRow, sumVreg, pregOneBlk);
        if constexpr (!outputBrc) {
            if constexpr (SupportType<T2, half>()) {
                MicroAPI::RegTensor<T2> castVreg;
                MicroAPI::Cast<T2, float, Internal::castTraitB32ToB16>(castVreg, sumVreg, pregOneBlk);
                MicroAPI::Pack<uint16_t, uint32_t>(
                    (MicroAPI::RegTensor<uint16_t>&)castVreg, (MicroAPI::RegTensor<uint32_t>&)castVreg);
                MicroAPI::DataCopyUnAlign(sumUb, castVreg, ureg0, factorRow);
                MicroAPI::DataCopyUnAlignPost(sumUb, ureg0, 0);
            } else {
                MicroAPI::DataCopy<float>(sumUb + i * factorRow, sumVreg, pregOneBlk);
            }
        }
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    if constexpr (isFlashV2 && outputBrc) {
        sreg = srcK * srcM;
        for (uint16_t i = 0; i < factor; ++i) {
            pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadE2B<float>(tmpVreg, tmpUb1 + i * factorRow);
            StoreIfNeedCast<T2>(sumUb + i * blockStride * factorRow, tmpVreg, pregOut);
        }
    }

    if constexpr (!isFlashV2) {
        sreg = srcK * srcM;
        for (uint16_t i = 0; i < factor; ++i) {
            pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadE2B<float>(sumVreg, tmpUb1 + i * factorRow);
            StoreIfNeedCast<T2>(sumUb + i * blockStride * factorRow, tmpVreg, pregOut);
            MicroAPI::DataCopy(tmpVreg, workUb + i * srcK * factorRow);
            MicroAPI::MaskAnd(pregDst, pregCnt, pregOut, pregFull);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregDst);
            if constexpr (isLog) {
                MicroAPI::Log10(dstVreg, dstVreg, pregDst);
            }
            StoreIfNeedCast<T1>(dstUb + i * srcK * factorRow, dstVreg, pregDst);
        }
    }
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__aicore__ inline void SingleSoftMaxGenericNDForBlkImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t factorRow = FLOAT_REPEAT_SIZE / srcK;
    uint16_t factor = CeilDivision(srcM, factorRow);
    uint16_t originK = originalSrcShape.k;
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK
                                                                 : B32_DATA_NUM_PER_BLOCK;
    uint64_t mask[2] = {0,0};
    CreateSpecialFormatMask(mask[0], originK, factorRow, srcK);
    SetVectorMask<uint32_t>(mask[1], mask[0]);

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* tmpUb0 = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* tmpUb1 = (__local_mem__ float*)workLocal.GetPhyAddr(factorRow * factor);
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(factorRow * factor * 2);

    SingleSoftMaxGenericNDForBlkVFImpl<T1, T2, isFlashV2, isLog, outputBrc>(dstUb, sumUb,
        maxUb, srcUb, tmpUb0, tmpUb1, workUb, srcM, srcK, factorRow, factor, blockStride);
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__no_simd_vf_fusion__ __simd_vf__ inline void SingleSoftMaxGenericNDAlignedWithBlkVFImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, __local_mem__ float* tmpUb0, __local_mem__ float* tmpUb, __local_mem__ float* tmpUb1,
    __local_mem__ float* workUb, uint16_t srcM, uint16_t srcK, uint16_t factorRow, uint16_t factor, uint16_t originK, uint16_t blockStride)
{
    uint16_t halfFactor = CeilDivision(srcM, factorRow * 2);
    uint32_t sreg = srcK * srcM;
    uint32_t sreg1 = srcM * blockStride;
    __local_mem__ float* tmpUb0Tmp0 = tmpUb0;
    __local_mem__ float* tmpUb0Tmp1 = tmpUb0;

    MicroAPI::MaskReg pregDst;
    MicroAPI::MaskReg pregTmp;
    MicroAPI::MaskReg pregOut;
    MicroAPI::MaskReg pregCnt = MicroAPI::MoveMask<uint32_t>();
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::UnalignReg ureg0;
    MicroAPI::UnalignReg ureg1;
    MicroAPI::UnalignReg ureg2;

    for (uint16_t i = 0; i < factor; ++i) {
        LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK * factorRow, pregFull);

        MicroAPI::ReduceMaxWithDataBlock(maxVreg, srcVreg, pregCnt);

        Duplicate(tmpVreg, 0);
        MicroAPI::DeInterleave(maxVreg, tmpVreg, maxVreg, tmpVreg);
        MicroAPI::Max(maxVreg, maxVreg, tmpVreg, pregFull);
        if constexpr (!outputBrc) {
            if constexpr (SupportType<T2, half>()) {
                MicroAPI::RegTensor<T2> castVreg;
                MicroAPI::Cast<T2, float, Internal::castTraitB32ToB16>(castVreg, maxVreg, pregOneBlk);
                MicroAPI::Pack<uint16_t, uint32_t>(
                    (MicroAPI::RegTensor<uint16_t>&)castVreg, (MicroAPI::RegTensor<uint32_t>&)castVreg);
                MicroAPI::DataCopyUnAlign(maxUb, castVreg, ureg2, factorRow);
            } else {
                MicroAPI::DataCopyUnAlign(maxUb, maxVreg, ureg2, factorRow);
            }
        }
        if constexpr (sizeof(T2) == sizeof(float) && outputBrc) {
            MicroAPI::DataCopyUnAlign(tmpUb0Tmp0, maxVreg, ureg0, factorRow);
        }
        MicroAPI::Interleave(maxVreg, tmpVreg, maxVreg, maxVreg);
        MicroAPI::DataCopy(tmpUb1 + i * 2 * factorRow, maxVreg, pregOneBlk);
    }
    if constexpr (sizeof(T2) == sizeof(float) && outputBrc) {
        MicroAPI::DataCopyUnAlignPost(tmpUb0Tmp0, ureg0, 0);
    } else if constexpr (!outputBrc) {
        MicroAPI::DataCopyUnAlignPost(maxUb, ureg2, 0);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    if constexpr (sizeof(T2) == sizeof(float) && outputBrc) {
        for (uint16_t i = 0; i < halfFactor; ++i) {
            pregTmp = MicroAPI::UpdateMask<uint32_t>(sreg1);
            LoadE2B<float>(tmpVreg, tmpUb0 + i * DEFAULT_BLK_NUM);
            StoreIfNeedCast<T2>(maxUb + i * blockStride * factorRow * 2, tmpVreg, pregTmp);
        }
    }

    sreg = srcK * srcM;
    for (uint16_t i = 0; i < factor; ++i) {
        pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
        LoadE2B<float>(maxVreg, tmpUb1 + i * DEFAULT_BLK_NUM);
        if constexpr (sizeof(T2) == sizeof(half) && outputBrc) {
            StoreIfNeedCast<T2>(maxUb + i * blockStride * factorRow, maxVreg, pregOut);
        }

        LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK * factorRow, pregFull);
        MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregFull);
        MicroAPI::Exp(tmpVreg, dstVreg, pregFull);
        if constexpr (!isFlashV2) {
            MicroAPI::DataCopy(workUb + i * srcK * factorRow, tmpVreg, pregOut);
        } else {
            MicroAPI::MaskAnd(pregDst, pregCnt, pregOut, pregFull);
            StoreIfNeedCast<T1>(dstUb + i * srcK * factorRow, tmpVreg, pregDst);
        }

        MicroAPI::ReduceSumWithDataBlock(sumVreg, tmpVreg, pregCnt);

        Duplicate(tmpVreg, 0);
        MicroAPI::DeInterleave(sumVreg, tmpVreg, sumVreg, tmpVreg);
        MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        if constexpr (!outputBrc) {
            if constexpr (SupportType<T2, half>()) {
                MicroAPI::RegTensor<T2> castVreg;
                MicroAPI::Cast<T2, float, Internal::castTraitB32ToB16>(castVreg, sumVreg, pregOneBlk);
                MicroAPI::Pack<uint16_t, uint32_t>(
                    (MicroAPI::RegTensor<uint16_t>&)castVreg, (MicroAPI::RegTensor<uint32_t>&)castVreg);
                MicroAPI::DataCopyUnAlign(sumUb, castVreg, ureg2, factorRow);
            } else {
                MicroAPI::DataCopyUnAlign(sumUb, sumVreg, ureg2, factorRow);
            }
        }
        if constexpr (sizeof(T2) == sizeof(float) && outputBrc) {
            MicroAPI::DataCopyUnAlign(tmpUb0Tmp1, sumVreg, ureg1, factorRow);
        }
        MicroAPI::Interleave(sumVreg, tmpVreg, sumVreg, sumVreg);
        MicroAPI::DataCopy(tmpUb + i * 2 * factorRow, sumVreg, pregOneBlk);
    }
    if constexpr (sizeof(T2) == sizeof(float) && outputBrc) {
        MicroAPI::DataCopyUnAlignPost(tmpUb0Tmp1, ureg1, 0);
    } else if (!outputBrc) {
        MicroAPI::DataCopyUnAlignPost(sumUb, ureg2, 0);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    if constexpr (outputBrc) {
        if constexpr (sizeof(T2) == sizeof(float)) {
            sreg1 = srcM * blockStride;
            for (uint16_t i = 0; i < halfFactor; ++i) {
                pregTmp = MicroAPI::UpdateMask<uint32_t>(sreg1);
                LoadE2B<float>(tmpVreg, tmpUb0 + i * DEFAULT_BLK_NUM);
                StoreIfNeedCast<T2>(sumUb + i * blockStride * factorRow * 2, tmpVreg, pregTmp);
            }
        } else if constexpr (sizeof(T2) == sizeof(half)) {
            sreg = srcM * blockStride;
            for (uint16_t i = 0; i < factor; ++i) {
                pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
                LoadE2B<float>(tmpVreg, tmpUb + i * DEFAULT_BLK_NUM);
                StoreIfNeedCast<T2>(sumUb + i * blockStride * factorRow, tmpVreg, pregOut);
            }
        }
    }

    if constexpr (!isFlashV2) {
        sreg = srcK * srcM;
        for (uint16_t i = 0; i < factor; ++i) {
            pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadE2B<float>(sumVreg, tmpUb + i * DEFAULT_BLK_NUM);
            MicroAPI::DataCopy(tmpVreg, workUb + i * srcK * factorRow);
            MicroAPI::MaskAnd(pregDst, pregCnt, pregOut, pregFull);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregDst);
            if constexpr (isLog) {
                MicroAPI::Log10(dstVreg, dstVreg, pregDst);
            }
            StoreIfNeedCast<T1>(dstUb + i * srcK * factorRow, dstVreg, pregDst);
        }
    }
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__aicore__ inline void SingleSoftMaxGenericNDAlignedWithBlkImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t factorRow = FLOAT_REPEAT_SIZE / srcK;
    uint16_t factor = CeilDivision(srcM, factorRow);
    uint16_t originK = originalSrcShape.k;
    uint16_t offset = static_cast<uint16_t>(CeilDivision(srcM, B32_DATA_NUM_PER_BLOCK) * B32_DATA_NUM_PER_BLOCK);
    uint16_t offset1 = (factor - 1) * factorRow * 2 + FLOAT_REPEAT_SIZE * 2;
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK
                                                                 : B32_DATA_NUM_PER_BLOCK;
    uint64_t mask[2] = {0,0};
    CreateSpecialFormatMask(mask[0], originK, factorRow, srcK);
    SetVectorMask<uint32_t>(mask[1], mask[0]);

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* tmpUb0 = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK);
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK + offset);
    __local_mem__ float* tmpUb1 = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK + offset + offset1);

    SingleSoftMaxGenericNDAlignedWithBlkVFImpl<T1, T2, isFlashV2, isLog, outputBrc>(dstUb, sumUb,
        maxUb, srcUb, tmpUb0, tmpUb, tmpUb1, workUb, srcM, srcK, factorRow, factor, originK, blockStride);
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__no_simd_vf_fusion__ __simd_vf__ inline void SingleSoftMaxGenericNDVFImpl(__local_mem__ T1* dstUb, __local_mem__ T2* sumUb,
    __local_mem__ T2* maxUb, __local_mem__ T1* srcUb, __local_mem__ float* tmpUb, __local_mem__ float* workUb,
    uint16_t srcM, uint16_t srcK, uint16_t originK, uint16_t blockStride)
{
    uint32_t sreg = originK;
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    MicroAPI::MaskReg pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOnePt = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg pregOneBlk;
    if constexpr (IsSameType<T2, half>::value) {
        pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL16>();
    } else {
        pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    }
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;

    for (uint16_t i = 0; i < srcM; ++i) {
        LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK, pregFull);

        MicroAPI::ReduceMax(maxVreg, srcVreg, pregCnt);
        if constexpr (!outputBrc) {
            StoreIfNeedCastM1<T2>(maxUb + i, maxVreg, pregOnePt);
        } else {
            Duplicate(maxVreg, maxVreg, pregOneBlk);
            StoreIfNeedCast<T2>(maxUb + i * blockStride, maxVreg, pregOneBlk);
        }

        Duplicate(maxVreg, maxVreg, pregFull);
        MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregCnt);
        MicroAPI::Exp(tmpVreg, dstVreg, pregCnt);
        if constexpr (!isFlashV2) {
            MicroAPI::DataCopy(workUb + i * srcK, tmpVreg, pregCnt);
        } else {
            StoreIfNeedCast<T1>(dstUb + i * srcK, tmpVreg, pregCnt);
        }
        MicroAPI::ReduceSum(sumVreg, tmpVreg, pregCnt);
        if constexpr (!outputBrc) {
            StoreIfNeedCastM1<T2>(sumUb + i, sumVreg, pregOnePt);
        } else {
            Duplicate(sumVreg, sumVreg, pregOneBlk);
            StoreIfNeedCast<T2>(sumUb + i * blockStride, sumVreg, pregOneBlk);
        }
        if constexpr (!isFlashV2 && sizeof(T2) == sizeof(half)) {
            MicroAPI::DataCopy(tmpUb + i * blockStride, sumVreg, pregOneBlk);
        }
    }

    if constexpr (!isFlashV2) {
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t i = 0; i < srcM; ++i) {
            if constexpr (sizeof(T2) == sizeof(half)) {
                MicroAPI::DataCopy(sumVreg, tmpUb + i * blockStride);
            } else {
                MicroAPI::DataCopy(sumVreg, sumUb + i * blockStride);
            }
            Duplicate(sumVreg, sumVreg, pregFull);
            MicroAPI::DataCopy(tmpVreg, workUb + i * srcK);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregCnt);
            if constexpr (isLog) {
                MicroAPI::Log10(dstVreg, dstVreg, pregCnt);
            }
            StoreIfNeedCast<T1>(dstUb + i * srcK, dstVreg, pregCnt);
        }
    }
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false, bool outputBrc = true>
__aicore__ inline void SingleSoftMaxGenericNDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK
                                                                 : B32_DATA_NUM_PER_BLOCK;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * blockStride);

    SingleSoftMaxGenericNDVFImpl<T1, T2, isFlashV2, isLog, outputBrc>(dstUb, sumUb,
        maxUb, srcUb, tmpUb, workUb, srcM, srcK, originK, blockStride);
}

template <typename T1, typename T2, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxNDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    static_assert((SupportType<T1, float>() && SupportType<T2, float>()) ||
                  (SupportType<T1, half>() && SupportType<T2, half>()) ||
                  (SupportType<T1, half>() && SupportType<T2, float>()),
                  "SoftMax api only support half/float on current device");
    if constexpr (isBasicBlock) {
        SoftMaxGenericNDImpl<T1, T2, false>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
    } else {
        if (tiling.srcK == B32_DATA_NUM_PER_BLOCK && IsSameType<T1, float>::value) {
            SingleSoftMaxGenericNDForBlkImpl<T1, T2, false>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        } else if (tiling.srcK == B32_DATA_NUM_PER_BLOCK * 2) {
            SingleSoftMaxGenericNDAlignedWithBlkImpl<T1, T2, false>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        } else if (originalSrcShape.k <= FLOAT_REPEAT_SIZE) {
            SingleSoftMaxGenericNDImpl<T1, T2, false>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        } else if (originalSrcShape.k % FLOAT_REPEAT_SIZE != 0) {
            SoftMaxGenericNDWithTailImpl<T1, T2, false>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        } else {
            SoftMaxGenericNDImpl<T1, T2, false>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        }
    }
}

template <typename T, bool isReuseSource = false>
__no_simd_vf_fusion__ __simd_vf__ inline void SoftMaxGenericNDWithTailVFImpl(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    __local_mem__ float* sumUb, __local_mem__ float* workUb, uint16_t srcM, uint16_t srcK, uint16_t originK, uint16_t repeatTimes)
{
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;
    __local_mem__ float* tmpUb = sumUb;

    MicroAPI::MaskReg pregCnt;
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> minVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::UnalignReg ureg0;

    Duplicate(minVreg, notNum.f);
    for (uint16_t i = 0; i < srcM; ++i) {
        uint32_t sreg = originK;
        Duplicate(maxVreg, notNum.f);
        for (uint16_t j = 0; j < static_cast<uint16_t>(repeatTimes - 1); ++j) {
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregCnt);
        }
        pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
        LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK + (repeatTimes - 1) * FLOAT_REPEAT_SIZE, pregFull);
        MicroAPI::Select(srcVreg, srcVreg, minVreg, pregCnt);
        MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregFull);

        MicroAPI::ReduceMax(maxVreg, maxVreg, pregFull);

        Duplicate(sumVreg, 0);
        Duplicate(maxVreg, maxVreg, pregFull);
        sreg = originK;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregCnt);
            MicroAPI::Exp(tmpVreg, dstVreg, pregCnt);
            MicroAPI::DataCopy(workUb + i * srcK + j * FLOAT_REPEAT_SIZE, tmpVreg, pregCnt);
            MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        }
        MicroAPI::ReduceSum(sumVreg, sumVreg, pregFull);
        MicroAPI::DataCopyUnAlign(sumUb, sumVreg, ureg0, 1);
    }
    MicroAPI::DataCopyUnAlignPost(sumUb, ureg0, 0);

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(sumVreg, tmpUb, 1);
        uint32_t sreg = originK;
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy(tmpVreg, workUb + i * srcK + j * FLOAT_REPEAT_SIZE);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregCnt);
            StoreIfNeedCast<T>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, dstVreg, pregCnt);
        }
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SoftMaxGenericNDWithTailImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(originK, FLOAT_REPEAT_SIZE));
    uint16_t offset = static_cast<uint16_t>(CeilDivision(srcM, B32_DATA_NUM_PER_BLOCK) * B32_DATA_NUM_PER_BLOCK);

    __local_mem__ T* dstUb = (__local_mem__ T*)dst.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)src.GetPhyAddr();
    __local_mem__ float* sumUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset);
    if constexpr (isReuseSource && sizeof(T) == sizeof(float)) {
        workUb = (__local_mem__ float*)src.GetPhyAddr();
    }

    SoftMaxGenericNDWithTailVFImpl<T, isReuseSource>(dstUb, srcUb, sumUb, workUb, srcM, srcK, originK, repeatTimes);
}

template <typename T, bool isReuseSource = false>
__no_simd_vf_fusion__ __simd_vf__ inline void SoftMaxGenericNDVFImpl(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    __local_mem__ float* sumUb, __local_mem__ float* maxUb, __local_mem__ float* workUb, __local_mem__ float* sumUb0,
    __local_mem__ float* sumUb1, uint16_t srcM, uint16_t srcK, uint16_t originK, uint16_t repeatTimes, uint16_t halfM,
    uint16_t tailM, uint16_t mainM)
{
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::RegTensor<float> srcVreg0;
    MicroAPI::RegTensor<float> maxVreg0;
    MicroAPI::RegTensor<float> sumVreg0;
    MicroAPI::RegTensor<float> tmpVreg0;
    MicroAPI::RegTensor<float> dstVreg0;

    MicroAPI::RegTensor<float> srcVreg1;
    MicroAPI::RegTensor<float> maxVreg1;
    MicroAPI::RegTensor<float> sumVreg1;
    MicroAPI::RegTensor<float> tmpVreg1;
    MicroAPI::RegTensor<float> dstVreg1;

    for (uint16_t i = 0; i < halfM; ++i) {
        Duplicate(maxVreg0, notNum.f);
        Duplicate(maxVreg1, notNum.f);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            LoadIfNeedCast<T>(srcVreg0, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            LoadIfNeedCast<T>(srcVreg1, srcUb + (i + halfM) * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::Max(maxVreg0, maxVreg0, srcVreg0, pregFull);
            MicroAPI::Max(maxVreg1, maxVreg1, srcVreg1, pregFull);
        }
        MicroAPI::ReduceMax(maxVreg0, maxVreg0, pregFull);
        MicroAPI::ReduceMax(maxVreg1, maxVreg1, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((maxUb + i), maxVreg0, pregOne);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((maxUb + i + halfM), maxVreg1, pregOne);
    }
    for (uint16_t i = 0; i < tailM; ++i) {
        Duplicate(maxVreg0, notNum.f);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            LoadIfNeedCast<T>(srcVreg0, srcUb + mainM * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::Max(maxVreg0, maxVreg0, srcVreg0, pregFull);
        }
        MicroAPI::ReduceMax(maxVreg0, maxVreg0, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((maxUb + mainM), maxVreg0, pregOne);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    for (uint16_t i = 0; i < halfM; i++) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg0, maxUb + i);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg1, maxUb + (i + halfM));
        Duplicate(sumVreg0, 0);
        Duplicate(sumVreg1, 0);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            LoadIfNeedCast<T>(srcVreg0, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            LoadIfNeedCast<T>(srcVreg1, srcUb + (i + halfM) * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::FusedExpSub(tmpVreg0, srcVreg0, maxVreg0, pregFull);
            MicroAPI::FusedExpSub(tmpVreg1, srcVreg1, maxVreg1, pregFull);
            MicroAPI::DataCopy(workUb + i * srcK + j * FLOAT_REPEAT_SIZE, tmpVreg0, pregFull);
            MicroAPI::DataCopy(workUb + (i + halfM) * srcK + j * FLOAT_REPEAT_SIZE, tmpVreg1, pregFull);
            MicroAPI::Add(sumVreg0, sumVreg0, tmpVreg0, pregFull);
            MicroAPI::Add(sumVreg1, sumVreg1, tmpVreg1, pregFull);
        }
        MicroAPI::ReduceSum(sumVreg0, sumVreg0, pregFull);
        MicroAPI::ReduceSum(sumVreg1, sumVreg1, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((sumUb0 + i), sumVreg0, pregOne);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((sumUb1 + i), sumVreg1, pregOne);
    }

    for (uint16_t i = 0; i < tailM; ++i) {
        Duplicate(sumVreg0, 0);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg0, maxUb + mainM);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            LoadIfNeedCast<T>(srcVreg0, srcUb + mainM * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
            MicroAPI::FusedExpSub(tmpVreg0, srcVreg0, maxVreg0, pregFull);
            MicroAPI::DataCopy(workUb + mainM * srcK + j * FLOAT_REPEAT_SIZE, tmpVreg0, pregFull);
            MicroAPI::Add(sumVreg0, sumVreg0, tmpVreg0, pregFull);
        }
        MicroAPI::ReduceSum(sumVreg0, sumVreg0, pregFull);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((sumUb + mainM), sumVreg0, pregOne);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < halfM; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(sumVreg0, sumUb + i);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(sumVreg1, sumUb + (i + halfM));
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            MicroAPI::DataCopy(tmpVreg0, workUb + i * srcK + j * FLOAT_REPEAT_SIZE);
            MicroAPI::DataCopy(tmpVreg1, workUb + (i + halfM) * srcK + j * FLOAT_REPEAT_SIZE);
            MicroAPI::Div(dstVreg0, tmpVreg0, sumVreg0, pregFull);
            MicroAPI::Div(dstVreg1, tmpVreg1, sumVreg1, pregFull);
            StoreIfNeedCast<T>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, dstVreg0, pregFull);
            StoreIfNeedCast<T>(dstUb + (i + halfM) * srcK + j * FLOAT_REPEAT_SIZE, dstVreg1, pregFull);
        }
    }
    for (uint16_t i = 0; i < tailM; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(sumVreg0, sumUb + mainM);
        for (uint16_t j = 0; j < repeatTimes; ++j) {
            MicroAPI::DataCopy(tmpVreg0, workUb + mainM * srcK + j * FLOAT_REPEAT_SIZE);
            MicroAPI::Div(dstVreg0, tmpVreg0, sumVreg0, pregFull);
            StoreIfNeedCast<T>(dstUb + mainM * srcK + j * FLOAT_REPEAT_SIZE, dstVreg0, pregFull);
        }
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SoftMaxGenericNDImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(originK, FLOAT_REPEAT_SIZE));
    uint16_t offset = static_cast<uint16_t>(CeilDivision(srcM, B32_DATA_NUM_PER_BLOCK) * B32_DATA_NUM_PER_BLOCK);
    uint16_t halfM = srcM / 2;
    uint16_t tailM = srcM % 2;
    uint16_t mainM = halfM * 2;

    __local_mem__ T* dstUb = (__local_mem__ T*)dst.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)src.GetPhyAddr();
    __local_mem__ float* sumUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* maxUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset);
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset * 2);

    __local_mem__ float* sumUb0 = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* sumUb1 = (__local_mem__ float*)workLocal.GetPhyAddr() + halfM;

    if constexpr (isReuseSource && sizeof(T) == sizeof(float)) {
        workUb = (__local_mem__ float*)src.GetPhyAddr();
    }

    SoftMaxGenericNDVFImpl<T, isReuseSource>(dstUb, srcUb, sumUb, maxUb, workUb, sumUb0,
        sumUb1, srcM, srcK, originK, repeatTimes, halfM, tailM, mainM);
}

template <typename T, bool isReuseSource = false>
__no_simd_vf_fusion__ __simd_vf__ inline void SingleSoftMaxGenericNDForBlkVFImpl(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    __local_mem__ float* sumUb, __local_mem__ float* tmpUb, __local_mem__ float* workUb, uint16_t srcM, uint16_t srcK, 
    uint16_t factorRow, uint16_t factor)
{
    uint32_t sreg = srcK * srcM;
    MicroAPI::MaskReg pregOut;
    MicroAPI::MaskReg pregCnt = MicroAPI::MoveMask<uint32_t>();
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;

    for (uint16_t i = 0; i < factor; ++i) {
        LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK * factorRow, pregFull);

        MicroAPI::ReduceMaxWithDataBlock(maxVreg, srcVreg, pregCnt);
        MicroAPI::DataCopy(tmpUb + i * factorRow, maxVreg, pregOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < factor; ++i) {
        LoadE2B<float>(maxVreg, tmpUb + i * factorRow);

        LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK * factorRow, pregFull);
        MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregFull);
        MicroAPI::Exp(tmpVreg, dstVreg, pregFull);
        MicroAPI::DataCopy(workUb + i * srcK * factorRow, tmpVreg, pregFull);

        MicroAPI::ReduceSumWithDataBlock(sumVreg, tmpVreg, pregCnt);
        MicroAPI::DataCopy(sumUb + i * factorRow, sumVreg, pregOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < factor; ++i) {
        pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
        LoadE2B<float>(sumVreg, sumUb + i * factorRow);
        MicroAPI::DataCopy(tmpVreg, workUb + i * srcK * factorRow);
        MicroAPI::MaskAnd(pregOneBlk, pregCnt, pregOut, pregFull);
        MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregOneBlk);
        StoreIfNeedCast<T>(dstUb + i * srcK * factorRow, dstVreg, pregOneBlk);
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SingleSoftMaxGenericNDForBlkImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t factorRow = FLOAT_REPEAT_SIZE / srcK;
    uint16_t factor = CeilDivision(srcM, factorRow);
    uint16_t originK = originalSrcShape.k;
    uint16_t offset = factor * factorRow;
    uint64_t mask[2] = {0,0};
    CreateSpecialFormatMask(mask[0], originK, factorRow, srcK);
    SetVectorMask<uint32_t>(mask[1], mask[0]);

    __local_mem__ T* dstUb = (__local_mem__ T*)dst.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)src.GetPhyAddr();
    __local_mem__ float* sumUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset);
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset * 2);
    if constexpr (isReuseSource && sizeof(T) == sizeof(float)) {
        workUb = (__local_mem__ float*)src.GetPhyAddr();
    }

    SingleSoftMaxGenericNDForBlkVFImpl<T, isReuseSource>(dstUb, srcUb, sumUb, tmpUb, workUb, srcM, srcK, factorRow, factor);
}

template <typename T, bool isReuseSource = false>
__no_simd_vf_fusion__ __simd_vf__ inline void SingleSoftMaxGenericNDAlignedWithBlkVFImpl(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    __local_mem__ float* sumUb, __local_mem__ float* tmpUb, __local_mem__ float* workUb, uint16_t srcM, uint16_t srcK, 
    uint16_t factorRow, uint16_t factor)
{
    uint32_t sreg = srcK * srcM;
    MicroAPI::MaskReg pregOut;
    MicroAPI::MaskReg pregCnt = MicroAPI::MoveMask<uint32_t>();
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregOneBlk = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::VL8>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;

    for (uint16_t i = 0; i < factor; ++i) {
        LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK * factorRow, pregFull);

        MicroAPI::ReduceMaxWithDataBlock(maxVreg, srcVreg, pregCnt);

        Duplicate(tmpVreg, 0);
        MicroAPI::DeInterleave(maxVreg, tmpVreg, maxVreg, tmpVreg);
        MicroAPI::Max(maxVreg, maxVreg, tmpVreg, pregFull);
        MicroAPI::Interleave(maxVreg, tmpVreg, maxVreg, maxVreg);
        MicroAPI::DataCopy(tmpUb + i * 2 * factorRow, maxVreg, pregOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < factor; ++i) {
        LoadE2B<float>(maxVreg, tmpUb + i * DEFAULT_BLK_NUM);

        LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK * factorRow, pregFull);
        MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregFull);
        MicroAPI::Exp(tmpVreg, dstVreg, pregFull);
        MicroAPI::DataCopy(workUb + i * srcK * factorRow, tmpVreg, pregFull);

        MicroAPI::ReduceSumWithDataBlock(sumVreg, tmpVreg, pregCnt);

        Duplicate(tmpVreg, 0);
        MicroAPI::DeInterleave(sumVreg, tmpVreg, sumVreg, tmpVreg);
        MicroAPI::Add(sumVreg, sumVreg, tmpVreg, pregFull);
        MicroAPI::Interleave(sumVreg, tmpVreg, sumVreg, sumVreg);
        MicroAPI::DataCopy(sumUb + i * 2 * factorRow, sumVreg, pregOneBlk);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < factor; ++i) {
        pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
        LoadE2B<float>(sumVreg, sumUb + i * DEFAULT_BLK_NUM);
        MicroAPI::DataCopy(tmpVreg, workUb + i * srcK * factorRow);
        MicroAPI::MaskAnd(pregOneBlk, pregCnt, pregOut, pregFull);
        MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregOneBlk);
        StoreIfNeedCast<T>(dstUb + i * srcK * factorRow, dstVreg, pregOneBlk);
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SingleSoftMaxGenericNDAlignedWithBlkImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t factorRow = FLOAT_REPEAT_SIZE / srcK;
    uint16_t factor = CeilDivision(srcM, factorRow);
    uint16_t originK = originalSrcShape.k;
    uint16_t offset = factor * factorRow * srcK;
    uint16_t offset1 = (factor - 1) * factorRow * 2 + FLOAT_REPEAT_SIZE * 2;
    uint64_t mask[2] = {0,0};
    CreateSpecialFormatMask(mask[0], originK, factorRow, srcK);
    SetVectorMask<uint32_t>(mask[1], mask[0]);

    __local_mem__ T* dstUb = (__local_mem__ T*)dst.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)src.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset);
    __local_mem__ float* sumUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset + offset1);
    if constexpr (isReuseSource && sizeof(T) == sizeof(float)) {
        workUb = (__local_mem__ float*)src.GetPhyAddr();
    }

    SingleSoftMaxGenericNDAlignedWithBlkVFImpl<T, isReuseSource>(dstUb, srcUb, sumUb, tmpUb, workUb, srcM, srcK,
        factorRow, factor);
}

template <typename T, bool isReuseSource = false>
__no_simd_vf_fusion__ __simd_vf__ inline void SingleSoftMaxGenericNDVFImpl(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    __local_mem__ float* sumUb, __local_mem__ float* tmpUb, __local_mem__ float* workUb, uint16_t srcM, uint16_t srcK, 
    uint16_t originK)
{
    uint32_t sreg = originK;
    MicroAPI::MaskReg pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> maxVreg;
    MicroAPI::RegTensor<float> sumVreg;
    MicroAPI::RegTensor<float> tmpVreg;
    MicroAPI::RegTensor<float> dstVreg;
    MicroAPI::UnalignReg ureg0;

    for (uint16_t i = 0; i < srcM; ++i) {
        LoadIfNeedCast<T>(srcVreg, srcUb + i * srcK, pregFull);

        MicroAPI::ReduceMax(maxVreg, srcVreg, pregCnt);

        Duplicate(maxVreg, maxVreg, pregFull);

        MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregCnt);
        MicroAPI::Exp(tmpVreg, dstVreg, pregCnt);
        MicroAPI::DataCopy(workUb + i * srcK, tmpVreg, pregCnt);

        MicroAPI::ReduceSum(sumVreg, tmpVreg, pregCnt);
        MicroAPI::DataCopyUnAlign(sumUb, sumVreg, ureg0, 1);
    }
    MicroAPI::DataCopyUnAlignPost(sumUb, ureg0, 0);

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < srcM; ++i) {
        MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(sumVreg, tmpUb, 1);
        MicroAPI::DataCopy(tmpVreg, workUb + i * srcK);
        MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregCnt);
        StoreIfNeedCast<T>(dstUb + i * srcK, dstVreg, pregCnt);
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SingleSoftMaxGenericNDImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t offset = static_cast<uint16_t>(CeilDivision(srcM, B32_DATA_NUM_PER_BLOCK) * B32_DATA_NUM_PER_BLOCK);
    
    __local_mem__ T* dstUb = (__local_mem__ T*)dst.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)src.GetPhyAddr();
    __local_mem__ float* sumUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* tmpUb = sumUb;
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset);
    if constexpr (isReuseSource && sizeof(T) == sizeof(float)) {
        workUb = (__local_mem__ float*)src.GetPhyAddr();
    }

    SingleSoftMaxGenericNDVFImpl<T, isReuseSource>(dstUb, srcUb, sumUb, tmpUb, workUb, srcM, srcK,
        originK);
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxNDImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    static_assert(SupportType<T, half, float>(), "SoftMax api only support half/float on current device");
    if constexpr (isBasicBlock) {
        SoftMaxGenericNDImpl<T, isReuseSource>(dst, src, workLocal, originalSrcShape, tiling);
    } else {
        if (tiling.srcK == B32_DATA_NUM_PER_BLOCK && IsSameType<T, float>::value) {
            SingleSoftMaxGenericNDForBlkImpl<T, isReuseSource>(dst, src, workLocal, originalSrcShape, tiling);
        } else if (tiling.srcK == B32_DATA_NUM_PER_BLOCK * 2) {
            SingleSoftMaxGenericNDAlignedWithBlkImpl<T, isReuseSource>(dst, src, workLocal, originalSrcShape, tiling);
        } else if (originalSrcShape.k <= FLOAT_REPEAT_SIZE) {
            SingleSoftMaxGenericNDImpl<T, isReuseSource>(dst, src, workLocal, originalSrcShape, tiling);
        } else if (originalSrcShape.k % FLOAT_REPEAT_SIZE != 0) {
            SoftMaxGenericNDWithTailImpl<T, isReuseSource>(dst, src, workLocal, originalSrcShape, tiling);
        } else {
            SoftMaxGenericNDImpl<T, isReuseSource>(dst, src, workLocal, originalSrcShape, tiling);
        }
    }
}

template <typename T1, typename T2, uint32_t stepSize, uint32_t stride>
__aicore__ inline void AdjustSoftMaxResNZImpl(__local_mem__ T1* resUb, __local_mem__ T2* maxUb,
    __local_mem__ uint64_t* maskUb, const uint32_t from, const T1 to, const uint32_t dataBlock,
    const uint16_t mRepeatTimes, const uint16_t kRepeatTimes)
{
    MicroAPI::RegTensor<T1> srcVreg;
    MicroAPI::RegTensor<T1> tmpVreg;
    MicroAPI::RegTensor<T1> dstVreg;
    MicroAPI::RegTensor<T2> maxVreg;
    MicroAPI::MaskReg cmpMaskReg;
    MicroAPI::MaskReg cmpMaskReg0;
    MicroAPI::MaskReg cmpMaskReg1;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg dstMask = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALLF>();

    bool isUpdateNeedCheck = false;
    for (uint16_t i = 0; i < mRepeatTimes; ++i) {
        if constexpr (sizeof(T2) == sizeof(float)) {
            MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg, maxUb + i * stepSize);
            // either full mask or zero mask
            MicroAPI::CompareScalar(cmpMaskReg, (MicroAPI::RegTensor<uint32_t>&)maxVreg, from, maskFull);
        } else if constexpr (sizeof(T2) == sizeof(half)) {
            MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_BRC_B16>(maxVreg, maxUb + i * stepSize);
            // either full mask or zero mask
            MicroAPI::CompareScalar(cmpMaskReg, (MicroAPI::RegTensor<uint16_t>&)maxVreg, (uint16_t)from, maskFull);
        }
        if constexpr (sizeof(T1) != sizeof(T2)) {
            MicroAPI::MaskPack(cmpMaskReg0, cmpMaskReg);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::HIGHEST>(cmpMaskReg1, cmpMaskReg);
            MicroAPI::MaskOr(cmpMaskReg, cmpMaskReg0, cmpMaskReg1, maskFull);
        }
        MicroAPI::MaskOr(dstMask, dstMask, cmpMaskReg, maskFull);
        MicroAPI::Duplicate(tmpVreg, to, maskFull);
        for (uint16_t j = 0; j < kRepeatTimes; ++j) {
            MicroAPI::DataCopy(srcVreg, resUb + i * stride + j * dataBlock);
            MicroAPI::Select(dstVreg, tmpVreg, srcVreg, cmpMaskReg);
            MicroAPI::DataCopy(resUb + i * stride + j * dataBlock, dstVreg, maskFull);
        }
    }
    MicroAPI::DataCopy((__local_mem__ uint8_t*)maskUb, dstMask);
}

template <typename T1, typename T2, uint32_t stepSize, uint32_t stride>
__aicore__ inline void AdjustSoftMaxResNDImpl(__local_mem__ T1* resUb, __local_mem__ T2* maxUb,
    __local_mem__ uint64_t* maskUb, const uint32_t from, const T1 to, const uint32_t srcK, const uint16_t srcM,
    const uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T1> srcVreg;
    MicroAPI::RegTensor<T1> tmpVreg;
    MicroAPI::RegTensor<T1> dstVreg;
    MicroAPI::RegTensor<T2> maxVreg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg cmpMaskReg;
    MicroAPI::MaskReg cmpMaskReg0;
    MicroAPI::MaskReg cmpMaskReg1;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg dstMask = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALLF>();

    for (uint16_t i = 0; i < srcM; i++) {
        if constexpr (sizeof(T2) == sizeof(float)) {
            MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg, maxUb + i * stepSize);
            // either full mask or zero mask
            MicroAPI::CompareScalar(cmpMaskReg, (MicroAPI::RegTensor<uint32_t>&)maxVreg, from, maskFull);
        } else if constexpr (sizeof(T2) == sizeof(half)) {
            MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_BRC_B16>(maxVreg, maxUb + i * stepSize);
            // either full mask or zero mask
            MicroAPI::CompareScalar(cmpMaskReg, (MicroAPI::RegTensor<uint16_t>&)maxVreg, (uint16_t)from, maskFull);
        }
        if constexpr (sizeof(T1) == sizeof(half) && sizeof(T2) == sizeof(float)) {
            MicroAPI::MaskPack(cmpMaskReg0, cmpMaskReg);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::HIGHEST>(cmpMaskReg1, cmpMaskReg);
            MicroAPI::MaskOr(cmpMaskReg, cmpMaskReg0, cmpMaskReg1, maskFull);
        }
        MicroAPI::MaskOr(dstMask, dstMask, cmpMaskReg, maskFull);
        MicroAPI::Duplicate(tmpVreg, to, maskFull);
        uint32_t sreg = srcK;
        for (uint16_t j = 0; j < repeatTimes; j++) {
            maskReg = MicroAPI::UpdateMask<T1>(sreg);
            MicroAPI::DataCopy(srcVreg, resUb + i * srcK + j * stride);
            MicroAPI::Select(dstVreg, tmpVreg, srcVreg, cmpMaskReg);
            MicroAPI::DataCopy(resUb + i * srcK + j * stride, dstVreg, maskReg);
        }
    }
    MicroAPI::DataCopy((__local_mem__ uint8_t*)maskUb, dstMask);
}

template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResBaseImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    SoftmaxApiSupportedTypeCheck<T1>();
    SoftmaxApiSupportedTypeCheck<T2>();
    static_assert((SupportType<Tuple<T1, T2>, Tuple<half, float>, Tuple<half, half>,
                Tuple<float, float>>()), "Failed to check dtype in SimpleSoftMax, current api "
                "support dtype combination is T1 : half, T2 : float; T1 : half, T2 : half; "
                "T1 : float, T2 : float");
    constexpr uint32_t stride = GetVecLen() / sizeof(T1);
    __local_mem__ T1* resUb = (__local_mem__ T1*)softMaxRes.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ uint64_t* maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 4);

    if constexpr (isDataFormatNZ) {
        constexpr uint32_t stepSize = GetDataBlockSizeInBytes() / sizeof(T2);
        uint32_t dataBlock = softmaxShapeInfo.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        uint16_t mRepeatTimes = static_cast<uint16_t>(CeilDivision(dataBlock, stride));
        uint16_t kRepeatTimes = static_cast<uint16_t>(softmaxShapeInfo.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
        VF_CALL<AdjustSoftMaxResNZImpl<T1, T2, stepSize, stride>>(resUb, maxUb, maskBuf, from,
                                                                  to, dataBlock, mRepeatTimes, kRepeatTimes);
    } else {
        uint32_t srcK = softmaxShapeInfo.srcK;
        uint16_t srcM = softmaxShapeInfo.srcM;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(softmaxShapeInfo.srcK, stride));
        if constexpr (stepSizeMode != 0) {
            constexpr uint32_t stepSize = 1;
            VF_CALL<AdjustSoftMaxResNDImpl<T1, T2, stepSize, stride>>(resUb, maxUb, maskBuf, from,
                                                                    to, srcK, srcM, repeatTimes);
        } else {
            constexpr uint32_t stepSize = GetDataBlockSizeInBytes() / sizeof(T2);
            VF_CALL<AdjustSoftMaxResNDImpl<T1, T2, stepSize, stride>>(resUb, maxUb, maskBuf, from,
                                                                    to, srcK, srcM, repeatTimes);
        }
    }
    auto eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
    bool isUpdateNeedCheck = *((__local_mem__ uint8_t*)maskBuf);
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
    return isUpdateNeedCheck;
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_IMPL_H
