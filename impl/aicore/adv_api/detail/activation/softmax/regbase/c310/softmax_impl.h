/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SOFTMAX_IMPL_H
#define AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SOFTMAX_IMPL_H

#include "softmax_common_impl.h"

namespace AscendC {
template <typename T1, typename T2, bool isLog = false>
__aicore__ inline void SoftMaxGenericNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = tiling.srcM;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t originM = originalSrcShape.m;

    uint16_t dataBlock = srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint16_t mRepeatTimes = static_cast<uint16_t>(CeilDivision(dataBlock, FLOAT_REPEAT_SIZE));
    uint16_t kRepeatTimes = originK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint16_t VcgFoldRepeat = static_cast<uint16_t>(CeilDivision(srcM, FLOAT_REPEAT_SIZE));
    uint16_t e2bRep = srcM / DEFAULT_BLK_NUM;
    uint16_t dtypeRepStride = IsSameType<T2, half>::value ? HALF_REPEAT_SIZE : FLOAT_REPEAT_SIZE;
    uint16_t dtypeBlkStride = dtypeRepStride / DEFAULT_BLK_NUM;
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* expUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ T2* tmpUb = (__local_mem__ T2*)workLocal.GetPhyAddr(srcM * srcK);
    __local_mem__ float* workUb =
        (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK + VcgFoldRepeat * FLOAT_REPEAT_SIZE);

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
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
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(
                workUb + i * DEFAULT_BLK_NUM, maxVreg, pregOneBlk);
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
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(
                workUb + i * DEFAULT_BLK_NUM, sumVreg, pregOneBlk);
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

    uint16_t dataBlock = srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint16_t mRepeatTimes = static_cast<uint16_t>(CeilDivision(dataBlock, FLOAT_REPEAT_SIZE));
    uint16_t kRepeatTimes = originK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    uint16_t VcgFoldRepeat = static_cast<uint16_t>(CeilDivision(srcM, FLOAT_REPEAT_SIZE));
    uint16_t e2bRep = srcM / DEFAULT_BLK_NUM;
    uint16_t dtypeRepStride = IsSameType<T2, half>::value ? HALF_REPEAT_SIZE : FLOAT_REPEAT_SIZE;
    uint16_t dtypeBlkStride = dtypeRepStride / DEFAULT_BLK_NUM;
    uint64_t mask[2] = {0, 0};
    CreateSpecialFormatMask(
        mask[0], originK % SOFTMAX_SHAPE_NZ_BASIC_COUNT, FLOAT_REPEAT_SIZE / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
    SetVectorMask<uint32_t>(mask[1], mask[0]);
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* expUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ T2* tmpUb = (__local_mem__ T2*)workLocal.GetPhyAddr(srcM * srcK);
    __local_mem__ float* workUb =
        (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK + VcgFoldRepeat * FLOAT_REPEAT_SIZE);

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
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
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(
                workUb + i * DEFAULT_BLK_NUM, maxVreg, pregOneBlk);
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
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(
                workUb + i * DEFAULT_BLK_NUM, sumVreg, pregOneBlk);
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
}

template <typename T1, typename T2, bool isBasicBlock = false>
__aicore__ inline void SoftMaxNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    static_assert((SupportType<T1, float>() && SupportType<T2, float>())
                      || (SupportType<T1, half>() && SupportType<T2, half>())
                      || (SupportType<T1, half>() && SupportType<T2, float>()),
        "SoftMax api only support half/float on current device");
    if (tiling.srcK != originalSrcShape.k) {
        SoftMaxGenericNZWithTailImpl<T1, T2>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
    } else {
        SoftMaxGenericNZImpl<T1, T2>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
    }
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false>
__aicore__ inline void SoftMaxGenericNDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(originK, FLOAT_REPEAT_SIZE));
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK : B32_DATA_NUM_PER_BLOCK;
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * blockStride);

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
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
            Duplicate(maxVreg, notNum.f);
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK + j * FLOAT_REPEAT_SIZE, pregFull);
                MicroAPI::Max(maxVreg, maxVreg, srcVreg, pregFull);
            }
            MicroAPI::ReduceMax(maxVreg, maxVreg, pregFull);
            Duplicate(maxVreg, maxVreg, pregOneBlk);
            StoreIfNeedCast<T2>(maxUb + i * blockStride, maxVreg, pregOneBlk);

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
            Duplicate(sumVreg, sumVreg, pregOneBlk);
            StoreIfNeedCast<T2>(sumUb + i * blockStride, sumVreg, pregOneBlk);
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
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false>
__aicore__ inline void SoftMaxGenericNDWithTailImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(originK, FLOAT_REPEAT_SIZE));
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK : B32_DATA_NUM_PER_BLOCK;
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * blockStride);

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
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
            Duplicate(maxVreg, maxVreg, pregOneBlk);
            StoreIfNeedCast<T2>(maxUb + i * blockStride, maxVreg, pregOneBlk);

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
            Duplicate(sumVreg, sumVreg, pregOneBlk);
            StoreIfNeedCast<T2>(sumUb + i * blockStride, sumVreg, pregOneBlk);
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
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false>
__aicore__ inline void SingleSoftMaxGenericNDForBlkImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t factorRow = FLOAT_REPEAT_SIZE / srcK;
    uint16_t factor = CeilDivision(srcM, factorRow);
    uint16_t originK = originalSrcShape.k;
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK : B32_DATA_NUM_PER_BLOCK;
    uint32_t sreg = srcK * srcM;
    uint64_t mask[2] = {0, 0};
    CreateSpecialFormatMask(mask[0], originK, factorRow, srcK);
    SetVectorMask<uint32_t>(mask[1], mask[0]);

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* tmpUb0 = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* tmpUb1 = (__local_mem__ float*)workLocal.GetPhyAddr(factorRow * factor);
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(factorRow * factor * 2);

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
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

        for (uint16_t i = 0; i < factor; ++i) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK * factorRow, pregFull);

            MicroAPI::ReduceMaxWithDataBlock(maxVreg, srcVreg, pregCnt);
            MicroAPI::DataCopy(tmpUb0 + i * factorRow, maxVreg, pregOneBlk);
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        for (uint16_t i = 0; i < factor; ++i) {
            pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadE2B<float>(maxVreg, tmpUb0 + i * factorRow);
            StoreIfNeedCast<T2>(maxUb + i * blockStride * factorRow, maxVreg, pregOut);

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
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        sreg = srcK * srcM;
        for (uint16_t i = 0; i < factor; ++i) {
            pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadE2B<float>(tmpVreg, tmpUb1 + i * factorRow);
            StoreIfNeedCast<T2>(sumUb + i * blockStride * factorRow, tmpVreg, pregOut);
        }

        if constexpr (!isFlashV2) {
            sreg = srcK * srcM;
            for (uint16_t i = 0; i < factor; ++i) {
                pregOut = MicroAPI::UpdateMask<uint32_t>(sreg);
                LoadE2B<float>(sumVreg, tmpUb1 + i * factorRow);
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
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false>
__aicore__ inline void SingleSoftMaxGenericNDAlignedWithBlkImpl(const LocalTensor<T1>& dst,
    const LocalTensor<T2>& sumTensor, const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src,
    const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t factorRow = FLOAT_REPEAT_SIZE / srcK;
    uint16_t factor = CeilDivision(srcM, factorRow);
    uint16_t halfFactor = CeilDivision(srcM, factorRow * 2);
    uint16_t originK = originalSrcShape.k;
    uint16_t offset = static_cast<uint16_t>(CeilDivision(srcM, B32_DATA_NUM_PER_BLOCK) * B32_DATA_NUM_PER_BLOCK);
    uint16_t offset1 = (factor - 1) * factorRow * 2 + FLOAT_REPEAT_SIZE * 2;
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK : B32_DATA_NUM_PER_BLOCK;
    uint32_t sreg = srcK * srcM;
    uint32_t sreg1 = srcM * blockStride;
    uint64_t mask[2] = {0, 0};
    CreateSpecialFormatMask(mask[0], originK, factorRow, srcK);
    SetVectorMask<uint32_t>(mask[1], mask[0]);

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* tmpUb0 = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK);
    __local_mem__ float* tmpUb0Tmp0 = tmpUb0;
    __local_mem__ float* tmpUb0Tmp1 = tmpUb0;
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK + offset);
    __local_mem__ float* tmpUb1 = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * srcK + offset + offset1);

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
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

        for (uint16_t i = 0; i < factor; ++i) {
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK * factorRow, pregFull);

            MicroAPI::ReduceMaxWithDataBlock(maxVreg, srcVreg, pregCnt);

            Duplicate(tmpVreg, 0);
            MicroAPI::DeInterleave(maxVreg, tmpVreg, maxVreg, tmpVreg);
            MicroAPI::Max(maxVreg, maxVreg, tmpVreg, pregFull);
            if constexpr (sizeof(T2) == sizeof(float)) {
                MicroAPI::DataCopyUnAlign(tmpUb0Tmp0, maxVreg, ureg0, factorRow);
            }
            MicroAPI::Interleave(maxVreg, tmpVreg, maxVreg, maxVreg);
            MicroAPI::DataCopy(tmpUb1 + i * 2 * factorRow, maxVreg, pregOneBlk);
        }
        if constexpr (sizeof(T2) == sizeof(float)) {
            MicroAPI::DataCopyUnAlignPost(tmpUb0Tmp0, ureg0, 0);
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        if constexpr (sizeof(T2) == sizeof(float)) {
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
            if constexpr (sizeof(T2) == sizeof(half)) {
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
            if constexpr (sizeof(T2) == sizeof(float)) {
                MicroAPI::DataCopyUnAlign(tmpUb0Tmp1, sumVreg, ureg1, factorRow);
            }
            MicroAPI::Interleave(sumVreg, tmpVreg, sumVreg, sumVreg);
            MicroAPI::DataCopy(tmpUb + i * 2 * factorRow, sumVreg, pregOneBlk);
        }
        if constexpr (sizeof(T2) == sizeof(float)) {
            MicroAPI::DataCopyUnAlignPost(tmpUb0Tmp1, ureg1, 0);
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

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
}

template <typename T1, typename T2, bool isFlashV2 = false, bool isLog = false>
__aicore__ inline void SingleSoftMaxGenericNDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    uint16_t srcM = originalSrcShape.m;
    uint16_t srcK = tiling.srcK;
    uint16_t originK = originalSrcShape.k;
    constexpr uint16_t blockStride = IsSameType<T2, half>::value ? B16_DATA_NUM_PER_BLOCK : B32_DATA_NUM_PER_BLOCK;
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    __local_mem__ T1* dstUb = (__local_mem__ T1*)dst.GetPhyAddr();
    __local_mem__ T2* sumUb = (__local_mem__ T2*)sumTensor.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ T1* srcUb = (__local_mem__ T1*)src.GetPhyAddr();
    __local_mem__ float* tmpUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(srcM * blockStride);

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
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
            uint32_t sreg = originK;
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            LoadIfNeedCast<T1>(srcVreg, srcUb + i * srcK, pregFull);

            MicroAPI::ReduceMax(maxVreg, srcVreg, pregCnt);
            Duplicate(maxVreg, maxVreg, pregOneBlk);
            StoreIfNeedCast<T2>(maxUb + i * blockStride, maxVreg, pregOneBlk);

            Duplicate(maxVreg, maxVreg, pregFull);
            MicroAPI::Sub(dstVreg, srcVreg, maxVreg, pregCnt);
            MicroAPI::Exp(tmpVreg, dstVreg, pregCnt);
            if constexpr (!isFlashV2) {
                MicroAPI::DataCopy(workUb + i * srcK, tmpVreg, pregCnt);
            } else {
                StoreIfNeedCast<T1>(dstUb + i * srcK, tmpVreg, pregCnt);
            }
            MicroAPI::ReduceSum(sumVreg, tmpVreg, pregCnt);
            Duplicate(sumVreg, sumVreg, pregOneBlk);
            StoreIfNeedCast<T2>(sumUb + i * blockStride, sumVreg, pregOneBlk);
            if constexpr (!isFlashV2 && sizeof(T2) == sizeof(half)) {
                MicroAPI::DataCopy(tmpUb + i * blockStride, sumVreg, pregOneBlk);
            }
        }

        if constexpr (!isFlashV2) {
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < srcM; ++i) {
                uint32_t sreg = originK;
                pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
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
}

template <typename T1, typename T2, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxNDImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const LastAxisShapeND& originalSrcShape, const SoftMaxTiling& tiling)
{
    static_assert((SupportType<T1, float>() && SupportType<T2, float>())
                      || (SupportType<T1, half>() && SupportType<T2, half>())
                      || (SupportType<T1, half>() && SupportType<T2, float>()),
        "SoftMax api only support half/float on current device");
    if constexpr (isBasicBlock) {
        SoftMaxGenericNDImpl<T1, T2, false>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
    } else {
        if (tiling.srcK == B32_DATA_NUM_PER_BLOCK && IsSameType<T1, float>::value) {
            SingleSoftMaxGenericNDForBlkImpl<T1, T2, false>(
                dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        } else if (tiling.srcK == B32_DATA_NUM_PER_BLOCK * 2) {
            SingleSoftMaxGenericNDAlignedWithBlkImpl<T1, T2, false>(
                dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        } else if (originalSrcShape.k <= FLOAT_REPEAT_SIZE) {
            SingleSoftMaxGenericNDImpl<T1, T2, false>(
                dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        } else if (originalSrcShape.k % FLOAT_REPEAT_SIZE != 0) {
            SoftMaxGenericNDWithTailImpl<T1, T2, false>(
                dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        } else {
            SoftMaxGenericNDImpl<T1, T2, false>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
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
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

    __local_mem__ T* dstUb = (__local_mem__ T*)dst.GetPhyAddr();
    __local_mem__ T* srcUb = (__local_mem__ T*)src.GetPhyAddr();
    __local_mem__ float* sumUb = (__local_mem__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* tmpUb = sumUb;
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr(offset);
    if constexpr (isReuseSource && sizeof(T) == sizeof(float)) {
        workUb = (__local_mem__ float*)src.GetPhyAddr();
    }

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
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
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                sumVreg, tmpUb, 1);
            uint32_t sreg = originK;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
                MicroAPI::DataCopy(tmpVreg, workUb + i * srcK + j * FLOAT_REPEAT_SIZE);
                MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregCnt);
                StoreIfNeedCast<T>(dstUb + i * srcK + j * FLOAT_REPEAT_SIZE, dstVreg, pregCnt);
            }
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
    NotNumUnion notNum;
    notNum.i = F32_NEG_INF;

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

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
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
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                (maxUb + i + halfM), maxVreg1, pregOne);
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
    uint32_t sreg = srcK * srcM;
    uint64_t mask[2] = {0, 0};
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

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
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
    uint32_t sreg = srcK * srcM;
    uint64_t mask[2] = {0, 0};
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

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
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

#pragma no_simd_vf_fusion
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> srcVreg;
        MicroAPI::RegTensor<float> maxVreg;
        MicroAPI::RegTensor<float> sumVreg;
        MicroAPI::RegTensor<float> tmpVreg;
        MicroAPI::RegTensor<float> dstVreg;
        MicroAPI::UnalignReg ureg0;

        for (uint16_t i = 0; i < srcM; ++i) {
            uint32_t sreg = originK;
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
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
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                sumVreg, tmpUb, 1);
            uint32_t sreg = originK;
            pregCnt = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy(tmpVreg, workUb + i * srcK);
            MicroAPI::Div(dstVreg, tmpVreg, sumVreg, pregCnt);
            StoreIfNeedCast<T>(dstUb + i * srcK, dstVreg, pregCnt);
        }
    }
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
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SOFTMAX_IMPL_H
