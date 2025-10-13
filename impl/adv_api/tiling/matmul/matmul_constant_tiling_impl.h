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
 * \file matmul_constant_tiling_impl.h
 * \brief
 */
#ifndef TILING_MATMUL_MATMUL_CONSTANT_TILING_IMPL_H
#define TILING_MATMUL_MATMUL_CONSTANT_TILING_IMPL_H

#include "matmul_constant_tiling_utils.h"

namespace AscendC {
template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr L1Status GetL1StatusBL1FullLoad(const MatmulConfig &mmCFG, int32_t l1Size)
{
    int32_t reduceC0Size = GetReduceC0Size<typename A_TYPE::T>();
    int32_t k = CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceC0Size);
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    int32_t kBL1 = Align<int32_t>(k, kL0);
    int32_t maxNBL1 = GetMaxNBL1(mmCFG);
    L1Status l1Status {kL0, kBL1, 1, maxNBL1, 1, 1, 0};
    // if mmCFG use OUTER_PRODUCT, stepM > 1 in ORDER_M and stepN > 1 in ORDER_N
    if (((mmCFG.doNorm && A_TYPE::layout == LayoutMode::NONE) || mmCFG.doMultiDataLoad) &&
        (mmCFG.scheduleType == ScheduleType::OUTER_PRODUCT && mmCFG.iterateOrder == IterateOrder::ORDER_N)) {
        l1Status.mAL1 = Impl::OUTER_STEP;
    }
    if (GetL1Size<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG) > l1Size) {
        return {0, 0, 0, 0, 0, 0, INT32_MAX};
    }
    int32_t kbAlignValue = GetKBAlignValue<A_TYPE, B_TYPE>();
    int32_t n = CeilNoLog<int32_t>(mmCFG.singleCoreN, Impl::HW_C0);
    // be consistent with initbuffer
    int32_t bL1Size = MaxValue<int32_t>(maxNBL1 * mmCFG.basicN, n * Impl::HW_C0) *
        CeilNoLog<int32_t>(k, kL0) * Align(mmCFG.basicK, static_cast<uint32_t>(kbAlignValue * reduceC0Size)) *
        GetBitSize<typename B_TYPE::T>() / ONE_BYTE_BIT_SIZE;
    int32_t aL1Size = PhyPosIsL1(B_TYPE::pos) ? l1Size : l1Size - bL1Size;
    l1Status.dbAL1 = Impl::DB_ON;
    l1Status.dbAL1 = GetL1Size<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG) > l1Size ?
        Impl::DB_OFF : l1Status.dbAL1;
    int32_t biasSize = GetBiasL1Size<BIAS_TYPE>(l1Status, mmCFG);
    int32_t dequantSize = GetDeQuantSize(l1Status, mmCFG);
    int32_t kaAlignValue = GetKAAlignValue<A_TYPE>();
    l1Status.kAL1 = MinValue<int32_t>(CalcL1MaxLen<A_TYPE, B_TYPE, BIAS_TYPE>((aL1Size - biasSize - dequantSize),
        l1Status, mmCFG, kaAlignValue, L1TilingType::KAL1_16), k);
    int32_t aL1Times = MinValue<int32_t>(l1Status.kAL1 / kL0, GetMaxKAL1<A_TYPE>(mmCFG));
    aL1Times = GetNearestFactor(CeilNoLog<int32_t>(k, kL0), aL1Times);
    l1Status.kAL1 = aL1Times * kL0;
    if (l1Status.kAL1 == k) {
        l1Status.mAL1 = MinValue<int32_t>(CalcL1MaxLen<A_TYPE, B_TYPE, BIAS_TYPE>(aL1Size - biasSize, l1Status, mmCFG,
            kaAlignValue, L1TilingType::M_AL1), GetMaxMAL1(mmCFG));
        int32_t mRepeat = CeilNoLog<int32_t>(mmCFG.singleCoreM, mmCFG.basicM);
        l1Status.mAL1 = GetNearestFactor(mRepeat, l1Status.mAL1);
        if (l1Status.mAL1 * mmCFG.basicM == mmCFG.singleCoreM) {
            l1Status.dbAL1 = Impl::DB_OFF;
        }
    }
    bool invalidL1Status = (l1Status.mAL1 == 0 || l1Status.kAL1 == 0);
    int32_t nRepeat = CeilNoLog<int32_t>(mmCFG.singleCoreN, mmCFG.basicN);
    int32_t possibleNRepeat = (l1Status.kAL1 == k) ? 1 : nRepeat;
    int32_t m = CeilNoLog<int32_t>(mmCFG.singleCoreM, Impl::HW_C0);
    l1Status.loadSize = invalidL1Status ? INT32_MAX : (PhyPosIsL1(B_TYPE::pos) ? 0 : n) + possibleNRepeat * m;
    return l1Status;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr L1Status GetL1StatusMFirst(const L1Status &l1Status, const MatmulConfig &mmCFG, int32_t l1Size)
{
    int32_t nRepeat = CeilNoLog<int32_t>(mmCFG.singleCoreN, mmCFG.basicN);
    int32_t mRepeat = CeilNoLog<int32_t>(mmCFG.singleCoreM, mmCFG.basicM);
    L1Status l1MFirst {l1Status};
    int32_t bL1Size = GetBL1Size<A_TYPE, B_TYPE>(l1MFirst, mmCFG);
    int32_t aL1Size = l1Size - bL1Size;
    int32_t kaAlignValue = GetKAAlignValue<A_TYPE>();
    int32_t kbAlignValue = GetKBAlignValue<A_TYPE, B_TYPE>();
    int32_t biasSize = GetBiasL1Size<BIAS_TYPE>(l1MFirst, mmCFG);
    int32_t dequantSize = GetDeQuantSize(l1MFirst, mmCFG);
    l1MFirst.mAL1 = MaxValue<int32_t>(MinValue<int32_t>(CalcL1MaxLen<A_TYPE, B_TYPE, BIAS_TYPE>(aL1Size - biasSize - dequantSize,
        l1MFirst, mmCFG, kaAlignValue, L1TilingType::M_AL1), GetMaxMAL1(mmCFG), mRepeat), 1);
    l1MFirst.mAL1 = GetNearestFactor(mRepeat, l1MFirst.mAL1);
    aL1Size = GetAL1Size<A_TYPE>(l1MFirst, mmCFG);
    bL1Size = l1Size - aL1Size;
    l1MFirst.nBL1 = MaxValue<int32_t>(MinValue<int32_t>(CalcL1MaxLen<A_TYPE, B_TYPE, BIAS_TYPE>(bL1Size - biasSize - dequantSize,
        l1MFirst, mmCFG, kbAlignValue, L1TilingType::N_BL1), GetMaxNBL1(mmCFG), nRepeat), 1);
    l1MFirst.nBL1 = GetNearestFactor(mRepeat, l1MFirst.nBL1);
    int32_t mL0 = GetML0(mmCFG);
    int32_t m = CeilNoLog<int32_t>(mmCFG.singleCoreM, Impl::HW_C0);
    int32_t n = CeilNoLog<int32_t>(mmCFG.singleCoreN, Impl::HW_C0);
    l1MFirst.loadSize = m + n * CeilNoLog<int32_t>(m, l1MFirst.mAL1 * mL0);
    return l1MFirst;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr L1Status GetL1StatusNFirst(const L1Status &l1Status, const MatmulConfig &mmCFG, int32_t l1Size)
{
    int32_t nRepeat = CeilNoLog<int32_t>(mmCFG.singleCoreN, mmCFG.basicN);
    int32_t mRepeat = CeilNoLog<int32_t>(mmCFG.singleCoreM, mmCFG.basicM);
    L1Status l1NFirst {l1Status};
    int32_t aL1Size = GetAL1Size<A_TYPE>(l1NFirst, mmCFG);
    int32_t bL1Size = l1Size - aL1Size;
    int32_t kbAlignValue = GetKBAlignValue<A_TYPE, B_TYPE>();
    int32_t biasSize = GetBiasL1Size<BIAS_TYPE>(l1NFirst, mmCFG);
    int32_t dequantSize = GetDeQuantSize(l1NFirst, mmCFG);
    l1NFirst.nBL1 = MaxValue<int32_t>(MinValue<int32_t>(CalcL1MaxLen<A_TYPE, B_TYPE, BIAS_TYPE>(bL1Size - biasSize - dequantSize,
        l1Status, mmCFG, kbAlignValue, L1TilingType::N_BL1), GetMaxNBL1(mmCFG), nRepeat), 1);
    l1NFirst.nBL1 = GetNearestFactor(nRepeat, l1NFirst.nBL1);
    bL1Size = GetBL1Size<A_TYPE, B_TYPE>(l1NFirst, mmCFG);
    aL1Size = l1Size - bL1Size;
    int32_t kaAlignValue = GetKAAlignValue<A_TYPE>();
    l1NFirst.mAL1 = MaxValue<int32_t>(MinValue<int32_t>(CalcL1MaxLen<A_TYPE, B_TYPE, BIAS_TYPE>(aL1Size - biasSize - dequantSize,
        l1NFirst, mmCFG, kaAlignValue, L1TilingType::M_AL1), GetMaxMAL1(mmCFG), mRepeat), 1);
    l1NFirst.mAL1 = GetNearestFactor(mRepeat, l1NFirst.mAL1);
    l1NFirst.nBL1 = GetNearestFactor(mRepeat, l1NFirst.nBL1);
    int32_t nL0 = GetNL0(mmCFG);
    int32_t m = CeilNoLog<int32_t>(mmCFG.singleCoreM, Impl::HW_C0);
    int32_t n = CeilNoLog<int32_t>(mmCFG.singleCoreN, Impl::HW_C0);
    l1NFirst.loadSize = n + m * CeilNoLog<int32_t>(n, l1NFirst.nBL1 * nL0);
    return l1NFirst;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr L1Status GetL1DbNeitherFullLoad(const MatmulConfig &mmCFG, int32_t l1Size)
{
    using SrcAT = typename A_TYPE::T;
    int32_t reduceC0Size = GetReduceC0Size<SrcAT>();
    int32_t k = CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceC0Size);
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    L1Status l1Status {kL0, Impl::DB_ON, 1, 1, Impl::DB_ON, Impl::DB_ON, 0};
    // if mmCFG use OUTER_PRODUCT, stepM > 1 in ORDER_M and stepN > 1 in ORDER_N
    if (((mmCFG.doNorm && A_TYPE::layout == LayoutMode::NONE) || mmCFG.doMultiDataLoad) &&
        mmCFG.scheduleType == ScheduleType::OUTER_PRODUCT) {
        if (mmCFG.iterateOrder == IterateOrder::ORDER_M) {
            l1Status.nBL1 = Impl::OUTER_STEP;
        } else if (mmCFG.iterateOrder == IterateOrder::ORDER_N) {
            l1Status.mAL1 = Impl::OUTER_STEP;
        }
    }
    if (GetL1Size<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG) > l1Size) {
        l1Status.dbBL1 = Impl::DB_OFF;
        if (GetL1Size<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG) > l1Size) {
            l1Status.dbAL1 = Impl::DB_OFF;
        }
    }
    l1Status.kBL1 = k;
    int32_t m = CeilNoLog<int32_t>(mmCFG.singleCoreM, Impl::HW_C0);
    int32_t mL0 = GetML0(mmCFG);
    bool bothDoubleBuffer = m != mL0 && mmCFG.singleCoreK > mmCFG.basicK &&
        GetL1Size<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG) > l1Size;
    l1Status.kBL1 = kL0;
    if (bothDoubleBuffer) {
        l1Status.dbAL1 = Impl::DB_ON;
        l1Status.dbBL1 = Impl::DB_ON;
        if (GetL1Size<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG) > l1Size) {
            l1Status.dbBL1 = Impl::DB_OFF;
            if (GetL1Size<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG) > l1Size) {
                l1Status.dbAL1 = Impl::DB_OFF;
            }
        }
    }
    return l1Status;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr L1Status GetKL1NeitherFullLoadForNZ(const L1Status &l1Nz,
    const MatmulConfig &mmCFG, int32_t l1Size, int32_t biasSize, int32_t dequantSize)
{
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    L1Status l1Status {l1Nz};
    int32_t maxMAL1 = GetMaxMAL1(mmCFG);
    int32_t reduceC0Size = GetReduceC0Size<SrcAT>();
    int32_t k = CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceC0Size);
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    int32_t kaAlignValue = GetKAAlignValue<A_TYPE>();
    int32_t kbAlignValue = GetKBAlignValue<A_TYPE, B_TYPE>();
    int32_t aL1Times = CeilNoLog<int32_t>(k, kL0);
    if (GetL1Size<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG) <= l1Size) {
        int32_t bL1Size = GetBL1Size<A_TYPE, B_TYPE>(l1Status, mmCFG);
        int32_t aL1Size = l1Size - bL1Size;
        l1Status.kAL1 = MinValue<int32_t>(CalcL1MaxLen<A_TYPE, B_TYPE, BIAS_TYPE>((aL1Size - biasSize - dequantSize), l1Status,
            mmCFG, kaAlignValue, L1TilingType::KAL1_16), k);
        aL1Times = MaxValue<int32_t>(MinValue<int32_t>(l1Status.kAL1 / kL0, maxMAL1), 1);
        aL1Times = GetNearestFactor(CeilNoLog<int32_t>(k, kL0), aL1Times);
        l1Status.kAL1 = aL1Times * kL0;
    } else {
        // when NeitherFullLoadMN change the nBL1 and mAL1
        int32_t perK = MinValue<int32_t>((l1Size - biasSize - dequantSize) /
            (mmCFG.basicM * Impl::C0_BYTE_SIZE * l1Status.dbAL1 +
            mmCFG.basicN * Impl::C0_BYTE_SIZE * l1Status.dbBL1) /
            kL0 * kL0, k);
        const int32_t aAlignedPerK = Align<int32_t>(perK, kaAlignValue);
        const int32_t bAlignedPerK = Align<int32_t>(perK, kbAlignValue);
        int32_t aL1 = l1Status.mAL1 * mmCFG.basicM * aAlignedPerK * l1Status.dbAL1 *
            GetTypeSize<SrcAT>();
        int32_t bL1 = l1Status.nBL1 * mmCFG.basicN * bAlignedPerK * l1Status.dbBL1 *
            GetTypeSize<SrcBT>();
        if (IsSameTypeV<SrcAT, float> && (aL1 + bL1 + dequantSize + biasSize) > l1Size) {
            perK -= 1;
        }
        int32_t perTimes = MinValue<int32_t>(perK / kL0, MaxValue<int32_t>(GetMaxKAL1<A_TYPE>(mmCFG),
            GetMaxKBL1<A_TYPE>(mmCFG)));
        perTimes = GetNearestFactor(aL1Times, perTimes);
        perTimes = MinValue<int32_t>(perTimes, aL1Times);
        perK = perTimes * kL0;
        l1Status.kAL1 = perK;
        l1Status.kBL1 = perK;
    }
    return l1Status;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr L1Status GetKL1NeitherFullLoad(const L1Status &l1Db,
    const MatmulConfig &mmCFG, int32_t l1Size, int32_t biasSize, int32_t dequantSize)
{
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    L1Status l1Status {l1Db};
    int32_t reduceC0Size = GetReduceC0Size<SrcAT>();
    int32_t k = CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceC0Size);
    int32_t kL0 = GetKL0<A_TYPE>(mmCFG);
    int32_t kMaxAxis = GetKMaxAxis<A_TYPE, B_TYPE>(mmCFG);
    int32_t aL1Times = CeilNoLog<int32_t>(k, kL0);
    if (kMaxAxis == 0) {
        l1Status.kBL1 = k;
        l1Status = GetKL1NeitherFullLoadForNZ<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status,
            mmCFG, l1Size, biasSize, dequantSize);
    } else if (kMaxAxis == 1) {
        // first get k_al1, second get k_bl1
        l1Status.kBL1 = kL0;
        int32_t bL1Size = GetBL1Size<A_TYPE, B_TYPE>(l1Status, mmCFG);
        int32_t aL1Size = l1Size - bL1Size;
        l1Status.kAL1 = MinValue<int32_t>((aL1Size - biasSize - dequantSize) /
            (l1Status.mAL1 * mmCFG.basicM * l1Status.dbAL1 * Impl::C0_BYTE_SIZE), k);
        aL1Times = MaxValue<int32_t>(l1Status.kAL1 / kL0, 1);
        aL1Times = GetNearestFactor(CeilNoLog<int32_t>(k, kL0), aL1Times);
        l1Status.kAL1 = aL1Times * kL0;
        aL1Size = l1Status.kAL1 * l1Status.mAL1 * mmCFG.basicM * Impl::C0_BYTE_SIZE * l1Status.dbAL1;
        bL1Size = l1Size - aL1Size;
        l1Status.kBL1 = MinValue<int32_t>((bL1Size - dequantSize - biasSize) / (l1Status.nBL1 * mmCFG.basicN *
            l1Status.dbBL1 * mmCFG.basicK * kL0 * GetBitSize<SrcBT>() / ONE_BYTE_BIT_SIZE), k);
        int32_t bL1Times = MaxValue<int32_t>(MinValue<int32_t>(l1Status.kBL1 / kL0, GetMaxKBL1<A_TYPE>(mmCFG)), 1);
        bL1Times = GetNearestFactor(CeilNoLog<int32_t>(k, kL0), bL1Times);
        l1Status.kBL1 = bL1Times * kL0;
    } else if (kMaxAxis == 2) {
        // first get k_bl1, second get k_al1
        l1Status.kAL1 = kL0;
        int32_t aL1Size = GetAL1Size<A_TYPE>(l1Status, mmCFG);
        int32_t bL1Size = l1Size - aL1Size;
        l1Status.kBL1 = MinValue<int32_t>((bL1Size - biasSize - dequantSize) /
            (l1Status.nBL1 * mmCFG.basicN * l1Status.dbBL1 * Impl::C0_BYTE_SIZE), k);
        int32_t bL1Times = MaxValue<int32_t>(l1Status.kBL1 / kL0, 1);
        bL1Times = GetNearestFactor(aL1Times, bL1Times);
        l1Status.kBL1 = bL1Times * kL0;
        bL1Size = l1Status.kBL1 * l1Status.nBL1 * mmCFG.basicN * Impl::C0_BYTE_SIZE * l1Status.dbBL1;
        aL1Size = l1Size - bL1Size;
        l1Status.kAL1 = MinValue<int32_t>((aL1Size - dequantSize - biasSize) / (l1Status.mAL1 * mmCFG.basicM *
            l1Status.dbAL1 * mmCFG.basicK * kL0 * GetBitSize<SrcAT>() / ONE_BYTE_BIT_SIZE), k);
        aL1Times = MaxValue<int32_t>(MinValue<int32_t>(l1Status.kAL1 / kL0, GetMaxKAL1<A_TYPE>(mmCFG)), 1);
        aL1Times = GetNearestFactor(CeilNoLog<int32_t>(k, kL0), aL1Times);
        l1Status.kAL1 = aL1Times * kL0;
    }
    return l1Status;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr L1Status GetL1StatusNeitherFullLoad(const MatmulConfig &mmCFG, int32_t l1Size)
{
    using SrcAT = typename A_TYPE::T;
    int32_t reduceC0Size = GetReduceC0Size<SrcAT>();
    int32_t k = CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceC0Size);
    L1Status l1Status = GetL1DbNeitherFullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Size);
    int32_t biasSize = GetBiasL1Size<BIAS_TYPE>(l1Status, mmCFG);
    int32_t dequantSize = GetDeQuantSize(l1Status, mmCFG);
    l1Status = GetKL1NeitherFullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG, l1Size,
        biasSize, dequantSize);
    if (l1Status.kAL1 > l1Status.kBL1 && l1Status.kAL1 % l1Status.kBL1 != 0) {
        while (l1Status.kAL1 % l1Status.kBL1 != 0 || (l1Status.kAL1 != l1Status.kBL1 && k % l1Status.kAL1 != 0)) {
            l1Status.kAL1 -= 1;
        }
    }
    if (l1Status.kAL1 < l1Status.kBL1 && l1Status.kBL1 % l1Status.kAL1 != 0) {
        while (l1Status.kBL1 % l1Status.kAL1 != 0 || (l1Status.kAL1 != l1Status.kBL1 && k % l1Status.kBL1 != 0)) {
            l1Status.kBL1 -= 1;
        }
    }
    auto l1MFirst = GetL1StatusMFirst<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG, l1Size);
    auto l1NFirst = GetL1StatusNFirst<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(l1Status, mmCFG, l1Size);
    if (l1Status.kAL1 >= k && l1Status.kBL1 >= k) {
        l1Status = l1NFirst.loadSize > l1MFirst.loadSize ? l1MFirst : l1NFirst;
    }
    if (l1Status.kAL1 >= k && l1Status.kBL1 < k) {
        l1Status.nBL1 = 1;
    }
    if (l1Status.kAL1 < k && l1Status.kBL1 >= k) {
        l1Status.mAL1 = 1;
    }
    if (l1Status.kAL1 < k && l1Status.kBL1 < k) {
        l1Status.mAL1 = 1;
        l1Status.nBL1 = 1;
        int32_t m = CeilNoLog<int32_t>(mmCFG.singleCoreM, Impl::HW_C0);
        int32_t n = CeilNoLog<int32_t>(mmCFG.singleCoreN, Impl::HW_C0);
        int32_t nL0 = GetNL0(mmCFG);
        l1Status.loadSize = m * CeilNoLog<int32_t>(n, nL0) + n * CeilNoLog<int32_t>(m, nL0);
    }
    return l1Status;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr L1Status GetL1Factor(const MatmulConfig &mmCFG, int32_t l1Size)
{
    L1Status l1Status = GetL1StatusBothFullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Size);
    bool bothND = A_TYPE::format == CubeFormat::ND && B_TYPE::format == CubeFormat::ND;
    int32_t reduceSize = GetReduceC0Size<typename A_TYPE::T>();
    int32_t kAL1Factor = (l1Status.kAL1 > 0) ? CeilNoLog<int32_t>(CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceSize), l1Status.kAL1) : 1;
    int32_t kBL1Factor = (l1Status.kBL1 > 0) ? CeilNoLog<int32_t>(CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceSize), l1Status.kBL1) : 1;
    bool bothFullLoad = bothND ? (l1Status.loadSize != INT32_MAX && kAL1Factor == 1 && kBL1Factor == 1) :
        (l1Status.loadSize != INT32_MAX);
    if (bothFullLoad) {
        return l1Status;
    }
    L1Status aL1FullLoad {0, 0, 0, 0, 0, 0, INT32_MAX};
    if constexpr (!PhyPosIsL1(B_TYPE::pos)) {
        aL1FullLoad = GetL1StatusAL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Size);
    }
    L1Status bL1FullLoad {0, 0, 0, 0, 0, 0, INT32_MAX};
    if constexpr (!PhyPosIsL1(A_TYPE::pos)) {
        bL1FullLoad = GetL1StatusBL1FullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Size);
    }
    L1Status neitherFullLoad {0, 0, 0, 0, 0, 0, INT32_MAX};
    if constexpr (!PhyPosIsL1(A_TYPE::pos) && !PhyPosIsL1(B_TYPE::pos)) {
        neitherFullLoad = GetL1StatusNeitherFullLoad<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG, l1Size);
    }
    kAL1Factor = (aL1FullLoad.loadSize != INT32_MAX) ?
        CeilNoLog<int32_t>(CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceSize), aL1FullLoad.kAL1) : 1;
    bool isAL1FullLoad = bothND ?
        (aL1FullLoad.loadSize != INT32_MAX && kAL1Factor == 1) : (aL1FullLoad.loadSize != INT32_MAX);
    if (isAL1FullLoad) {
        return aL1FullLoad;
    }
    kBL1Factor = (bL1FullLoad.loadSize != INT32_MAX) ?
        CeilNoLog<int32_t>(CeilNoLog<int32_t>(mmCFG.singleCoreK, reduceSize), bL1FullLoad.kBL1) : 1;
    bool isBL1FullLoad = bothND ?
        (bL1FullLoad.loadSize != INT32_MAX && kBL1Factor == 1) : (bL1FullLoad.loadSize != INT32_MAX);
    if (isBL1FullLoad) {
        return bL1FullLoad;
    }
    return neitherFullLoad;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr bool CalcAL1FullLoadTiling(int32_t l1Size, MatmulApiStaticTiling &tiling)
{
    if (tiling.singleCoreM > Impl::MIN_MN_SIZE || tiling.singleCoreN > Impl::MIN_MN_SIZE) {
        return false;
    }
    l1Size = tiling.isBias ? (l1Size - Impl::BT_SIZE / Impl::BITS_PER_BYTE) : l1Size;
    int32_t baseKaAlign = Align<int32_t>(tiling.baseK,
        GetKAAlignValue<A_TYPE>() * GetReduceC0Size<typename A_TYPE::T>());
    int32_t baseKbAlign = Align<int32_t>(tiling.baseK,
        GetKBAlignValue<A_TYPE, B_TYPE>() * GetReduceC0Size<typename B_TYPE::T>());
    int32_t baseA = tiling.baseM * baseKaAlign * GetBitSize<typename A_TYPE::T>() / Impl::BITS_PER_BYTE;
    int32_t baseB = tiling.baseN * baseKbAlign * GetBitSize<typename B_TYPE::T>() / Impl::BITS_PER_BYTE;
    int32_t depthA1 = (l1Size - Impl::DB_ON * baseB) / baseA;
    if (depthA1 * tiling.baseM * baseKaAlign < tiling.singleCoreM * tiling.singleCoreK) {
        return false;
    }
    depthA1 = MaxValue(tiling.singleCoreM, tiling.baseM) * MaxValue(tiling.singleCoreK, baseKaAlign) /
        tiling.baseM / baseKaAlign;
    tiling.depthA1 = depthA1;
    tiling.stepKa = depthA1;

    int32_t stepKb = (l1Size - depthA1 * baseA) / baseB / Impl::DB_ON;
    if (stepKb * Impl::DB_ON * baseB > tiling.singleCoreK * tiling.singleCoreN) {
        stepKb = MaxValue(tiling.singleCoreK, baseKbAlign) * MaxValue(tiling.singleCoreN, tiling.baseN) /
            tiling.baseN / baseKbAlign / Impl::DB_ON;
    }
    if (stepKb < 1) {
        tiling.depthB1 = 1;
        tiling.stepKb = 1;
        return true;
    }
    while (tiling.stepKa % stepKb != 0 && stepKb % tiling.stepKa != 0 && stepKb > 1) {
        stepKb--;
    }
    tiling.depthB1 = stepKb * Impl::DB_ON;
    tiling.stepKb = stepKb;
    return true;
}

template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
__aicore__ constexpr MxScaleStatus GetMxScaleFactor(const MatmulApiStaticTiling &tiling, int32_t l1Size)
{
    using SrcAT = typename A_TYPE::T;
    using SrcBT = typename B_TYPE::T;
    using SrcBiasT = typename BIAS_TYPE::T;
    MxScaleStatus mxScaleFactor{ 1, 1, 0 };

    int dataUsedL1Size = tiling.depthA1 * tiling.baseM * tiling.baseK * GetBitSize<SrcAT>() / ONE_BYTE_BIT_SIZE +
        tiling.depthB1 * tiling.baseN * tiling.baseK * GetBitSize<SrcBT>() / ONE_BYTE_BIT_SIZE;
    int bias = tiling.isBias ? 1 : 0;
    int biasUsedL1Size = bias * tiling.baseN * GetBitSize<SrcBiasT>() / ONE_BYTE_BIT_SIZE;
    int remainedL1Size = l1Size - (dataUsedL1Size + biasUsedL1Size);
    int kStep = CeilNoLog<int32_t>(tiling.singleCoreK, tiling.baseK);
    int maxScaleFactorKa = CeilNoLog<int32_t>(kStep, tiling.stepKa);
    int maxScaleFactorKb = CeilNoLog<int32_t>(kStep, tiling.stepKb);

    mxScaleFactor.scaleFactorKa =
    remainedL1Size / Impl::MX_L1_BUFFER_NUM / (tiling.stepKa * tiling.baseM * tiling.baseK / Impl::SCALE_K_SIZE);
    mxScaleFactor.scaleFactorKb =
    remainedL1Size / Impl::MX_L1_BUFFER_NUM / (tiling.stepKb * tiling.baseN * tiling.baseK / Impl::SCALE_K_SIZE);
    mxScaleFactor.scaleFactorKa =
    mxScaleFactor.scaleFactorKa > maxScaleFactorKa ? maxScaleFactorKa : mxScaleFactor.scaleFactorKa;
    mxScaleFactor.scaleFactorKb =
    mxScaleFactor.scaleFactorKb > maxScaleFactorKb ? maxScaleFactorKb : mxScaleFactor.scaleFactorKb;

    // scaleFactor is in range of [1, 127]
    mxScaleFactor.scaleFactorKa = mxScaleFactor.scaleFactorKa >= 1 ? mxScaleFactor.scaleFactorKa : 1;
    mxScaleFactor.scaleFactorKb = mxScaleFactor.scaleFactorKb >= 1 ? mxScaleFactor.scaleFactorKb : 1;
    mxScaleFactor.scaleFactorKa =
    mxScaleFactor.scaleFactorKa <= Impl::SCALE_FACTOR_MAX_VALUE ? mxScaleFactor.scaleFactorKa : Impl::SCALE_FACTOR_MAX_VALUE;
    mxScaleFactor.scaleFactorKb =
    mxScaleFactor.scaleFactorKb <= Impl::SCALE_FACTOR_MAX_VALUE ? mxScaleFactor.scaleFactorKb : Impl::SCALE_FACTOR_MAX_VALUE;

    if constexpr ((A_TYPE::format == CubeFormat::ND && A_TYPE::isTrans == true && A_TYPE::scalePosition == TPosition::TSCM) &&
        (B_TYPE::format == CubeFormat::ND && B_TYPE::isTrans == false && B_TYPE::scalePosition == TPosition::TSCM)) {
        mxScaleFactor.scaleFactorKa = static_cast<uint8_t>(1);
        mxScaleFactor.scaleFactorKb = static_cast<uint8_t>(1);
    } else {
        if constexpr (A_TYPE::scalePosition == TPosition::TSCM) {
            mxScaleFactor.scaleFactorKa = static_cast<uint8_t>(1);
        }

        if constexpr (B_TYPE::scalePosition == TPosition::TSCM) {
            mxScaleFactor.scaleFactorKb = static_cast<uint8_t>(1);
        }
    }

    // 8bit: 0~6bit:scaleFactor, 7bit(reserved):double buffer flag
    mxScaleFactor.scaleFactorKa = mxScaleFactor.scaleFactorKa & 0x7f;
    mxScaleFactor.scaleFactorKb = mxScaleFactor.scaleFactorKb & 0x7f;
    mxScaleFactor.mxTypePara = static_cast<int32_t>(static_cast<uint32_t>(mxScaleFactor.mxTypePara) | mxScaleFactor.scaleFactorKa);
    mxScaleFactor.mxTypePara = static_cast<int32_t>(static_cast<uint32_t>(mxScaleFactor.mxTypePara) | (mxScaleFactor.scaleFactorKb << 8U));
    return mxScaleFactor;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr void GetMxMatmulApiTiling(MatmulApiStaticTiling &tiling, int32_t l1Size)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    if constexpr (HasScalePosition<A_TYPE>::value || HasScalePosition<B_TYPE>::value) {
        MxScaleStatus mxScaleFactor = GetMxScaleFactor<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tiling, l1Size);
        tiling.mxTypePara = mxScaleFactor.mxTypePara;
        // For MxMatmul : usedL1Size = tensorASize + scaleASize + tensorBSize + scaleBSize + biasSize
        int32_t needUsedL1Size = tiling.baseM * tiling.baseK * tiling.depthA1 * GetBitSize<typename A_TYPE::T>() /
            ONE_BYTE_BIT_SIZE + tiling.baseM * tiling.baseK / Impl::SCALE_K_SIZE * tiling.depthA1 * mxScaleFactor.scaleFactorKa * 1 +
            tiling.baseN * tiling.baseK * tiling.depthB1 * GetBitSize<typename B_TYPE::T>() /
            ONE_BYTE_BIT_SIZE + tiling.baseN * tiling.baseK / Impl::SCALE_K_SIZE * tiling.depthB1 * mxScaleFactor.scaleFactorKb * 1 +
            int32_t(tiling.isBias) * tiling.baseN * GetBitSize<typename BIAS_TYPE::T>() / ONE_BYTE_BIT_SIZE;
        if (needUsedL1Size > l1Size) {
            tiling.stepM = 1;
            tiling.stepN = 1;
            tiling.stepKa = 1;
            tiling.stepKb = 1;
            tiling.depthA1 = 1;
            tiling.depthB1 = 1;
            tiling.dbL0A = 1;
            tiling.dbL0B = 1;
            tiling.mxTypePara = Impl::MIN_MX_PARAM; // scaleFactorKa = 1, scaleFactorKb = 1
        }
    }
#endif
}
} // namespace AscendC
#endif // _MATMUL_CONSTANT_TILING_IMPL_