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
 * \file philox_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_PHILOX_PHILOX_C310_IMPL_H
#define IMPL_MATH_PHILOX_PHILOX_C310_IMPL_H
namespace AscendC {
namespace PhiloxInternal {
// philox algorithm constant
constexpr uint32_t CONST_MUL_0 = 0xD2511F53;
constexpr uint32_t CONST_MUL_1 = 0xCD9E8D57;
constexpr uint32_t CONST_KEY_ADD_0 = 0x9E3779B9;
constexpr uint32_t CONST_KEY_ADD_1 = 0xBB67AE85;

// philox algorithm each iter 128bit(4*32bit)
constexpr uint16_t PHILOX_ONCE_COUNTER_BIT = 128;
constexpr uint16_t PHILOX_ONCE_COUNTER_BYTE = PHILOX_ONCE_COUNTER_BIT / 8;
// philox algorithm each iter 4 element, current type is B32(uint32_t/int32_t/float)
constexpr uint16_t PHILOX_ONCE_COUNTER_NUM = PHILOX_ONCE_COUNTER_BYTE / sizeof(uint32_t);
// philox parallel one column is VL/B32
constexpr uint32_t ELE_CNT_B32_ONCE = GetVecLen() / sizeof(uint32_t);
constexpr uint16_t PHILOX_ONCE_REPEAT_NUM = PHILOX_ONCE_COUNTER_NUM * ELE_CNT_B32_ONCE;

// uint32 to float32
// |1|_____8____|___________23___________|
// |s|exponent  | mantissa               |
constexpr uint32_t MANTISSA = static_cast<uint32_t>(0x7fffffu);  // 23 bit mantissa
constexpr uint32_t EXP_MASK = static_cast<uint32_t>(127) << 23u; // 7 bit exp
} // namespace PhiloxInternal

// 64 bit key and 128-bit counter, little endian
using PhiloxKey = uint32_t[2];
using PhiloxCounter = uint32_t[4];

struct PhiloxRandomParams {
    uint32_t stride;
    uint32_t row;
    uint32_t column;
};

__aicore__ inline void AddWith128Bits(MicroAPI::RegTensor<uint32_t> &ctr0, MicroAPI::RegTensor<uint32_t> &ctr1,
    MicroAPI::RegTensor<uint32_t> &ctr2, MicroAPI::RegTensor<uint32_t> &ctr3, MicroAPI::RegTensor<uint32_t> &value,
    MicroAPI::MaskReg &pg)
{
    MicroAPI::MaskReg pd;
    MicroAPI::RegTensor<uint32_t> vZero;
    Duplicate(vZero, 0x0);
    AddCarryOut(pd, ctr0, ctr0, value, pg);
    AddCarryOuts(pd, ctr1, ctr1, vZero, pd, pg);
    AddCarryOuts(pd, ctr2, ctr2, vZero, pd, pg);
    AddCarryOuts(pd, ctr3, ctr3, vZero, pd, pg);
}

__aicore__ inline void UInt2Float(MicroAPI::RegTensor<uint32_t> &tmpCtr0, MicroAPI::RegTensor<uint32_t> &tmpCtr1,
    MicroAPI::RegTensor<uint32_t> &tmpCtr2, MicroAPI::RegTensor<uint32_t> &tmpCtr3, MicroAPI::MaskReg &pg)
{
    MicroAPI::RegTensor<uint32_t> vb32ManMask, vb32ExpMask;
    Duplicate(vb32ManMask, PhiloxInternal::MANTISSA);
    Duplicate(vb32ExpMask, PhiloxInternal::EXP_MASK);
    And(tmpCtr0, tmpCtr0, vb32ManMask, pg);
    And(tmpCtr1, tmpCtr1, vb32ManMask, pg);
    And(tmpCtr2, tmpCtr2, vb32ManMask, pg);
    And(tmpCtr3, tmpCtr3, vb32ManMask, pg);
    Or(tmpCtr0, tmpCtr0, vb32ExpMask, pg);
    Or(tmpCtr1, tmpCtr1, vb32ExpMask, pg);
    Or(tmpCtr2, tmpCtr2, vb32ExpMask, pg);
    Or(tmpCtr3, tmpCtr3, vb32ExpMask, pg);
    Adds((MicroAPI::RegTensor<float> &)tmpCtr0, (MicroAPI::RegTensor<float> &)tmpCtr0, -1.0f, pg);
    Adds((MicroAPI::RegTensor<float> &)tmpCtr1, (MicroAPI::RegTensor<float> &)tmpCtr1, -1.0f, pg);
    Adds((MicroAPI::RegTensor<float> &)tmpCtr2, (MicroAPI::RegTensor<float> &)tmpCtr2, -1.0f, pg);
    Adds((MicroAPI::RegTensor<float> &)tmpCtr3, (MicroAPI::RegTensor<float> &)tmpCtr3, -1.0f, pg);
}

template <uint16_t Rounds>
__aicore__ inline void SpNetworkKernel(MicroAPI::RegTensor<uint32_t> &tmpL0, MicroAPI::RegTensor<uint32_t> &tmpH0,
    MicroAPI::RegTensor<uint32_t> &tmpL1, MicroAPI::RegTensor<uint32_t> &tmpH1, MicroAPI::RegTensor<uint32_t> &tmpCtr0,
    MicroAPI::RegTensor<uint32_t> &tmpCtr1, MicroAPI::RegTensor<uint32_t> &tmpCtr2,
    MicroAPI::RegTensor<uint32_t> &tmpCtr3, MicroAPI::RegTensor<uint32_t> &tmpKey0,
    MicroAPI::RegTensor<uint32_t> &tmpKey1, MicroAPI::RegTensor<uint32_t> &cMul0, MicroAPI::RegTensor<uint32_t> &cMul1,
    MicroAPI::MaskReg &pg)
{
    // pragma unroll vs manual unroll(442).
    // when count=16384, round 10, cycles(6904 vs 5549), vex(6822 vs 4260), ipc(0.988 vs 0.768)
    // #pragma unroll  // ccec may have a bug: https://codehub-y.huawei.com/c00564736/D_Compile_Issue/issues/172
    for (uint16_t j = 0; j < Rounds; j++) {
        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);
    }
}

template <uint16_t Rounds, typename T, bool DstUnalign = false>
__aicore__ inline void SpNetworkFull(__ubuf__ uint32_t *dstUbTail, uint16_t tailCount,
    MicroAPI::RegTensor<uint32_t> &ctr0, MicroAPI::RegTensor<uint32_t> &ctr1, MicroAPI::RegTensor<uint32_t> &ctr2,
    MicroAPI::RegTensor<uint32_t> &ctr3, MicroAPI::RegTensor<uint32_t> &key0, MicroAPI::RegTensor<uint32_t> &key1,
    MicroAPI::RegTensor<uint32_t> &cMul0, MicroAPI::RegTensor<uint32_t> &cMul1, MicroAPI::MaskReg &pg)
{
    MicroAPI::RegTensor<uint32_t> tmpCtr3, tmpCtr2, tmpCtr1, tmpCtr0;
    tmpCtr0 = ctr0;
    tmpCtr1 = ctr1;
    tmpCtr2 = ctr2;
    tmpCtr3 = ctr3;
    MicroAPI::RegTensor<uint32_t> tmpKey0 = key0;
    MicroAPI::RegTensor<uint32_t> tmpKey1 = key1;
    MicroAPI::RegTensor<uint32_t> tmpL0, tmpH0, tmpL1, tmpH1;
    SpNetworkKernel<Rounds>(tmpL0, tmpH0, tmpL1, tmpH1, tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, tmpKey0, tmpKey1, cMul0,
        cMul1, pg);

    if constexpr (std::is_same_v<T, float>) {
        UInt2Float(tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, pg);
    }

    if constexpr (DstUnalign) {
        MicroAPI::RegTensor<uint32_t> reorderIndex;
        MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)reorderIndex, 0);
        Muls(reorderIndex, reorderIndex, PhiloxInternal::PHILOX_ONCE_COUNTER_NUM, pg);
        // column % 4 = 0, scatter pgTail is tailCount / 4
        uint32_t sreg = static_cast<uint32_t>(tailCount / PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
        MicroAPI::MaskReg pgTail = MicroAPI::UpdateMask<T>(sreg);
        DataCopyScatter(dstUbTail, tmpCtr0, reorderIndex, pgTail);
        DataCopyScatter(dstUbTail + 1, tmpCtr1, reorderIndex, pgTail);
        DataCopyScatter(dstUbTail + 2, tmpCtr2, reorderIndex, pgTail);
        DataCopyScatter(dstUbTail + 3, tmpCtr3, reorderIndex, pgTail);
    } else {
        Interleave(tmpCtr0, tmpCtr2, tmpCtr0, tmpCtr2);
        Interleave(tmpCtr1, tmpCtr3, tmpCtr1, tmpCtr3);
        Interleave(tmpCtr0, tmpCtr1, tmpCtr0, tmpCtr1);
        Interleave(tmpCtr2, tmpCtr3, tmpCtr2, tmpCtr3);
        uint32_t sreg = static_cast<uint32_t>(tailCount);
        MicroAPI::MaskReg pgTail = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbTail, tmpCtr0,
            PhiloxInternal::ELE_CNT_B32_ONCE, pgTail);
        pgTail = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbTail, tmpCtr1,
            PhiloxInternal::ELE_CNT_B32_ONCE, pgTail);
        pgTail = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbTail, tmpCtr2,
            PhiloxInternal::ELE_CNT_B32_ONCE, pgTail);
        pgTail = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbTail, tmpCtr3,
            PhiloxInternal::ELE_CNT_B32_ONCE, pgTail);
    }
}

template <bool DstUnalign = false>
__aicore__ inline void PhiloxUnrollStoreTmpCtrl(__ubuf__ uint32_t *&dstUbT, MicroAPI::RegTensor<uint32_t> &tmpCtr0,
    MicroAPI::RegTensor<uint32_t> &tmpCtr1, MicroAPI::RegTensor<uint32_t> &tmpCtr2,
    MicroAPI::RegTensor<uint32_t> &tmpCtr3, MicroAPI::MaskReg &pg)
{
    if constexpr (DstUnalign) {
        MicroAPI::UnalignReg ureg;
        MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, tmpCtr0, ureg,
            PhiloxInternal::ELE_CNT_B32_ONCE);
        MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, tmpCtr1, ureg,
            PhiloxInternal::ELE_CNT_B32_ONCE);
        MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, tmpCtr2, ureg,
            PhiloxInternal::ELE_CNT_B32_ONCE);
        MicroAPI::DataCopyUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, tmpCtr3, ureg,
            PhiloxInternal::ELE_CNT_B32_ONCE);
        MicroAPI::DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, ureg, 0);
    } else {
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, tmpCtr0,
            PhiloxInternal::ELE_CNT_B32_ONCE, pg);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, tmpCtr1,
            PhiloxInternal::ELE_CNT_B32_ONCE, pg);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, tmpCtr2,
            PhiloxInternal::ELE_CNT_B32_ONCE, pg);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstUbT, tmpCtr3,
            PhiloxInternal::ELE_CNT_B32_ONCE, pg);
    }
}

template <bool DstUnalign = false>
__aicore__ inline void PhiloxUnrollLoadTmpCtrl(__ubuf__ uint32_t *&dstUbTT0, __ubuf__ uint32_t *&dstUbTT1,
    __ubuf__ uint32_t *&dstUbTT2, __ubuf__ uint32_t *&dstUbTT3, MicroAPI::RegTensor<uint32_t> &tmpCtr0,
    MicroAPI::RegTensor<uint32_t> &tmpCtr1, MicroAPI::RegTensor<uint32_t> &tmpCtr2,
    MicroAPI::RegTensor<uint32_t> &tmpCtr3)
{
    if constexpr (DstUnalign) {
        MicroAPI::UnalignReg ureg;
        MicroAPI::DataCopyUnAlignPre(ureg, dstUbTT0);
        MicroAPI::DataCopyUnAlign(tmpCtr0, ureg, dstUbTT0,
            PhiloxInternal::ELE_CNT_B32_ONCE * PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
        MicroAPI::DataCopyUnAlignPre(ureg, dstUbTT1);
        MicroAPI::DataCopyUnAlign(tmpCtr1, ureg, dstUbTT1,
            PhiloxInternal::ELE_CNT_B32_ONCE * PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
        MicroAPI::DataCopyUnAlignPre(ureg, dstUbTT2);
        MicroAPI::DataCopyUnAlign(tmpCtr2, ureg, dstUbTT2,
            PhiloxInternal::ELE_CNT_B32_ONCE * PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
        MicroAPI::DataCopyUnAlignPre(ureg, dstUbTT3);
        MicroAPI::DataCopyUnAlign(tmpCtr3, ureg, dstUbTT3,
            PhiloxInternal::ELE_CNT_B32_ONCE * PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
    } else {
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tmpCtr0, dstUbTT0,
            PhiloxInternal::ELE_CNT_B32_ONCE * PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tmpCtr1, dstUbTT1,
            PhiloxInternal::ELE_CNT_B32_ONCE * PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tmpCtr2, dstUbTT2,
            PhiloxInternal::ELE_CNT_B32_ONCE * PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
        MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(tmpCtr3, dstUbTT3,
            PhiloxInternal::ELE_CNT_B32_ONCE * PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
    }
}

template <typename T, bool DstUnalign = false>
__aicore__ inline void PhiloxRound10MainBlockUnroll442(__ubuf__ uint32_t *dstUb, uint16_t mainIter,
    MicroAPI::RegTensor<uint32_t> &ctr0, MicroAPI::RegTensor<uint32_t> &ctr1, MicroAPI::RegTensor<uint32_t> &ctr2,
    MicroAPI::RegTensor<uint32_t> &ctr3, MicroAPI::RegTensor<uint32_t> &key0, MicroAPI::RegTensor<uint32_t> &key1,
    MicroAPI::RegTensor<uint32_t> &cMul0, MicroAPI::RegTensor<uint32_t> &cMul1,
    MicroAPI::RegTensor<uint32_t> &vEleStrideB32OneRow, MicroAPI::MaskReg &pg)
{
    MicroAPI::RegTensor<uint32_t> tmpCtr3, tmpCtr2, tmpCtr1, tmpCtr0;
    __ubuf__ uint32_t *dstUbT = dstUb;

    MicroAPI::RegTensor<uint32_t> reorderIndex;
    Arange((MicroAPI::RegTensor<int32_t> &)reorderIndex, 0);
    Muls(reorderIndex, reorderIndex, PhiloxInternal::PHILOX_ONCE_COUNTER_NUM, pg);

    for (uint16_t i = 0; i < mainIter; i++) {
        tmpCtr0 = ctr0;
        tmpCtr1 = ctr1;
        tmpCtr2 = ctr2;
        tmpCtr3 = ctr3;
        MicroAPI::RegTensor<uint32_t> tmpKey0 = key0;
        MicroAPI::RegTensor<uint32_t> tmpKey1 = key1;
        MicroAPI::RegTensor<uint32_t> tmpL0, tmpH0, tmpL1, tmpH1;

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        PhiloxUnrollStoreTmpCtrl<DstUnalign>(dstUbT, tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, pg);
        AddWith128Bits(ctr0, ctr1, ctr2, ctr3, vEleStrideB32OneRow, pg);
    }

    dstUbT = dstUb;
    __ubuf__ uint32_t *dstUbTT0 = dstUbT;
    __ubuf__ uint32_t *dstUbTT1 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE;
    __ubuf__ uint32_t *dstUbTT2 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE * 2;
    __ubuf__ uint32_t *dstUbTT3 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE * 3;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < mainIter; i++) {
        PhiloxUnrollLoadTmpCtrl<DstUnalign>(dstUbTT0, dstUbTT1, dstUbTT2, dstUbTT3, tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3);

        MicroAPI::RegTensor<uint32_t> tmpKey0 = key0;
        MicroAPI::RegTensor<uint32_t> tmpKey1 = key1;

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        MicroAPI::RegTensor<uint32_t> tmpL0, tmpH0, tmpL1, tmpH1;

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        PhiloxUnrollStoreTmpCtrl<DstUnalign>(dstUbT, tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, pg);
    }

    dstUbT = dstUb;
    dstUbTT0 = dstUbT;
    dstUbTT1 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE;
    dstUbTT2 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE * 2;
    dstUbTT3 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE * 3;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < mainIter; i++) {
        PhiloxUnrollLoadTmpCtrl<DstUnalign>(dstUbTT0, dstUbTT1, dstUbTT2, dstUbTT3, tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3);

        MicroAPI::RegTensor<uint32_t> tmpKey0 = key0;
        MicroAPI::RegTensor<uint32_t> tmpKey1 = key1;

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        MicroAPI::RegTensor<uint32_t> tmpL0, tmpH0, tmpL1, tmpH1;

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;

        if constexpr (std::is_same_v<T, float>) {
            UInt2Float(tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, pg);
        }

        DataCopyScatter(dstUb, tmpCtr0, reorderIndex, pg);
        DataCopyScatter(dstUb + 1, tmpCtr1, reorderIndex, pg);
        DataCopyScatter(dstUb + 2, tmpCtr2, reorderIndex, pg);
        DataCopyScatter(dstUb + 3, tmpCtr3, reorderIndex, pg);
        Adds(reorderIndex, reorderIndex, PhiloxInternal::PHILOX_ONCE_REPEAT_NUM, pg);
    }
}

template <typename T, bool DstUnalign = false>
__aicore__ inline void PhiloxRound7MainBlockUnroll43(__ubuf__ uint32_t *dstUb, uint16_t mainIter,
    MicroAPI::RegTensor<uint32_t> &ctr0, MicroAPI::RegTensor<uint32_t> &ctr1, MicroAPI::RegTensor<uint32_t> &ctr2,
    MicroAPI::RegTensor<uint32_t> &ctr3, MicroAPI::RegTensor<uint32_t> &key0, MicroAPI::RegTensor<uint32_t> &key1,
    MicroAPI::RegTensor<uint32_t> &cMul0, MicroAPI::RegTensor<uint32_t> &cMul1,
    MicroAPI::RegTensor<uint32_t> &vEleStrideB32OneRow, MicroAPI::MaskReg &pg)
{
    MicroAPI::RegTensor<uint32_t> tmpCtr3, tmpCtr2, tmpCtr1, tmpCtr0;
    __ubuf__ uint32_t *dstUbT = dstUb;

    MicroAPI::RegTensor<uint32_t> reorderIndex;
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t> &)reorderIndex, 0);
    Muls(reorderIndex, reorderIndex, PhiloxInternal::PHILOX_ONCE_COUNTER_NUM, pg);

    for (uint16_t i = 0; i < mainIter; i++) {
        tmpCtr0 = ctr0;
        tmpCtr1 = ctr1;
        tmpCtr2 = ctr2;
        tmpCtr3 = ctr3;
        MicroAPI::RegTensor<uint32_t> tmpKey0 = key0;
        MicroAPI::RegTensor<uint32_t> tmpKey1 = key1;
        MicroAPI::RegTensor<uint32_t> tmpL0, tmpH0, tmpL1, tmpH1;

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        PhiloxUnrollStoreTmpCtrl<DstUnalign>(dstUbT, tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, pg);
        AddWith128Bits(ctr0, ctr1, ctr2, ctr3, vEleStrideB32OneRow, pg);
    }

    dstUbT = dstUb;
    __ubuf__ uint32_t *dstUbTT0 = dstUbT;
    __ubuf__ uint32_t *dstUbTT1 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE;
    __ubuf__ uint32_t *dstUbTT2 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE * 2;
    __ubuf__ uint32_t *dstUbTT3 = dstUbT + PhiloxInternal::ELE_CNT_B32_ONCE * 3;
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < mainIter; i++) {
        PhiloxUnrollLoadTmpCtrl<DstUnalign>(dstUbTT0, dstUbTT1, dstUbTT2, dstUbTT3, tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3);

        MicroAPI::RegTensor<uint32_t> tmpKey0 = key0;
        MicroAPI::RegTensor<uint32_t> tmpKey1 = key1;

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        MicroAPI::RegTensor<uint32_t> tmpL0, tmpH0, tmpL1, tmpH1;

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;
        Adds(tmpKey0, tmpKey0, PhiloxInternal::CONST_KEY_ADD_0, pg);
        Adds(tmpKey1, tmpKey1, PhiloxInternal::CONST_KEY_ADD_1, pg);

        Mull(tmpL0, tmpH0, tmpCtr0, cMul0, pg);
        Mull(tmpL1, tmpH1, tmpCtr2, cMul1, pg);
        Xor(tmpH1, tmpH1, tmpCtr1, pg);
        Xor(tmpCtr0, tmpH1, tmpKey0, pg);
        Xor(tmpH0, tmpH0, tmpCtr3, pg);
        Xor(tmpCtr2, tmpH0, tmpKey1, pg);
        tmpCtr1 = tmpL1;
        tmpCtr3 = tmpL0;

        if constexpr (std::is_same_v<T, float>) {
            UInt2Float(tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, pg);
        }

        DataCopyScatter(dstUb, tmpCtr0, reorderIndex, pg);
        DataCopyScatter(dstUb + 1, tmpCtr1, reorderIndex, pg);
        DataCopyScatter(dstUb + 2, tmpCtr2, reorderIndex, pg);
        DataCopyScatter(dstUb + 3, tmpCtr3, reorderIndex, pg);
        Adds(reorderIndex, reorderIndex, PhiloxInternal::PHILOX_ONCE_REPEAT_NUM, pg);
    }
}

template <uint16_t Rounds = 7, typename T, bool DstUnalign = false>
__aicore__ inline void PhiloxRoundMainBlockUnroll(__ubuf__ uint32_t *dstUb, uint16_t mainIter,
    MicroAPI::RegTensor<uint32_t> &ctr0, MicroAPI::RegTensor<uint32_t> &ctr1, MicroAPI::RegTensor<uint32_t> &ctr2,
    MicroAPI::RegTensor<uint32_t> &ctr3, MicroAPI::RegTensor<uint32_t> &key0, MicroAPI::RegTensor<uint32_t> &key1,
    MicroAPI::RegTensor<uint32_t> &cMul0, MicroAPI::RegTensor<uint32_t> &cMul1,
    MicroAPI::RegTensor<uint32_t> &vEleStrideB32OneRow, MicroAPI::MaskReg &pg)
{
    if constexpr (Rounds == 10) {
        // main block with 4 + 4 + 2 unroll
        PhiloxRound10MainBlockUnroll442<T, DstUnalign>(dstUb, mainIter, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0,
            cMul1, vEleStrideB32OneRow, pg);
    } else {
        // main block with 4 + 3 unroll
        PhiloxRound7MainBlockUnroll43<T, DstUnalign>(dstUb, mainIter, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0, cMul1,
            vEleStrideB32OneRow, pg);
    }
}

__aicore__ inline void PhiloxCounterInit(const PhiloxCounter &philoxCounter, MicroAPI::RegTensor<uint32_t> &ctr0,
    MicroAPI::RegTensor<uint32_t> &ctr1, MicroAPI::RegTensor<uint32_t> &ctr2, MicroAPI::RegTensor<uint32_t> &ctr3,
    MicroAPI::RegTensor<int32_t> &incIdx, MicroAPI::MaskReg &pg)
{
    Duplicate(ctr0, philoxCounter[0]);
    Duplicate(ctr1, philoxCounter[1]);
    Duplicate(ctr2, philoxCounter[2]);
    Duplicate(ctr3, philoxCounter[3]);
    AddWith128Bits(ctr0, ctr1, ctr2, ctr3, (MicroAPI::RegTensor<uint32_t> &)incIdx, pg);
}

template <uint16_t Rounds = 7, typename T, bool DstUnalign = false>
__aicore__ inline void PhiloxRandomOneRow(__ubuf__ uint32_t *dstUb, __ubuf__ uint32_t *dstUbTail,
    const PhiloxKey &philoxKey, const PhiloxCounter &philoxCounter, uint16_t mainIter, uint16_t tailCount)
{
    MicroAPI::MaskReg pg = MicroAPI::CreateMask<uint32_t>();

    MicroAPI::RegTensor<uint32_t> ctr3, ctr2, ctr1, ctr0;
    MicroAPI::RegTensor<int32_t> incIdx;
    MicroAPI::Arange(incIdx, 0);
    PhiloxCounterInit(philoxCounter, ctr0, ctr1, ctr2, ctr3, incIdx, pg);

    MicroAPI::RegTensor<uint32_t> vEleStrideB32OneRow;
    Duplicate(vEleStrideB32OneRow, PhiloxInternal::ELE_CNT_B32_ONCE);

    MicroAPI::RegTensor<uint32_t> key1, key0;
    Duplicate(key0, philoxKey[0]);
    Duplicate(key1, philoxKey[1]);

    MicroAPI::RegTensor<uint32_t> cMul0, cMul1;
    Duplicate(cMul0, PhiloxInternal::CONST_MUL_0);
    Duplicate(cMul1, PhiloxInternal::CONST_MUL_1);

    PhiloxRoundMainBlockUnroll<Rounds, T, DstUnalign>(dstUb, mainIter, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0, cMul1,
        vEleStrideB32OneRow, pg);

    if (tailCount > 0) {
        SpNetworkFull<Rounds, T>(dstUbTail, tailCount, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0, cMul1, pg);
    }
}

/*
// derive index calculation
for (i : row)
  for (j : column)  // column < half one repeat
    index[i * column + j] = i * stride + j
==>
factor = ONE_REPEAT_LEN / column
i.i_extent = factor
i.o_extent = row / factor
i = i.o * factor + i.i  < row // contains an if constraint condition
for (i.o, 0, row / factor) {
  for (i.i, 0, factor) {
    if (i.o * factor + i.i  < row) {
      for (j, 0, column) {
        index[(i.o * factor + i.i) * column + j] = i.o * factor * stride + i.i * stride + j
      }
    }
  }
}
==>
j_fuse = i.i * column + j
j_fuse_extent = factor * column < ONE_REPEAT_LEN
i.i = j_fuse / column
j = j_fuse % column
for (i.o, 0, row / factor) {
  if (i.o * factor * column + j_fuse < row * column) {
    for (j_fuse, 0, factor * column) {
      index[i.o * factor * column + j_fuse] = i.o * factor * stride + j_fuse / column * stride + j_fuse % column
    }
  }
}
*/
__aicore__ inline void PhiloxRandomIndexCal(__ubuf__ int32_t *indexUb, const PhiloxRandomParams &params,
    const uint32_t fuseFactor)
{
    __ubuf__ int32_t *indexUbT = indexUb;
    MicroAPI::MaskReg pg = MicroAPI::CreateMask<uint32_t>();
    MicroAPI::RegTensor<int32_t> index, incIdx;
    MicroAPI::UnalignReg ureg;
    uint32_t stride = static_cast<uint32_t>(params.stride / PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);
    uint32_t elementNum = params.column / PhiloxInternal::PHILOX_ONCE_COUNTER_NUM;
    for (uint16_t i = 0; i < fuseFactor; i++) {
        MicroAPI::Duplicate(index, i);
        MicroAPI::Muls(index, index, stride, pg);
        MicroAPI::Arange(incIdx, 0);
        MicroAPI::Add(index, index, incIdx, pg);
        MicroAPI::DataCopyUnAlign<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(indexUbT, index, ureg, elementNum);
    }
    MicroAPI::DataCopyUnAlignPost<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(indexUbT, ureg, 0);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <uint16_t Rounds = 7, typename T, bool DstBlockUnalign, bool DstRepeatUnalign>
__aicore__ inline void PhiloxRandomMultiRowWithFuse(__ubuf__ uint32_t *dstUbStart, __ubuf__ int32_t *indexUb,
    const PhiloxKey &philoxKey, const PhiloxCounter &philoxCounter, const PhiloxRandomParams &params,
    const uint32_t fuseFactor, const uint32_t mainFuseAxis, const uint32_t mainRowsNum, const uint32_t tailFuseAxis)
{
    PhiloxRandomIndexCal(indexUb, params, fuseFactor);
    MicroAPI::RegTensor<uint32_t> ctr3, ctr2, ctr1, ctr0;

    MicroAPI::MaskReg pg = MicroAPI::CreateMask<uint32_t>();
    MicroAPI::RegTensor<int32_t> incIdx;
    MicroAPI::DataCopy(incIdx, indexUb);
    PhiloxCounterInit(philoxCounter, ctr0, ctr1, ctr2, ctr3, incIdx, pg);

    MicroAPI::RegTensor<uint32_t> key1, key0;
    Duplicate(key0, philoxKey[0]);
    Duplicate(key1, philoxKey[1]);

    MicroAPI::RegTensor<uint32_t> cMul0, cMul1;
    Duplicate(cMul0, PhiloxInternal::CONST_MUL_0);
    Duplicate(cMul1, PhiloxInternal::CONST_MUL_1);

    MicroAPI::RegTensor<uint32_t> vEleStrideB32OneRow;
    Duplicate(vEleStrideB32OneRow, fuseFactor * params.stride / PhiloxInternal::PHILOX_ONCE_COUNTER_NUM);

    if constexpr (!DstRepeatUnalign) {
        PhiloxRoundMainBlockUnroll<Rounds, T, DstBlockUnalign>(dstUbStart, mainRowsNum, ctr0, ctr1, ctr2, ctr3, key0,
            key1, cMul0, cMul1, vEleStrideB32OneRow, pg);
    } else {
        // if dst repat unalign, use MainBlockUnroll algo may be dst overlap. SpNetworkFull can control each element.
        for (uint16_t i = 0; i < mainRowsNum; i++) {
            __ubuf__ uint32_t *dstUb = dstUbStart + i * mainFuseAxis;
            SpNetworkFull<Rounds, T, DstBlockUnalign>(dstUb, mainFuseAxis, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0,
                cMul1, pg);
            AddWith128Bits(ctr0, ctr1, ctr2, ctr3, vEleStrideB32OneRow, pg);
        }
    }

    if (tailFuseAxis > 0) {
        __ubuf__ uint32_t *dstUbTail = dstUbStart + mainRowsNum * mainFuseAxis;
        SpNetworkFull<Rounds, T, DstBlockUnalign>(dstUbTail, tailFuseAxis, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0,
            cMul1, pg);
    }
}

template <uint16_t Rounds = 7, typename T, bool DstUnalign>
__aicore__ inline void PhiloxRandomMultiRowNoFuse(__ubuf__ uint32_t *dstUbStart, const PhiloxKey &philoxKey,
    const PhiloxCounter &philoxCounter, const PhiloxRandomParams &params, uint32_t strideCounterOneRow,
    uint16_t mainIter, uint16_t tailCount, uint16_t hasTail)
{
    MicroAPI::RegTensor<uint32_t> vEleStrideB32OneRow;
    Duplicate(vEleStrideB32OneRow, PhiloxInternal::ELE_CNT_B32_ONCE);

    MicroAPI::RegTensor<uint32_t> key1, key0;
    Duplicate(key0, philoxKey[0]);
    Duplicate(key1, philoxKey[1]);

    MicroAPI::RegTensor<uint32_t> cMul0, cMul1;
    Duplicate(cMul0, PhiloxInternal::CONST_MUL_0);
    Duplicate(cMul1, PhiloxInternal::CONST_MUL_1);

    MicroAPI::RegTensor<uint32_t> ctr3, ctr2, ctr1, ctr0;
    MicroAPI::MaskReg pg = MicroAPI::CreateMask<uint32_t>();

    for (uint16_t i = 0; i < params.row; i++) {
        __ubuf__ uint32_t *dstUb = dstUbStart + i * params.column;
        __ubuf__ uint32_t *dstUbTail = dstUb + mainIter * PhiloxInternal::PHILOX_ONCE_REPEAT_NUM;

        MicroAPI::RegTensor<int32_t> incIdx;
        MicroAPI::Arange(incIdx, i * strideCounterOneRow);
        PhiloxCounterInit(philoxCounter, ctr0, ctr1, ctr2, ctr3, incIdx, pg);
        PhiloxRoundMainBlockUnroll<Rounds, T, DstUnalign>(dstUb, mainIter, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0,
            cMul1, vEleStrideB32OneRow, pg);
        for (uint16_t j = 0; j < hasTail; j++) {
            SpNetworkFull<Rounds, T, DstUnalign>(dstUbTail, tailCount, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0, cMul1,
                pg);
        }
    }
}

template <uint16_t Rounds = 7, typename T>
__aicore__ inline void PhiloxRandomImpl(const LocalTensor<T> &dstLocal, const PhiloxKey &philoxKey,
    const PhiloxCounter &philoxCounter, uint16_t count)
{
    static_assert(SupportType<T, int32_t, uint32_t, float>(),
        "PhiloxRandom API only support int32_t/uint32_t/float type");
    static_assert(Rounds == 7 || Rounds == 10, "PhiloxRandom API only support 7 or 10 Rounds ");

    __ubuf__ uint32_t *dstUb = (__ubuf__ uint32_t *)dstLocal.GetPhyAddr();
    uint16_t mainIter = count / PhiloxInternal::PHILOX_ONCE_REPEAT_NUM;
    uint16_t tailCount = count - mainIter * PhiloxInternal::PHILOX_ONCE_REPEAT_NUM;
    __ubuf__ uint32_t *dstUbTail = dstUb + mainIter * PhiloxInternal::PHILOX_ONCE_REPEAT_NUM;
    VF_CALL<PhiloxRandomOneRow<Rounds, T, false>>(dstUb, dstUbTail, philoxKey, philoxCounter, mainIter, tailCount);
}

template <uint16_t Rounds = 7, typename T>
__aicore__ inline void PhiloxRandomImpl(const LocalTensor<T> &dstLocal, const PhiloxKey &philoxKey,
    const PhiloxCounter &philoxCounter, const PhiloxRandomParams &params)
{
    static_assert(SupportType<T, int32_t, uint32_t, float>(),
        "PhiloxRandom API only support int32_t/uint32_t/float type");
    static_assert(Rounds == 7 || Rounds == 10, "PhiloxRandom API only support 7 or 10 Rounds ");

    ASCENDC_ASSERT((params.stride % PhiloxInternal::PHILOX_ONCE_COUNTER_NUM == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "params.stride % 4 = 0!"); });
    ASCENDC_ASSERT((params.column % PhiloxInternal::PHILOX_ONCE_COUNTER_NUM == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "params.column % 4 = 0!"); });
    ASCENDC_ASSERT((params.stride >= params.column), { KERNEL_LOG(KERNEL_ERROR, "params.stride >= params.column!"); });
    ASCENDC_ASSERT((params.row > 0 && params.column > 0),
                   { KERNEL_LOG(KERNEL_ERROR, "params.row > 0 && params.column > 0!"); });

    __ubuf__ uint32_t *dstUbStart = (__ubuf__ uint32_t *)dstLocal.GetPhyAddr();

    // judge fuse axis and unalign pattern
    if (params.row == 1 || params.stride == params.column) {
        // if one row or stride == column, continuous and align, count = row * column
        PhiloxRandomImpl<Rounds, T>(dstLocal, philoxKey, philoxCounter, params.row * params.column);
    } else if (params.column <= PhiloxInternal::PHILOX_ONCE_REPEAT_NUM) {
        // fuse axis condition: params.column <= one repeat
        uint32_t fuseFactor = PhiloxInternal::PHILOX_ONCE_REPEAT_NUM / params.column; // fuseFactor >= 1
        uint32_t mainFuseAxis = params.column * fuseFactor;
        uint32_t mainRowsNum = params.row / fuseFactor;
        uint32_t tailFuseAxis = params.row * params.column - mainFuseAxis * mainRowsNum;
        LocalTensor<T> indexTensor;
        PopStackBuffer<T, TPosition::LCM>(indexTensor);
        __ubuf__ int32_t *indexUb = (__ubuf__ int32_t *)indexTensor.GetPhyAddr();
        if (mainFuseAxis == PhiloxInternal::PHILOX_ONCE_REPEAT_NUM) {
            // DstBlock align, DstRepeat align
            VF_CALL<PhiloxRandomMultiRowWithFuse<Rounds, T, false, false>>(dstUbStart, indexUb, philoxKey,
                philoxCounter, params, fuseFactor, mainFuseAxis, mainRowsNum, tailFuseAxis);
        } else if (mainFuseAxis * sizeof(uint32_t) % ONE_BLOCK_SIZE == 0) {
            // DstBlock align, DstRepeat unalign. use SpNetworkFull align condition
            VF_CALL<PhiloxRandomMultiRowWithFuse<Rounds, T, false, true>>(dstUbStart, indexUb, philoxKey, philoxCounter,
                params, fuseFactor, mainFuseAxis, mainRowsNum, tailFuseAxis);
        } else {
            // DstBlock unalign, DstRepeat unalign
            VF_CALL<PhiloxRandomMultiRowWithFuse<Rounds, T, true, true>>(dstUbStart, indexUb, philoxKey, philoxCounter,
                params, fuseFactor, mainFuseAxis, mainRowsNum, tailFuseAxis);
        }
    } else {
        // no fuse axis
        uint32_t strideCounterOneRow = params.stride / PhiloxInternal::PHILOX_ONCE_COUNTER_NUM;
        uint16_t mainIter = params.column / PhiloxInternal::PHILOX_ONCE_REPEAT_NUM;
        uint16_t tailCount = params.column - mainIter * PhiloxInternal::PHILOX_ONCE_REPEAT_NUM;
        uint16_t hasTail = static_cast<uint16_t>(tailCount > 0);
        if (params.column / PhiloxInternal::PHILOX_ONCE_COUNTER_NUM * sizeof(uint32_t) % ONE_BLOCK_SIZE == 0) {
            // align pattern. use PhiloxRoundMainBlockUnroll align condition
            VF_CALL<PhiloxRandomMultiRowNoFuse<Rounds, T, false>>(dstUbStart, philoxKey, philoxCounter, params,
                strideCounterOneRow, mainIter, tailCount, hasTail);
        } else {
            // unalign pattern. each process element: column / PHILOX_ONCE_COUNTER_NUM
            VF_CALL<PhiloxRandomMultiRowNoFuse<Rounds, T, true>>(dstUbStart, philoxKey, philoxCounter, params,
                strideCounterOneRow, mainIter, tailCount, hasTail);
        }
    }
}
} // namespace AscendC
#endif // IMPL_MATH_PHILOX_PHILOX_C310_IMPL_H
