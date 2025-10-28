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
 * \file sort_impl.h
 * \brief
 */
#ifndef IMPL_SORT_SORT_SORT_IMPL_H
#define IMPL_SORT_SORT_SORT_IMPL_H

#include "../sort_common_utils.h"

namespace AscendC {
enum class SortType {
    RADIX_SORT,
    MERGE_SORT
};
struct SortConfig {
    SortType type = SortType::RADIX_SORT;
    bool isDescend = false;
};

namespace internal {
constexpr SortConfig defaultSortConfig = { SortType::RADIX_SORT, false };

} // namespace internal

namespace MicroAPI {
namespace internal {
__simd_callee__ inline void ConvertRegToWithShift(__local_mem__ uint32_t *&input, RegTensor<uint8_t> &dst,
    const int16_t offset)
{
    constexpr uint32_t eleCountPerVL = GetVecLen() / sizeof(uint32_t);
    MaskReg fullMask = CreateMask<uint32_t>();
    RegTensor<uint32_t> inputP0, inputP1, inputP2, inputP3;
    DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(inputP0, input, eleCountPerVL);
    DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(inputP1, input, eleCountPerVL);
    DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(inputP2, input, eleCountPerVL);
    DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(inputP3, input, eleCountPerVL);

    ShiftRights(inputP0, inputP0, offset, fullMask);
    ShiftRights(inputP1, inputP1, offset, fullMask);
    ShiftRights(inputP2, inputP2, offset, fullMask);
    ShiftRights(inputP3, inputP3, offset, fullMask);

    RegTensor<uint16_t> tmpU16Reg0, tmpU16Reg1, tmpU16Reg2;

    DeInterleave(tmpU16Reg0, tmpU16Reg1, (RegTensor<uint16_t> &)inputP0, (RegTensor<uint16_t> &)inputP1);
    DeInterleave(tmpU16Reg2, tmpU16Reg1, (RegTensor<uint16_t> &)inputP2, (RegTensor<uint16_t> &)inputP3);

    RegTensor<uint8_t> tmpReg;
    DeInterleave(dst, tmpReg, (RegTensor<uint8_t> &)tmpU16Reg0, (RegTensor<uint8_t> &)tmpU16Reg2);
}

__simd_callee__ inline void ConvertRegToWithShift(__local_mem__ uint64_t *&input, RegTensor<uint8_t> &dst,
    const int16_t offset)
{
    constexpr uint32_t eleCountPerVL = GetVecLen() / sizeof(uint64_t);
    MaskReg fullMask = CreateMask<uint64_t>();
    RegTensor<uint64_t> inputP0, inputP1, inputP2, inputP3, inputP4, inputP5, inputP6, inputP7;
    DataCopy<uint64_t, PostLiteral::POST_MODE_UPDATE>(inputP0, input, eleCountPerVL);
    DataCopy<uint64_t, PostLiteral::POST_MODE_UPDATE>(inputP1, input, eleCountPerVL);
    DataCopy<uint64_t, PostLiteral::POST_MODE_UPDATE>(inputP2, input, eleCountPerVL);
    DataCopy<uint64_t, PostLiteral::POST_MODE_UPDATE>(inputP3, input, eleCountPerVL);
    DataCopy<uint64_t, PostLiteral::POST_MODE_UPDATE>(inputP4, input, eleCountPerVL);
    DataCopy<uint64_t, PostLiteral::POST_MODE_UPDATE>(inputP5, input, eleCountPerVL);
    DataCopy<uint64_t, PostLiteral::POST_MODE_UPDATE>(inputP6, input, eleCountPerVL);
    DataCopy<uint64_t, PostLiteral::POST_MODE_UPDATE>(inputP7, input, eleCountPerVL);

    ShiftRights(inputP0, inputP0, offset, fullMask);
    ShiftRights(inputP1, inputP1, offset, fullMask);
    ShiftRights(inputP2, inputP2, offset, fullMask);
    ShiftRights(inputP3, inputP3, offset, fullMask);
    ShiftRights(inputP4, inputP4, offset, fullMask);
    ShiftRights(inputP5, inputP5, offset, fullMask);
    ShiftRights(inputP6, inputP6, offset, fullMask);
    ShiftRights(inputP7, inputP7, offset, fullMask);

    RegTensor<uint32_t> tmpU32Reg0, tmpU32Reg1, tmpU32Reg2, tmpU32Reg3, tmpU32Reg4;
    DeInterleave(tmpU32Reg0, tmpU32Reg4, (RegTensor<uint32_t> &)inputP0, (RegTensor<uint32_t> &)inputP1);
    DeInterleave(tmpU32Reg1, tmpU32Reg4, (RegTensor<uint32_t> &)inputP2, (RegTensor<uint32_t> &)inputP3);
    DeInterleave(tmpU32Reg2, tmpU32Reg4, (RegTensor<uint32_t> &)inputP4, (RegTensor<uint32_t> &)inputP5);
    DeInterleave(tmpU32Reg3, tmpU32Reg4, (RegTensor<uint32_t> &)inputP6, (RegTensor<uint32_t> &)inputP7);

    RegTensor<uint16_t> tmpU16Reg0, tmpU16Reg1, tmpU16Reg2;

    DeInterleave(tmpU16Reg0, tmpU16Reg1, (RegTensor<uint16_t> &)tmpU32Reg0, (RegTensor<uint16_t> &)tmpU32Reg1);
    DeInterleave(tmpU16Reg2, tmpU16Reg1, (RegTensor<uint16_t> &)tmpU32Reg2, (RegTensor<uint16_t> &)tmpU32Reg3);

    RegTensor<uint8_t> tmpReg;
    DeInterleave(dst, tmpReg, (RegTensor<uint8_t> &)tmpU16Reg0, (RegTensor<uint8_t> &)tmpU16Reg2);
}

__simd_callee__ inline void ConvertRegToWithShift(__local_mem__ uint16_t *&input, RegTensor<uint8_t> &dst,
    const int16_t offset)
{
    constexpr uint32_t eleCountPerVL = GetVecLen() / sizeof(uint16_t);
    MaskReg fullMask = CreateMask<uint16_t>();
    RegTensor<uint16_t> inputP0, inputP1;
    DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(inputP0, input, eleCountPerVL);
    DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(inputP1, input, eleCountPerVL);

    ShiftRights(inputP0, inputP0, offset, fullMask);
    ShiftRights(inputP1, inputP1, offset, fullMask);

    RegTensor<uint8_t> tmpReg;
    DeInterleave(dst, tmpReg, (RegTensor<uint8_t> &)inputP0, (RegTensor<uint8_t> &)inputP1);
}

__simd_callee__ inline void ConvertRegToWithShift(__local_mem__ uint8_t *&input, RegTensor<uint8_t> &dst,
    const int16_t offset)
{
    constexpr uint32_t eleCountPerVL = GetVecLen();
    DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(dst, input, eleCountPerVL);
}

template <typename T, typename U, bool isDescend = false>
__simd_vf__ inline void PreProcess(__local_mem__ U *src, __local_mem__ U *dst, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen() / sizeof(T));
    uint32_t currCount = count;
    constexpr uint32_t postUpdateSize = GetVecLen() / sizeof(U);
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<U>(currCount);

        RegTensor<U> local;
        DataCopy<U, PostLiteral::POST_MODE_UPDATE>(local, (__local_mem__ U *&)src, postUpdateSize);
        if constexpr (SupportType<T, int8_t, int16_t, int32_t, float, half, bfloat16_t, int64_t>()) {
            AscendC::MicroAPI::Internal::TwiddleIn<T>(local, local, maskReg);
        }
        if constexpr (isDescend) {
            Not(local, local, maskReg);
        }
        DataCopy<U, PostLiteral::POST_MODE_UPDATE>((__local_mem__ U *&)dst, local, postUpdateSize, maskReg);
    }
}

template <typename T, typename U, bool isDescend = false>
__simd_vf__ inline void PostProcess(__local_mem__ U *src, __local_mem__ U *dst, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen() / sizeof(T));
    constexpr uint32_t postUpdateSize = GetVecLen() / sizeof(U);
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<U>(count);

        RegTensor<U> local;
        DataCopy<U, PostLiteral::POST_MODE_UPDATE>(local, (__local_mem__ U *&)src, postUpdateSize);
        if constexpr (isDescend) {
            Not(local, local, maskReg);
        }
        if constexpr (SupportType<T, int8_t, int16_t, int32_t, float, half, bfloat16_t, int64_t>()) {
            AscendC::MicroAPI::Internal::TwiddleOut<T>(local, local, maskReg);
        }
        DataCopy<U, PostLiteral::POST_MODE_UPDATE>((__local_mem__ U *&)dst, local, postUpdateSize, maskReg);
    }
}

template <typename T>
__simd_vf__ inline void GetExclusiveSum(__local_mem__ T *srcValue, __local_mem__ uint8_t *tmpSrc,
    __local_mem__ uint8_t *tmpSrcCopy, __local_mem__ uint16_t *exclusiveSum, const uint32_t count, const int32_t round)
{
    // Traverse all data to get a Exclusive Sum
    uint16_t repeatTime = DivCeil(count, GetVecLen());
    int32_t bitoffset = round * 8;

    RegTensor<uint16_t> distributeHistP0;
    RegTensor<uint16_t> distributeHistP1;
    RegTensor<uint16_t> cumulativeHistP0;
    RegTensor<uint16_t> cumulativeHistP1;

    MaskReg b16FullMask = CreateMask<uint16_t>();

    Duplicate(distributeHistP0, 0, b16FullMask);
    Duplicate(distributeHistP1, 0, b16FullMask);
    Duplicate(cumulativeHistP0, 0, b16FullMask);
    Duplicate(cumulativeHistP1, 0, b16FullMask);

    uint32_t currCount = count;

    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<uint8_t>(currCount);
        RegTensor<uint8_t> b8Reg;
        ConvertRegToWithShift(srcValue, b8Reg, bitoffset);

        DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(tmpSrc, b8Reg, GetVecLen(), maskReg);
        DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(tmpSrcCopy, b8Reg, GetVecLen(), maskReg);

        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(distributeHistP0, b8Reg,
            maskReg);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::FREQUENCY>(distributeHistP1, b8Reg,
            maskReg);

        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(cumulativeHistP0, b8Reg,
            maskReg);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(cumulativeHistP1, b8Reg,
            maskReg);
    }

    RegTensor<uint16_t> exclusiveSumP0, exclusiveSumP1;
    Sub(exclusiveSumP0, cumulativeHistP0, distributeHistP0, b16FullMask);
    Sub(exclusiveSumP1, cumulativeHistP1, distributeHistP1, b16FullMask);

    DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(exclusiveSum, exclusiveSumP0, GetVecLen() / sizeof(uint16_t),
        b16FullMask);
    DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(exclusiveSum, exclusiveSumP1, GetVecLen() / sizeof(uint16_t),
        b16FullMask);
}

__simd_vf__ inline void EightBitsSort(__local_mem__ uint8_t *srcValueU8, __local_mem__ uint8_t *srcValueU8Back,
    __local_mem__ uint8_t *inputIndexAddr, __local_mem__ uint16_t *inputIndexU16Addr, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen());
    __local_mem__ uint8_t *srcAddr = srcValueU8;
    __local_mem__ uint8_t *srcAddrTmp = srcValueU8;
    __local_mem__ uint8_t *indexAddr = inputIndexAddr;
    __local_mem__ uint8_t *indexAddrTmp = inputIndexAddr;
    RegTensor<uint8_t> srcReg;
    RegTensor<uint8_t> indexReg;
    RegTensor<int8_t> shiftOffset;
    RegTensor<uint8_t> last2BitsMask;
    constexpr uint8_t last2BitMaskVal = 3;
    MaskReg fullMaskB8 = CreateMask<uint8_t>();

    RegTensor<int8_t> bucketIndex;
    Arange(bucketIndex, 0);

    __local_mem__ uint8_t *outputIndex = inputIndexAddr;
    uint32_t tmpCount = count;
    for (uint16_t j = 0; j < repeatTime; ++j) {
        MaskReg reg = UpdateMask<uint8_t>(tmpCount);
        DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(outputIndex, (RegTensor<uint8_t> &)bucketIndex,
            GetVecLen(), reg);
    }

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    ClearSpr<SpecialPurposeReg::AR>();

    Duplicate(shiftOffset, 0, fullMaskB8);
    Duplicate(last2BitsMask, last2BitMaskVal, fullMaskB8);
    srcAddr = srcValueU8;

    // Clear AR to make sure GatherMask not affected by other operations.
    ClearSpr<SpecialPurposeReg::AR>();
    // Makes 4 times 2 bits sort for 8 bits sort.
    for (uint16_t i = 0; i < 4; ++i) {
        indexAddr = inputIndexAddr;
        UnalignReg unalignIndex;
        uint32_t currCount = count;
        for (uint16_t j = 0; j < repeatTime; ++j) {
            MaskReg maskReg = UpdateMask<uint8_t>(currCount);

            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(srcReg, (__local_mem__ uint8_t *&)srcAddr,
                GetVecLen());
            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(indexReg, (__local_mem__ uint8_t *&)indexAddr,
                GetVecLen());
            RegTensor<uint8_t> shift2BitsReg;
            ShiftRight(shift2BitsReg, srcReg, shiftOffset, maskReg);

            And(shift2BitsReg, shift2BitsReg, last2BitsMask, maskReg);
            MaskReg cmpMask0, cmpMask1, cmpMask2, cmpMask3;
            CompareScalar<uint8_t, CMPMODE::EQ>(cmpMask0, shift2BitsReg, 0, maskReg);
            CompareScalar<uint8_t, CMPMODE::EQ>(cmpMask1, shift2BitsReg, 1, maskReg);
            CompareScalar<uint8_t, CMPMODE::EQ>(cmpMask2, shift2BitsReg, 2, maskReg);
            CompareScalar<uint8_t, CMPMODE::EQ>(cmpMask3, shift2BitsReg, 3, maskReg);
            RegTensor<uint8_t> gatherRes0;
            RegTensor<uint8_t> gatherRes1;
            RegTensor<uint8_t> gatherRes2;
            RegTensor<uint8_t> gatherRes3;

            GatherMask<uint8_t, GatherMaskMode::STORE_REG>(gatherRes0, indexReg, cmpMask0);
            DataCopyUnAlign(indexAddrTmp, gatherRes0, unalignIndex);
            GatherMask<uint8_t, GatherMaskMode::STORE_REG>(gatherRes1, indexReg, cmpMask1);
            DataCopyUnAlign(indexAddrTmp, gatherRes1, unalignIndex);
            GatherMask<uint8_t, GatherMaskMode::STORE_REG>(gatherRes2, indexReg, cmpMask2);
            DataCopyUnAlign(indexAddrTmp, gatherRes2, unalignIndex);
            GatherMask<uint8_t, GatherMaskMode::STORE_REG>(gatherRes3, indexReg, cmpMask3);
            DataCopyUnAlign(indexAddrTmp, gatherRes3, unalignIndex);
        }
        DataCopyUnAlignPost(indexAddrTmp, unalignIndex);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        indexAddr = inputIndexAddr;
        srcAddr = srcValueU8;
        srcAddrTmp = srcValueU8;
        __local_mem__ uint8_t *tmpSrcAddr = srcValueU8Back;
        currCount = count;
        for (uint16_t j = 0; j < repeatTime; ++j) {
            MaskReg maskReg = UpdateMask<uint8_t>(currCount);
            RegTensor<uint8_t> selectData, tmpIndexReg;
            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(srcReg, (__local_mem__ uint8_t *&)tmpSrcAddr,
                GetVecLen());
            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(tmpIndexReg, (__local_mem__ uint8_t *&)indexAddr,
                GetVecLen());
            Gather(selectData, srcReg, tmpIndexReg);
            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(srcAddrTmp, selectData, GetVecLen(), maskReg);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        ClearSpr<SpecialPurposeReg::AR>();
        Adds(shiftOffset, shiftOffset, 2, fullMaskB8);
        srcAddr = srcValueU8;
    }

    // Obtain sorted local u8 index and src values.
    __local_mem__ uint8_t *tmpIndexLoad = inputIndexAddr;
    __local_mem__ uint16_t *tmpIndexStore = inputIndexU16Addr;

    MaskReg fullB16Mask = CreateMask<uint16_t>();

    RegTensor<uint8_t> zeroReg;
    Duplicate(zeroReg, 0, fullMaskB8);

    RegTensor<uint16_t> offset;
    Duplicate(offset, 0, fullB16Mask);

    uint32_t indexUpdateCount = count;
    // Sorts per block elements and convert index representing overall offset.
    for (uint16_t j = 0; j < repeatTime; ++j) {
        MaskReg maskReg0 = UpdateMask<uint16_t>(indexUpdateCount);
        MaskReg maskReg1 = UpdateMask<uint16_t>(indexUpdateCount);

        RegTensor<uint8_t> sortedRes;
        DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(sortedRes, (__local_mem__ uint8_t *&)tmpIndexLoad,
            GetVecLen());
        // zero extend to u16
        RegTensor<uint8_t> sortedRes0, sortedRes1;
        Interleave(sortedRes0, sortedRes1, sortedRes, zeroReg);
        // add group offset
        RegTensor<uint16_t> sortedVal0, sortedVal1;
        Add(sortedVal0, (RegTensor<uint16_t> &)sortedRes0, offset, fullB16Mask);
        Add(sortedVal1, (RegTensor<uint16_t> &)sortedRes1, offset, fullB16Mask);
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(tmpIndexStore, sortedVal0, GetVecLen() / sizeof(uint16_t),
            maskReg0);
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(tmpIndexStore, sortedVal1, GetVecLen() / sizeof(uint16_t),
            maskReg1);
        Adds(offset, offset, GetVecLen(), fullB16Mask);
    }
}

__simd_vf__ inline void LocalSort(__local_mem__ uint8_t *srcU8, __local_mem__ uint16_t *tmpIndexU16,
    __local_mem__ uint16_t *exclusiveSumOrigin, __local_mem__ uint32_t *sortedLocalIndex, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen());
    RegTensor<uint8_t> key;
    RegTensor<uint16_t> indexU16Bucket0, indexU16Bucket1;
    MaskReg fullMaskB16 = CreateMask<uint16_t>();
    MaskReg fullMaskB8 = CreateMask<uint8_t>();

    RegTensor<int16_t> bucketLocalIndex0, bucketLocalIndex1;
    Arange(bucketLocalIndex0, 0);
    Arange(bucketLocalIndex1, 128);

    RegTensor<uint16_t> bucketOffset0, bucketOffset1;
    DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(bucketOffset0, (__local_mem__ uint16_t *&)exclusiveSumOrigin,
        GetVecLen() / sizeof(uint16_t));
    DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(bucketOffset1, (__local_mem__ uint16_t *&)exclusiveSumOrigin,
        GetVecLen() / sizeof(uint16_t));

    RegTensor<uint8_t> zeroReg;
    Duplicate(zeroReg, 0, fullMaskB8);

    uint32_t currCount = count;
    uint32_t currCountU8 = count;
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskRegB8 = UpdateMask<uint8_t>(currCountU8);
        DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(key, (__local_mem__ uint8_t *&)srcU8, GetVecLen());
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(indexU16Bucket0, (__local_mem__ uint16_t *&)tmpIndexU16,
            GetVecLen() / sizeof(uint16_t));
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(indexU16Bucket1, (__local_mem__ uint16_t *&)tmpIndexU16,
            GetVecLen() / sizeof(uint16_t));

        RegTensor<uint8_t> bucketOffestLow, bucketOffestHigh;
        DeInterleave(bucketOffestLow, bucketOffestHigh, (RegTensor<uint8_t> &)bucketOffset0,
            (RegTensor<uint8_t> &)bucketOffset1);

        RegTensor<uint16_t> distrubtionHist0, distrubtionHist1;
        RegTensor<uint16_t> cumulativeHist0, cumulativeHist1;
        Duplicate(distrubtionHist0, 0, fullMaskB16);
        Duplicate(distrubtionHist1, 0, fullMaskB16);
        Duplicate(cumulativeHist0, 0, fullMaskB16);
        Duplicate(cumulativeHist1, 0, fullMaskB16);

        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(distrubtionHist0, key,
            maskRegB8);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::FREQUENCY>(distrubtionHist1, key,
            maskRegB8);

        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(cumulativeHist0, key,
            maskRegB8);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(cumulativeHist1, key,
            maskRegB8);

        RegTensor<uint16_t> localOffset0, localOffset1;
        Sub(localOffset0, cumulativeHist0, distrubtionHist0, fullMaskB16);
        Sub(localOffset1, cumulativeHist1, distrubtionHist1, fullMaskB16);

        RegTensor<uint8_t> localOffsetLowbits, localOffsetHighbits;
        DeInterleave(localOffsetLowbits, localOffsetHighbits, (RegTensor<uint8_t> &)localOffset0,
            (RegTensor<uint8_t> &)localOffset1);

        // block_offset[key]
        RegTensor<uint8_t> bucketOffsetKeyLowbits, bucketOffsetKeyHighbits;
        Gather(bucketOffsetKeyLowbits, bucketOffestLow, key);
        Gather(bucketOffsetKeyHighbits, bucketOffestHigh, key);

        RegTensor<uint8_t> bucketOffsetKey0, bucketOffsetKey1;
        Interleave(bucketOffsetKey0, bucketOffsetKey1, bucketOffsetKeyLowbits, bucketOffsetKeyHighbits);

        // localOffset[key]
        // only select low bits, since count should less than 256
        RegTensor<uint8_t> localOffsetWriteKeyLowbits;
        Gather(localOffsetWriteKeyLowbits, localOffsetLowbits, key);

        RegTensor<uint8_t> localOffsetWriteKey0, localOffsetWriteKey1;
        Interleave(localOffsetWriteKey0, localOffsetWriteKey1, localOffsetWriteKeyLowbits, zeroReg);

        // block_rank = block_offset[key] + local_index - localOffset[key]
        RegTensor<uint16_t> bucketRank0, bucketRank1;
        Add(bucketRank0, (RegTensor<uint16_t> &)bucketOffsetKey0, (RegTensor<uint16_t> &)bucketLocalIndex0,
            fullMaskB16);
        Add(bucketRank1, (RegTensor<uint16_t> &)bucketOffsetKey1, (RegTensor<uint16_t> &)bucketLocalIndex1,
            fullMaskB16);

        Sub(bucketRank0, bucketRank0, (RegTensor<uint16_t> &)localOffsetWriteKey0, fullMaskB16);
        Sub(bucketRank1, bucketRank1, (RegTensor<uint16_t> &)localOffsetWriteKey1, fullMaskB16);

        RegTensor<uint32_t> bucketRankU32P0, bucketRankU32P1, bucketRankU32P2, bucketRankU32P3;
        Interleave((RegTensor<uint16_t> &)bucketRankU32P0, (RegTensor<uint16_t> &)bucketRankU32P1, bucketRank0,
            (RegTensor<uint16_t> &)zeroReg);
        Interleave((RegTensor<uint16_t> &)bucketRankU32P2, (RegTensor<uint16_t> &)bucketRankU32P3, bucketRank1,
            (RegTensor<uint16_t> &)zeroReg);

        RegTensor<uint32_t> indexU32P0, indexU32P1, indexU32P2, indexU32P3;
        Interleave((RegTensor<uint16_t> &)indexU32P0, (RegTensor<uint16_t> &)indexU32P1, indexU16Bucket0,
            (RegTensor<uint16_t> &)zeroReg);
        Interleave((RegTensor<uint16_t> &)indexU32P2, (RegTensor<uint16_t> &)indexU32P3, indexU16Bucket1,
            (RegTensor<uint16_t> &)zeroReg);

        // sortedLocalIndex[block_rank] = index
        MaskReg u32Mask0 = UpdateMask<uint32_t>(currCount);
        DataCopyScatter(sortedLocalIndex, indexU32P0, bucketRankU32P0, u32Mask0);
        MaskReg u32Mask1 = UpdateMask<uint32_t>(currCount);
        DataCopyScatter(sortedLocalIndex, indexU32P1, bucketRankU32P1, u32Mask1);
        MaskReg u32Mask2 = UpdateMask<uint32_t>(currCount);
        DataCopyScatter(sortedLocalIndex, indexU32P2, bucketRankU32P2, u32Mask2);
        MaskReg u32Mask3 = UpdateMask<uint32_t>(currCount);
        DataCopyScatter(sortedLocalIndex, indexU32P3, bucketRankU32P3, u32Mask3);

        // accumulate block offset
        Add(bucketOffset0, bucketOffset0, distrubtionHist0, fullMaskB16);
        Add(bucketOffset1, bucketOffset1, distrubtionHist1, fullMaskB16);
    }
}

// Gather B64 elements based on uint32_t offset and store at the correspoding memory.
__simd_callee__ inline void GatherAndStoreB64Elements(RegTensor<uint32_t> &localOffset, MaskReg &maskReg,
    __local_mem__ uint32_t *gatherIdxAddr, __local_mem__ uint32_t *storedAddr)
{
    MaskReg maskLow, maskHigh;
    RegTensor<uint32_t> indexLow;
    RegTensor<uint32_t> indexHigh;
    RegTensor<uint32_t> indexMergedLow;
    RegTensor<uint32_t> indexMergedHigh;
    RegTensor<uint32_t> localOffsetHigh;
    MaskInterleave<uint32_t>(maskLow, maskHigh, maskReg, maskReg);
    Muls(localOffsetHigh, localOffset, 2, maskReg);
    DataCopyGather(indexLow, gatherIdxAddr, localOffsetHigh, maskReg);
    Adds(localOffsetHigh, localOffsetHigh, 1, maskReg);
    DataCopyGather(indexHigh, gatherIdxAddr, localOffsetHigh, maskReg);
    Interleave(indexMergedLow, indexMergedHigh, indexLow, indexHigh);
    DataCopy<uint32_t>(storedAddr, indexMergedLow, maskLow);
    DataCopy<uint32_t>(storedAddr + GetVecLen() / sizeof(uint32_t), indexMergedHigh, maskHigh);
}

template <typename T>
__simd_vf__ inline void UpdateValueAndIndexByLocalIndex(__local_mem__ uint8_t *prevSortedValue,
    __local_mem__ T *prevSortedIndex, __local_mem__ uint32_t *sortedLocalIndex, __local_mem__ uint8_t *currSortedValue,
    __local_mem__ T *currSortedIndex, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen());
    uint32_t currCount = count;
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<uint8_t>(currCount);
        MaskReg maskRegP0, maskRegP1;
        MaskReg maskLowP0, maskHighP0;
        MaskReg maskLowP1, maskHighP1;
        MaskReg tmp = CreateMask<uint16_t, MaskPattern::ALLF>();
        MaskInterleave<uint8_t>(maskRegP0, maskRegP1, maskReg, tmp);
        MaskInterleave<uint16_t>(maskLowP0, maskHighP0, maskRegP0, tmp);
        MaskInterleave<uint16_t>(maskLowP1, maskHighP1, maskRegP1, tmp);

        RegTensor<uint32_t> localOffset0, localOffset1, localOffset2, localOffset3;
        RegTensor<uint16_t> indexP0, indexP1, indexTmp, resB8P0, resB8P1;
        RegTensor<uint8_t> resB8, tmpB8;

        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset0, (__local_mem__ uint32_t *&)sortedLocalIndex,
            GetVecLen() / sizeof(uint32_t));
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset1, (__local_mem__ uint32_t *&)sortedLocalIndex,
            GetVecLen() / sizeof(uint32_t));
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset2, (__local_mem__ uint32_t *&)sortedLocalIndex,
            GetVecLen() / sizeof(uint32_t));
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset3, (__local_mem__ uint32_t *&)sortedLocalIndex,
            GetVecLen() / sizeof(uint32_t));
        DeInterleave(indexP0, indexTmp, (RegTensor<uint16_t> &)localOffset0, (RegTensor<uint16_t> &)localOffset1);
        DeInterleave(indexP1, indexTmp, (RegTensor<uint16_t> &)localOffset2, (RegTensor<uint16_t> &)localOffset3);

        DataCopyGather(resB8P0, prevSortedValue, indexP0, maskRegP0);
        DataCopyGather(resB8P1, prevSortedValue, indexP1, maskRegP1);
        DeInterleave(resB8, tmpB8, (RegTensor<uint8_t> &)resB8P0, (RegTensor<uint8_t> &)resB8P1);

        if constexpr (sizeof(T) == 8) {
            GatherAndStoreB64Elements(localOffset0, maskLowP0, (__local_mem__ uint32_t *)prevSortedIndex,
                (__local_mem__ uint32_t *)(currSortedIndex + 2 * i * GetVecLen() / sizeof(uint16_t)));
            GatherAndStoreB64Elements(localOffset1, maskHighP0, (__local_mem__ uint32_t *)prevSortedIndex,
                (__local_mem__ uint32_t *)(currSortedIndex + 2 * i * GetVecLen() / sizeof(uint16_t) +
                GetVecLen() / sizeof(uint32_t)));
            GatherAndStoreB64Elements(localOffset2, maskLowP1, (__local_mem__ uint32_t *)prevSortedIndex,
                (__local_mem__ uint32_t *)(currSortedIndex + (2 * i + 1) * GetVecLen() / sizeof(uint16_t)));
            GatherAndStoreB64Elements(localOffset3, maskHighP1, (__local_mem__ uint32_t *)prevSortedIndex,
                (__local_mem__ uint32_t *)(currSortedIndex + (2 * i + 1) * GetVecLen() / sizeof(uint16_t) +
                GetVecLen() / sizeof(uint32_t)));
        } else {
            RegTensor<uint32_t> vecIndex0;
            RegTensor<uint32_t> vecIndex1;
            RegTensor<uint32_t> vecIndex2;
            RegTensor<uint32_t> vecIndex3;
            DataCopyGather(vecIndex0, prevSortedIndex, localOffset0, maskLowP0);
            DataCopyGather(vecIndex1, prevSortedIndex, localOffset1, maskHighP0);
            DataCopyGather(vecIndex2, prevSortedIndex, localOffset2, maskLowP1);
            DataCopyGather(vecIndex3, prevSortedIndex, localOffset3, maskHighP1);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedIndex,
                vecIndex0, GetVecLen() / sizeof(uint32_t), maskLowP0);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedIndex,
                vecIndex1, GetVecLen() / sizeof(uint32_t), maskHighP0);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedIndex,
                vecIndex2, GetVecLen() / sizeof(uint32_t), maskLowP1);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedIndex,
                vecIndex3, GetVecLen() / sizeof(uint32_t), maskHighP1);
        }
        DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint8_t *&)currSortedValue, resB8,
            GetVecLen(), maskReg);
    }
}

template <typename T>
__simd_vf__ inline void UpdateValueAndIndexByLocalIndex(__local_mem__ uint16_t *prevSortedValue,
    __local_mem__ T *prevSortedIndex, __local_mem__ uint32_t *sortedLocalIndex, __local_mem__ uint16_t *currSortedValue,
    __local_mem__ T *currSortedIndex, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen() / sizeof(uint16_t));
    uint32_t currCount = count;
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<uint16_t>(currCount);
        MaskReg maskLow, maskHigh;
        MaskReg tmp = CreateMask<uint16_t, MaskPattern::ALLF>();
        MaskInterleave<uint16_t>(maskLow, maskHigh, maskReg, tmp);
        RegTensor<uint32_t> localOffset0, localOffset1;
        RegTensor<uint16_t> indexP0;
        RegTensor<uint16_t> indexP1;
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset0,
            (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset1,
            (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));

        DeInterleave(indexP1, indexP0, (RegTensor<uint16_t> &)localOffset0,
            (RegTensor<uint16_t> &)localOffset1);
        DataCopyGather(indexP0, prevSortedValue, indexP1, maskReg);
        if constexpr (sizeof(T) == 8) {
            GatherAndStoreB64Elements(localOffset0, maskLow, (__local_mem__ uint32_t *)prevSortedIndex,
                (__local_mem__ uint32_t *)(currSortedIndex + i * GetVecLen() / sizeof(uint16_t)));
            GatherAndStoreB64Elements(localOffset1, maskHigh, (__local_mem__ uint32_t *)prevSortedIndex,
                (__local_mem__ uint32_t *)(currSortedIndex + i * GetVecLen() / sizeof(uint16_t) +
                GetVecLen() / sizeof(uint32_t)));
        } else {
            RegTensor<uint32_t> indexU32P0;
            RegTensor<uint32_t> indexU32P1;
            DataCopyGather(indexU32P0, prevSortedIndex, localOffset0, maskLow);
            DataCopyGather(indexU32P1, prevSortedIndex, localOffset1, maskHigh);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedIndex,
                indexU32P0, GetVecLen() / sizeof(uint32_t), maskLow);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedIndex,
                indexU32P1, GetVecLen() / sizeof(uint32_t), maskHigh);
        }
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint16_t *&)currSortedValue, indexP0,
            GetVecLen() / sizeof(uint16_t), maskReg);
    }
}

template <typename T>
__simd_vf__ inline void UpdateValueAndIndexByLocalIndex(__local_mem__ uint32_t *prevSortedValue,
    __local_mem__ T *prevSortedIndex, __local_mem__ uint32_t *sortedLocalIndex, __local_mem__ uint32_t *currSortedValue,
    __local_mem__ T *currSortedIndex, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen() / sizeof(uint32_t));
    uint32_t currCount = count;
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<uint32_t>(currCount);
        RegTensor<uint32_t> localOffset;
        RegTensor<uint32_t> index;
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset,
            (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
        DataCopyGather(index, prevSortedValue, localOffset, maskReg);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedValue, index,
            GetVecLen() / sizeof(uint32_t), maskReg);
        if constexpr (sizeof(T) == 8) {
            GatherAndStoreB64Elements(localOffset, maskReg, (__local_mem__ uint32_t *)prevSortedIndex,
                (__local_mem__ uint32_t *)(currSortedIndex + i * GetVecLen() / sizeof(uint32_t)));
        } else {
            RegTensor<uint32_t> indexU32;
            DataCopyGather(indexU32, prevSortedIndex, localOffset, maskReg);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedIndex, indexU32,
                GetVecLen() / sizeof(uint32_t), maskReg);
        }
    }
}

template <typename T>
__simd_vf__ inline void UpdateValueAndIndexByLocalIndex(__local_mem__ uint64_t *prevSortedValue,
    __local_mem__ T *prevSortedIndex, __local_mem__ uint32_t *sortedLocalIndex, __local_mem__ uint64_t *currSortedValue,
    __local_mem__ T *currSortedIndex, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen() / sizeof(uint64_t));
    uint32_t currCount = count;
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<uint32_t>(currCount);
        RegTensor<uint32_t> localOffset;
        RegTensor<uint32_t> index;
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset,
            (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
        GatherAndStoreB64Elements(localOffset, maskReg, (__local_mem__ uint32_t *)prevSortedValue,
            (__local_mem__ uint32_t *)(currSortedValue + i * GetVecLen() / sizeof(uint32_t)));
        if constexpr (sizeof(T) == 8) {
            GatherAndStoreB64Elements(localOffset, maskReg, (__local_mem__ uint32_t *)prevSortedIndex,
                (__local_mem__ uint32_t *)(currSortedIndex + i * GetVecLen() / sizeof(uint32_t)));
        } else {
            RegTensor<uint32_t> indexU32;
            DataCopyGather(indexU32, prevSortedIndex, localOffset, maskReg);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedIndex, indexU32,
                GetVecLen() / sizeof(uint32_t), maskReg);
        }
    }
}

template <typename T>
__simd_vf__ inline void UpdateValueByLocalIndex(__local_mem__ T *srcValue, __local_mem__ uint32_t *sortedLocalIndex,
    __local_mem__ T *currSortedValue, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen() / sizeof(T));
    uint32_t currCount = count;
    for (uint16_t i = 0; i < repeatTime; i++) {
        if constexpr (sizeof(T) == 1) {
            MaskReg maskReg = UpdateMask<uint8_t>(currCount);
            MaskReg maskRegP0, maskRegP1;
            MaskReg maskLowP0, maskHighP0;
            MaskReg maskLowP1, maskHighP1;
            MaskReg tmp = CreateMask<uint16_t, MaskPattern::ALLF>();
            MaskInterleave<uint8_t>(maskRegP0, maskRegP1, maskReg, tmp);
            MaskInterleave<uint16_t>(maskLowP0, maskHighP0, maskRegP0, tmp);
            MaskInterleave<uint16_t>(maskLowP1, maskHighP1, maskRegP1, tmp);

            RegTensor<uint32_t> localOffset0, localOffset1, localOffset2, localOffset3;
            RegTensor<uint16_t> indexP0, indexP1, indexTmp, resB8P0, resB8P1;
            RegTensor<uint8_t> resB8, tmpB8;

            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset0,
                (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset1,
                (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset2,
                (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset3,
                (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
            DeInterleave(indexP0, indexTmp, (RegTensor<uint16_t> &)localOffset0,
                (RegTensor<uint16_t> &)localOffset1);
            DeInterleave(indexP1, indexTmp, (RegTensor<uint16_t> &)localOffset2,
                (RegTensor<uint16_t> &)localOffset3);
            DataCopyGather(resB8P0, srcValue, indexP0, maskRegP0);
            DataCopyGather(resB8P1, srcValue, indexP1, maskRegP1);
            DeInterleave(resB8, tmpB8, (RegTensor<uint8_t> &)resB8P0, (RegTensor<uint8_t> &)resB8P1);

            DataCopy((__local_mem__ T *)(currSortedValue + i * (GetVecLen() / sizeof(T))), resB8, maskReg);
        } else if constexpr (sizeof(T) == 2) {
            MaskReg maskReg = UpdateMask<uint16_t>(currCount);
            MaskReg maskLow, maskHigh;
            MaskReg tmp = CreateMask<uint16_t, MaskPattern::ALLF>();
            MaskInterleave<uint16_t>(maskLow, maskHigh, maskReg, tmp);

            RegTensor<uint32_t> localOffset0, localOffset1;
            RegTensor<uint16_t> indexP0;
            RegTensor<uint16_t> indexP1;

            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset0,
                (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset1,
                (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
            DeInterleave(indexP1, indexP0, (RegTensor<uint16_t> &)localOffset0,
                (RegTensor<uint16_t> &)localOffset1);
            DataCopyGather(indexP0, srcValue, indexP1, maskReg);

            DataCopy((__local_mem__ T *)(currSortedValue + i * (GetVecLen() / sizeof(T))), indexP0, maskReg);
        } else {
            MaskReg maskReg = UpdateMask<uint32_t>(currCount);
            RegTensor<uint32_t> localOffset;
            RegTensor<uint32_t> indexP0;
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(localOffset,
                (__local_mem__ uint32_t *&)sortedLocalIndex, GetVecLen() / sizeof(uint32_t));
            if constexpr (sizeof(T) == 8) {
                GatherAndStoreB64Elements(localOffset, maskReg, (__local_mem__ uint32_t *)srcValue,
                    (__local_mem__ uint32_t *)(currSortedValue + i * GetVecLen() / sizeof(uint32_t)));
            } else {
                DataCopyGather(indexP0, srcValue, localOffset, maskReg);
                DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>((__local_mem__ uint32_t *&)currSortedValue,
                    indexP0, GetVecLen() / sizeof(uint32_t), maskReg);
            }
        }
    }
}

template <typename T>
__simd_vf__ inline void SaveBufferTo(__local_mem__ T *srcBuffer, __local_mem__ T *dstBuffer, uint32_t count)
{
    uint16_t repeatTime = DivCeil(count, GetVecLen() / sizeof(T));
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<T>(count);
        RegTensor<T> reg;
        DataCopy<T, PostLiteral::POST_MODE_UPDATE>(reg, srcBuffer, GetVecLen() / sizeof(T));
        DataCopy<T, PostLiteral::POST_MODE_UPDATE>(dstBuffer, reg, GetVecLen() / sizeof(T), maskReg);
    }
}

__simd_vf__ inline void SaveBufferTo(__local_mem__ uint64_t *srcBuffer, __local_mem__ uint64_t *dstBuffer,
    uint32_t count)
{
    uint32_t calCount = count << 1;
    uint16_t repeatTime = DivCeil(calCount, GetVecLen() / sizeof(uint32_t));
    __local_mem__ uint32_t *tmpSrc = (__local_mem__ uint32_t *)srcBuffer;
    __local_mem__ uint32_t *tmpDst = (__local_mem__ uint32_t *)dstBuffer;
    for (uint16_t i = 0; i < repeatTime; i++) {
        MaskReg maskReg = UpdateMask<uint32_t>(calCount);
        RegTensor<uint32_t> reg;
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(reg, tmpSrc, GetVecLen() / sizeof(uint32_t));
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(tmpDst, reg, GetVecLen() / sizeof(uint32_t), maskReg);
    }
}

template <typename T> __aicore__ inline void SwapBuffer(__local_mem__ T *&a, __local_mem__ T *&b)
{
    __local_mem__ T *tmp = a;
    a = b;
    b = tmp;
}

template <typename T, bool isDescend> __aicore__ inline constexpr bool CheckDataProcess()
{
    return isDescend || SupportType<T, int8_t, int16_t, int32_t, float, half, bfloat16_t, int64_t>();
}

template <typename T>
__aicore__ inline void SortU8ElementsWithRound(__local_mem__ T *srcValue, __local_mem__ uint8_t *tmpSrc,
    __local_mem__ uint8_t *tmpSrcCopy, __local_mem__ uint16_t *exclusiveSum, __local_mem__ uint8_t *tmpIndexU8,
    __local_mem__ uint16_t *tmpIndexU16, __local_mem__ uint32_t *sortedLocalIndex, const uint32_t count,
    const int32_t round)
{
    GetExclusiveSum(srcValue, tmpSrc, tmpSrcCopy, exclusiveSum, count, round);
    EightBitsSort(tmpSrc, tmpSrcCopy, tmpIndexU8, tmpIndexU16, count);
    LocalSort(tmpSrc, tmpIndexU16, exclusiveSum, sortedLocalIndex, count);
}

__aicore__ inline void ArrangeCommonTmpBuffer(const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t alignCount,
    __local_mem__ uint16_t *&totalExclusiveSum, __local_mem__ uint8_t *&srcU8, __local_mem__ uint16_t *&tmpIndexU16,
    __local_mem__ uint32_t *&sortedLocalIndex, __local_mem__ uint8_t *&srcU8Copy, __local_mem__ uint8_t *&tmpIndexU8)
{
    constexpr uint32_t bucketBuffer = 512;
    __local_mem__ uint8_t *tmp = (__local_mem__ uint8_t *)sharedTmpBuffer.GetPhyAddr();
    totalExclusiveSum = (__local_mem__ uint16_t *)tmp;
    srcU8 = (__local_mem__ uint8_t *)(tmp + bucketBuffer);
    tmpIndexU16 = (__local_mem__ uint16_t *)(srcU8 + alignCount);
    sortedLocalIndex = (__local_mem__ uint32_t *)((__local_mem__ uint8_t *)tmpIndexU16 + sizeof(uint16_t) * alignCount);
    // Tmp u8 value and index, could be reused with sortedLocalIndex.
    srcU8Copy = (__local_mem__ uint8_t *)sortedLocalIndex;
    tmpIndexU8 = srcU8Copy + alignCount;
}

template <typename T>
__aicore__ inline void DescendProcess(__ubuf__ T* dst, uint32_t count)
{
    constexpr uint32_t halfTypeMask = 0x80008000;
    constexpr uint32_t floatTypeMask = 0x80000000;
    MicroAPI::RegTensor<T> vreg;
    MicroAPI::RegTensor<uint32_t> vDupReg;
    MicroAPI::MaskReg mask;
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
    if constexpr (IsSameType<T, float>::value) {
        MicroAPI::Duplicate(vDupReg, floatTypeMask);
    } else {
        MicroAPI::Duplicate(vDupReg, halfTypeMask);
    }
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T>(count);
        MicroAPI::DataCopy(vreg, dst + i * oneRepElm);
        MicroAPI::Xor((MicroAPI::RegTensor<uint32_t>&)vreg, (MicroAPI::RegTensor<uint32_t>&)vreg, vDupReg, mask);
        MicroAPI::DataCopy(dst + i * oneRepElm, vreg, mask);
    }
}

template <typename T, bool isReuseSource, const SortConfig &config>
__aicore__ inline void SortImpl(LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    constexpr SortType sortType = config.type;
    constexpr bool isDescend = config.isDescend;
    ASCENDC_ASSERT(
        (srcLocal.GetSize() * sizeof(T) % ONE_BLK_SIZE == 0 && dstLocal.GetSize() * sizeof(T) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcTensor and dstTensor size must be 32B aligned"); });
    ASCENDC_ASSERT((count > 0 && count <= srcLocal.GetSize() && count <= dstLocal.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR,
            "count must be greater than 0 and less than or equals to srcTensor and dstTensor Size");
    });
    if constexpr (sortType == SortType::MERGE_SORT) {
        static_assert(SupportType<T, half, float>(),
            "Advanced Sort API MERGE_MODE only supports data value with half/float types.");
        ASCENDC_ASSERT((count % singleSortElementCountC310 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "calCount should be align to 32 in MERGE_MODE!"); });
        ASCENDC_ASSERT((dstLocal.GetSize() * sizeof(T) >= count * 8),
            { KERNEL_LOG(KERNEL_ERROR, "dstLocal size should be greater equal to 8 * count!"); });
        const int32_t repeatTime = static_cast<int32_t>(count / singleSortElementCountC310);
        if constexpr (!isDescend) {
            VF_CALL<DescendProcess<T>>((__ubuf__ T*)srcLocal.GetPhyAddr(), count);
        }
        Sort32(dstLocal, srcLocal, (const LocalTensor<uint32_t>&)srcLocal, repeatTime);
        DoFullSort(dstLocal, (const LocalTensor<T>&)sharedTmpBuffer, (const LocalTensor<uint32_t>&)sharedTmpBuffer,
            (LocalTensor<T>&)sharedTmpBuffer, repeatTime);
        Extract(dstLocal, (const LocalTensor<uint32_t>&)sharedTmpBuffer, dstLocal, repeatTime);
        if constexpr (!isDescend) {
            VF_CALL<DescendProcess<T>>((__ubuf__ T*)dstLocal.GetPhyAddr(), count);
        }
     } else {
        static_assert(SupportBytes<T, 2, 4, 8>() || SupportType<T, uint8_t, int8_t>(),
            "Advanced Sort API RADIX_MODE only supports data value with "
            "int8_t/uint8_t/half/bfloat16_t/int16_t/uint16_t/float/int32_t/uint32_t/int64_t/uint64_t types.");
        constexpr bool needProcess = CheckDataProcess<T, isDescend>();
        using ConvType = typename ::AscendC::Internal::ExtractTypeBySize<sizeof(T)>::T;
        uint32_t alignCount = AlignUp(count, ONE_BLK_SIZE);
        __local_mem__ ConvType *src = (__local_mem__ ConvType *)srcLocal.GetPhyAddr();
        __local_mem__ ConvType *dst = (__local_mem__ ConvType *)dstLocal.GetPhyAddr();
        __local_mem__ uint16_t *totalExclusiveSum, *tmpIndexU16;
        __local_mem__ uint8_t *srcU8, *srcU8Copy, *tmpIndexU8;
        __local_mem__ uint32_t *sortedLocalIndex;
        ArrangeCommonTmpBuffer(sharedTmpBuffer, alignCount, totalExclusiveSum, srcU8, tmpIndexU16, sortedLocalIndex,
            srcU8Copy, tmpIndexU8);
        __local_mem__ ConvType *srcForNextRound =
            (__local_mem__ ConvType *)((__local_mem__ uint8_t *)sortedLocalIndex + sizeof(uint32_t) * alignCount);
        __local_mem__ ConvType *currSortedValue = srcForNextRound;
        __local_mem__ ConvType *prevSortedValue = dst;
        __local_mem__ ConvType *initSrcValue = src;

        if constexpr (sizeof(T) == 1 && needProcess) {
            PreProcess<T, ConvType, isDescend>(src, dst, count);
            initSrcValue = dst;
        } else {
            if constexpr (needProcess) {
                PreProcess<T, ConvType, isDescend>(src, dst, count);
                initSrcValue = dst;
                if constexpr (isReuseSource) {
                    currSortedValue = src;
                }
            } else if constexpr (isReuseSource) {
                SaveBufferTo(src, dst, count);
                initSrcValue = dst;
                currSortedValue = src;
            }
        }

        if constexpr (sizeof(T) == 1) {
            SortU8ElementsWithRound(initSrcValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                sortedLocalIndex, count, 0);
            UpdateValueByLocalIndex(src, sortedLocalIndex, dst, count);
        } else {
            SortU8ElementsWithRound(initSrcValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                sortedLocalIndex, count, 0);
            UpdateValueByLocalIndex(initSrcValue, sortedLocalIndex, currSortedValue, count);

            SwapBuffer(currSortedValue, prevSortedValue);

            SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                sortedLocalIndex, count, 1);
            UpdateValueByLocalIndex(prevSortedValue, sortedLocalIndex, currSortedValue, count);

            if constexpr (sizeof(T) >= 4) {
                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 2);
                UpdateValueByLocalIndex(prevSortedValue, sortedLocalIndex, currSortedValue, count);

                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 3);
                UpdateValueByLocalIndex(prevSortedValue, sortedLocalIndex, currSortedValue, count);
            }

            if constexpr (sizeof(T) == 8) {
                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 4);
                UpdateValueByLocalIndex(prevSortedValue, sortedLocalIndex, currSortedValue, count);

                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 5);
                UpdateValueByLocalIndex(prevSortedValue, sortedLocalIndex, currSortedValue, count);

                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 6);
                UpdateValueByLocalIndex(prevSortedValue, sortedLocalIndex, currSortedValue, count);

                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 7);
                UpdateValueByLocalIndex(prevSortedValue, sortedLocalIndex, currSortedValue, count);
            }

            if constexpr (needProcess) {
                PostProcess<T, ConvType, isDescend>(dst, dst, count);
            }
        }
    }
}

__aicore__ inline void GenSrcIndex(__ubuf__ uint32_t* dst, uint32_t count)
{
    MicroAPI::RegTensor<uint32_t> vreg;
    MicroAPI::MaskReg mask;
    constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(uint32_t));
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
    MicroAPI::Arange((MicroAPI::RegTensor<int32_t>&)vreg, 0u);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<uint32_t>(count);
        MicroAPI::DataCopy(dst + i * oneRepElm, vreg, mask);
        MicroAPI::Adds(vreg, vreg, oneRepElm, mask);
    }
}

template <typename T, bool isReuseSource, const SortConfig &config>
__aicore__ inline void SortImpl(LocalTensor<T> &dstLocal, LocalTensor<uint32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    constexpr SortType sortType = config.type; 
    constexpr bool isDescend = config.isDescend;
    ASCENDC_ASSERT((srcLocal.GetSize() * sizeof(T) % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "srcTensor size must be 32B aligned"); });
    ASCENDC_ASSERT((dstIndexLocal.GetSize() * sizeof(uint32_t) % ONE_BLK_SIZE == 0 &&
        dstLocal.GetSize() * sizeof(T) % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "dstTensor and dstIndexTensor size must be 32B aligned"); });
    ASCENDC_ASSERT(
        (count > 0 && count <= srcLocal.GetSize() && count <= dstLocal.GetSize() && count <= dstIndexLocal.GetSize()), {
            KERNEL_LOG(KERNEL_ERROR,
                "count must be greater than 0 and less than or equals to srcTensor and dstTensor Size");
        });
    if constexpr (sortType == SortType::MERGE_SORT) {
        static_assert(SupportType<T, half, float>(),
            "Advanced Sort API MERGE_MODE only supports data value with half/float types.");
        ASCENDC_ASSERT((count % singleSortElementCountC310 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "calCount should be align to 32 in MERGE_MODE!"); });
        ASCENDC_ASSERT((dstLocal.GetSize() * sizeof(T) >= count * 8),
            { KERNEL_LOG(KERNEL_ERROR, "dstLocal size should be greater equal to 8 * count!"); });
        const int32_t repeatTime = static_cast<int32_t>(count / singleSortElementCountC310);
        if constexpr (!isDescend) {
            VF_CALL<DescendProcess<T>>((__ubuf__ T*)srcLocal.GetPhyAddr(), count);
        }
        VF_CALL<GenSrcIndex>((__ubuf__ uint32_t*)sharedTmpBuffer.GetPhyAddr(), count);
        Sort32(dstLocal, srcLocal, (const LocalTensor<uint32_t>&)sharedTmpBuffer, repeatTime);
        DoFullSort(dstLocal, (const LocalTensor<T>&)sharedTmpBuffer, (const LocalTensor<uint32_t>&)sharedTmpBuffer,
            (LocalTensor<T>&)sharedTmpBuffer, repeatTime);
        Extract(dstLocal, dstIndexLocal, dstLocal, repeatTime);
        if constexpr (!isDescend) {
            VF_CALL<DescendProcess<T>>((__ubuf__ T*)dstLocal.GetPhyAddr(), count);
        }
    } else {
        static_assert(SupportBytes<T, 2, 4, 8>() || SupportType<T, uint8_t, int8_t>(),
            "Advanced Sort API RADIX_MODE only supports data value with "
            "int8_t/uint8_t/half/bfloat16_t/int16_t/uint16_t/float/int32_t/uint32_t/int64_t/uint64_t types.");
        constexpr bool needProcess = CheckDataProcess<T, isDescend>();
        using ConvType = typename ::AscendC::Internal::ExtractTypeBySize<sizeof(T)>::T;
        uint32_t alignCount = AlignUp(count, ONE_BLK_SIZE);
        __local_mem__ ConvType *src = (__local_mem__ ConvType *)srcLocal.GetPhyAddr();
        __local_mem__ ConvType *dst = (__local_mem__ ConvType *)dstLocal.GetPhyAddr();
        __local_mem__ uint32_t *dstIndex = (__local_mem__ uint32_t *)dstIndexLocal.GetPhyAddr();
        __local_mem__ uint16_t *totalExclusiveSum, *tmpIndexU16;
        __local_mem__ uint8_t *srcU8, *srcU8Copy, *tmpIndexU8;
        __local_mem__ uint32_t *sortedLocalIndex;
        ArrangeCommonTmpBuffer(sharedTmpBuffer, alignCount, totalExclusiveSum, srcU8, tmpIndexU16, sortedLocalIndex,
            srcU8Copy, tmpIndexU8);
        __local_mem__ uint32_t *sortedIndexNextRound =
            (__local_mem__ uint32_t *)((__local_mem__ uint8_t *)sortedLocalIndex + sizeof(uint32_t) * alignCount);
        __local_mem__ ConvType *srcForNextRound =
            (__local_mem__ ConvType *)((__local_mem__ uint8_t *)sortedIndexNextRound + sizeof(uint32_t) * alignCount);
        __local_mem__ uint32_t *currSortedIndex = sortedIndexNextRound;
        __local_mem__ ConvType *currSortedValue = srcForNextRound;
        __local_mem__ uint32_t *prevSortedIndex = dstIndex;
        __local_mem__ ConvType *prevSortedValue = dst;
        __local_mem__ ConvType *initSrcValue = src;

        if constexpr (sizeof(T) == 1 && needProcess) {
            PreProcess<T, ConvType, isDescend>(src, dst, count);
            initSrcValue = dst;
        } else {
            if constexpr (needProcess) {
                PreProcess<T, ConvType, isDescend>(src, dst, count);
                initSrcValue = dst;
                if constexpr (isReuseSource) {
                    currSortedValue = src;
                }
            } else if constexpr (isReuseSource) {
                SaveBufferTo(src, dst, count);
                initSrcValue = dst;
                currSortedValue = src;
            }
        }

        if constexpr (sizeof(T) == 1) {
            SortU8ElementsWithRound(initSrcValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16, dstIndex,
                count, 0);
            UpdateValueByLocalIndex(src, dstIndex, dst, count);
        } else {
            SortU8ElementsWithRound(initSrcValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                currSortedIndex, count, 0);
            UpdateValueByLocalIndex(initSrcValue, currSortedIndex, currSortedValue, count);

            SwapBuffer(currSortedIndex, prevSortedIndex);
            SwapBuffer(currSortedValue, prevSortedValue);

            SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                sortedLocalIndex, count, 1);
            UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedIndex, sortedLocalIndex, currSortedValue,
                currSortedIndex, count);

            if constexpr (sizeof(T) >= 4) {
                SwapBuffer(currSortedIndex, prevSortedIndex);
                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 2);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedIndex, sortedLocalIndex, currSortedValue,
                    currSortedIndex, count);

                SwapBuffer(currSortedIndex, prevSortedIndex);
                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 3);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedIndex, sortedLocalIndex, currSortedValue,
                    currSortedIndex, count);
            }

            if constexpr (sizeof(T) == 8) {
                SwapBuffer(currSortedIndex, prevSortedIndex);
                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 4);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedIndex, sortedLocalIndex, currSortedValue,
                    currSortedIndex, count);

                SwapBuffer(currSortedIndex, prevSortedIndex);
                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 5);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedIndex, sortedLocalIndex, currSortedValue,
                    currSortedIndex, count);

                SwapBuffer(currSortedIndex, prevSortedIndex);
                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 6);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedIndex, sortedLocalIndex, currSortedValue,
                    currSortedIndex, count);

                SwapBuffer(currSortedIndex, prevSortedIndex);
                SwapBuffer(currSortedValue, prevSortedValue);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 7);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedIndex, sortedLocalIndex, currSortedValue,
                    currSortedIndex, count);
            }

            if constexpr (needProcess) {
                PostProcess<T, ConvType, isDescend>(dst, dst, count);
            }
        }
    }
}

template <typename T, typename U, bool isReuseSource, const SortConfig &config>
__aicore__ inline void SortImpl(const LocalTensor<T> &dstLocal, const LocalTensor<U> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<U> &srcIndexLocal, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t count)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    constexpr SortType sortType = config.type;
    constexpr bool isDescend = config.isDescend;
    ASCENDC_ASSERT(
        (srcLocal.GetSize() * sizeof(T) % ONE_BLK_SIZE == 0 && srcIndexLocal.GetSize() * sizeof(U) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcTensor and srcIndexTensor size must be 32B aligned"); });
    ASCENDC_ASSERT(
        (dstIndexLocal.GetSize() * sizeof(U) % ONE_BLK_SIZE == 0 && dstLocal.GetSize() * sizeof(T) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "dstTensor and dstIndexTensor size must be 32B aligned"); });
    ASCENDC_ASSERT((count > 0 && count <= srcLocal.GetSize() && count <= srcIndexLocal.GetSize() &&
        count <= dstLocal.GetSize() && count <= dstIndexLocal.GetSize()),
                   {
                       KERNEL_LOG(KERNEL_ERROR,
                           "count must be greater than 0 and less than or equals to all src and dst tensors Size");
                   });

    if constexpr (sortType == SortType::MERGE_SORT) {
        static_assert(SupportType<T, half, float>(),
            "Advanced Sort API MERGE_MODE only supports data value with half/float types.");
        static_assert(SupportType<U, uint32_t>(),
            "Advanced Sort API MERGE_MODE only supports src index with uint32_t types.");
        ASCENDC_ASSERT((count % singleSortElementCountC310 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "calCount should be align to 32 in MERGE_MODE!"); });
        ASCENDC_ASSERT((dstLocal.GetSize() * sizeof(T) >= count * 8),
            { KERNEL_LOG(KERNEL_ERROR, "dstLocal size should be greater equal to 8 * count!"); });
        const int32_t repeatTime = static_cast<int32_t>(count / singleSortElementCountC310);
        if constexpr (!isDescend) {
            VF_CALL<DescendProcess<T>>((__ubuf__ T*)srcLocal.GetPhyAddr(), count);
        }
        Sort32(dstLocal, srcLocal, srcIndexLocal, repeatTime);
        DoFullSort(dstLocal, (const LocalTensor<T>&)sharedTmpBuffer, (const LocalTensor<uint32_t>&)sharedTmpBuffer,
            (LocalTensor<T>&)sharedTmpBuffer, repeatTime);
        Extract(dstLocal, dstIndexLocal, dstLocal, repeatTime);
        if constexpr (!isDescend) {
            VF_CALL<DescendProcess<T>>((__ubuf__ T*)dstLocal.GetPhyAddr(), count);
        }
    } else {
        static_assert(SupportBytes<T, 2, 4, 8>() || SupportType<T, uint8_t, int8_t>(),
            "Advanced Sort API RADIX_MODE only supports data value with "
            "int8_t/uint8_t/half/bfloat16_t/int16_t/uint16_t/float/int32_t/uint32_t/int64_t/uint64_t types.");
        static_assert(SupportType<U, int32_t, uint32_t, int64_t, uint64_t>(),
            "Advanced Sort API RADIX_MODE only supports src index with int32_t/uint32_t/int64_t/uint64_t types.");
        constexpr bool needProcess = CheckDataProcess<T, isDescend>();
        using ConvTypeT = typename ::AscendC::Internal::ExtractTypeBySize<sizeof(T)>::T;
        using ConvTypeU = typename ::AscendC::Internal::ExtractTypeBySize<sizeof(U)>::T;
        uint32_t alignCount = AlignUp(count, ONE_BLK_SIZE);
        __local_mem__ ConvTypeT *src = (__local_mem__ ConvTypeT *)srcLocal.GetPhyAddr();
        __local_mem__ ConvTypeU *srcIndex = (__local_mem__ ConvTypeU *)srcIndexLocal.GetPhyAddr();
        __local_mem__ ConvTypeT *dst = (__local_mem__ ConvTypeT *)dstLocal.GetPhyAddr();
        __local_mem__ ConvTypeU *dstIndex = (__local_mem__ ConvTypeU *)dstIndexLocal.GetPhyAddr();
        __local_mem__ uint16_t *totalExclusiveSum, *tmpIndexU16;
        __local_mem__ uint8_t *srcU8, *srcU8Copy, *tmpIndexU8;
        __local_mem__ uint32_t *sortedLocalIndex;
        ArrangeCommonTmpBuffer(sharedTmpBuffer, alignCount, totalExclusiveSum, srcU8, tmpIndexU16, sortedLocalIndex,
            srcU8Copy, tmpIndexU8);
        __local_mem__ ConvTypeU *currSortedDstIndex =
            (__local_mem__ ConvTypeU *)((__local_mem__ uint8_t *)sortedLocalIndex + sizeof(uint32_t) * alignCount);
        __local_mem__ ConvTypeT *srcForNextRound =
            (__local_mem__ ConvTypeT *)((__local_mem__ uint8_t *)currSortedDstIndex + sizeof(ConvTypeU) * alignCount);
        __local_mem__ ConvTypeU *prevSortedDstIndex = dstIndex;
        __local_mem__ ConvTypeT *currSortedValue = srcForNextRound;
        __local_mem__ ConvTypeT *prevSortedValue = dst;
        __local_mem__ ConvTypeT *initSrcValue = src;
        __local_mem__ ConvTypeU *initSrcValueIndex = srcIndex;

        if constexpr (sizeof(T) == 1 && needProcess) {
            PreProcess<T, ConvTypeT, isDescend>(src, dst, count);
            initSrcValue = dst;
        } else {
            if constexpr (needProcess) {
                PreProcess<T, ConvTypeT, isDescend>(src, dst, count);
                initSrcValue = dst;
                if constexpr (isReuseSource) {
                    SaveBufferTo(srcIndex, dstIndex, count);
                    currSortedValue = src;
                    currSortedDstIndex = srcIndex;
                    initSrcValueIndex = dstIndex;
                }
            } else if constexpr (isReuseSource) {
                SaveBufferTo(src, dst, count);
                SaveBufferTo(srcIndex, dstIndex, count);
                initSrcValue = dst;
                initSrcValueIndex = dstIndex;
                currSortedValue = src;
                currSortedDstIndex = srcIndex;
            }
        }

        if constexpr (sizeof(T) == 1) {
            SortU8ElementsWithRound(initSrcValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                sortedLocalIndex, count, 0);
            UpdateValueAndIndexByLocalIndex(src, srcIndex, sortedLocalIndex, dst, dstIndex, count);
        } else {
            SortU8ElementsWithRound(initSrcValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                sortedLocalIndex, count, 0);
            UpdateValueAndIndexByLocalIndex(initSrcValue, initSrcValueIndex, sortedLocalIndex, currSortedValue, currSortedDstIndex,
                count);

            SwapBuffer(currSortedValue, prevSortedValue);
            SwapBuffer(currSortedDstIndex, prevSortedDstIndex);

            SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                sortedLocalIndex, count, 1);
            UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedDstIndex, sortedLocalIndex, currSortedValue, 
                currSortedDstIndex, count);

            if constexpr (sizeof(T) >= 4) {
                SwapBuffer(currSortedValue, prevSortedValue);
                SwapBuffer(currSortedDstIndex, prevSortedDstIndex);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 2);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedDstIndex, sortedLocalIndex, currSortedValue, 
                    currSortedDstIndex, count);

                SwapBuffer(currSortedValue, prevSortedValue);
                SwapBuffer(currSortedDstIndex, prevSortedDstIndex);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 3);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedDstIndex, sortedLocalIndex, currSortedValue, 
                    currSortedDstIndex, count);
            }

            if constexpr (sizeof(T) == 8) {
                SwapBuffer(currSortedValue, prevSortedValue);
                SwapBuffer(currSortedDstIndex, prevSortedDstIndex);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 4);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedDstIndex, sortedLocalIndex, currSortedValue, 
                    currSortedDstIndex, count);

                SwapBuffer(currSortedValue, prevSortedValue);
                SwapBuffer(currSortedDstIndex, prevSortedDstIndex);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 5);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedDstIndex, sortedLocalIndex, currSortedValue, 
                    currSortedDstIndex, count);

                SwapBuffer(currSortedValue, prevSortedValue);
                SwapBuffer(currSortedDstIndex, prevSortedDstIndex);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 6);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedDstIndex, sortedLocalIndex, currSortedValue, 
                    currSortedDstIndex, count);

                SwapBuffer(currSortedValue, prevSortedValue);
                SwapBuffer(currSortedDstIndex, prevSortedDstIndex);

                SortU8ElementsWithRound(prevSortedValue, srcU8, srcU8Copy, totalExclusiveSum, tmpIndexU8, tmpIndexU16,
                    sortedLocalIndex, count, 7);
                UpdateValueAndIndexByLocalIndex(prevSortedValue, prevSortedDstIndex, sortedLocalIndex, currSortedValue, 
                    currSortedDstIndex, count);
            }

            if constexpr (needProcess) {
                PostProcess<T, ConvTypeT, isDescend>(dst, dst, count);
            }
        }
    }
}

template <typename T, bool isReuseSource, const SortConfig &config>
__aicore__ inline void SortImpl(LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    // Using the Stack Space to Allocate tmpBuffer
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    SortImpl<T, isReuseSource, config>(dstLocal, srcLocal, sharedTmpBuffer, calCount);
}

template <typename T, bool isReuseSource, const SortConfig &config>
__aicore__ inline void SortImpl(LocalTensor<T> &dstLocal, LocalTensor<uint32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    // Using the Stack Space to Allocate tmpBuffer
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    SortImpl<T, isReuseSource, config>(dstLocal, dstIndexLocal, srcLocal, sharedTmpBuffer, calCount);
}

template <typename T, typename U, bool isReuseSource, const SortConfig &config>
__aicore__ inline void SortImpl(const LocalTensor<T> &dstLocal, const LocalTensor<U> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<U> &srcIndexLocal, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    // Using the Stack Space to Allocate tmpBuffer
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    SortImpl<T, U, isReuseSource, config>(dstLocal, dstIndexLocal, srcLocal, srcIndexLocal, sharedTmpBuffer, calCount);
}
} // namespace internal
} // namespace MicroAPI
};     // namespace AscendC
#endif // IMPL_SORT_SORT_SORT_IMPL_H
