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
 * \file kernel_operator_data_copy_impl.h
 * \brief AscendC l310 support data copy api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H

#include "kernel_operator_common_impl.h"
#include "kernel_operator_vec_duplicate_impl.h"
#include "kernel_struct_unary.h"

namespace AscendC {

constexpr uint8_t BYTE_32_ALIGN = 32;          // in unit of 32 bytes

// GM -> L1: copy_gm_to_cbuf_align
template <typename T, bool isDataCopyPad = false>
__aicore__ inline void CopyGmToCbufAlign(__cbuf__ T* dst, __gm__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint8_t leftPadding, const uint8_t rightPadding,
    const uint32_t srcStride, const uint32_t dstStride)
{
    uint32_t burstLength = isDataCopyPad ? blockLen : (blockLen * BYTE_32_ALIGN);
    uint64_t actSrcStride = isDataCopyPad ? srcStride : (srcStride * BYTE_32_ALIGN);
    uint32_t actDstStride = dstStride;

    if constexpr (sizeof(T) == B64_BYTE_SIZE) {
        copy_gm_to_cbuf_align((__cbuf__ uint32_t*)dst, (__gm__ uint32_t*)src, 0, blockCount, burstLength,
            leftPadding * 2, rightPadding * 2, actSrcStride, actDstStride);
    }  else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_gm_to_cbuf_align((__cbuf__ uint32_t*)dst, (__gm__ uint32_t*)src, 0, blockCount, burstLength,
            leftPadding, rightPadding, actSrcStride, actDstStride);
    }  else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_gm_to_cbuf_align((__cbuf__ uint16_t*)dst, (__gm__ uint16_t*)src, 0, blockCount, burstLength,
            leftPadding, rightPadding, actSrcStride, actDstStride);
    }  else if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        copy_gm_to_cbuf_align((__cbuf__ uint8_t*)dst, (__gm__ uint8_t*)src, 0, blockCount, burstLength,
            leftPadding, rightPadding, actSrcStride, actDstStride);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data type of copy from gm to cbuf on current device"); });
    }
}

// L1 -> GM: copy_cbuf_to_gm_align
template <typename T>
__aicore__ inline void CopyCbufToGmAlign(__gm__ T* dst, __cbuf__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint32_t srcStride, const uint32_t dstStride)
{
    uint32_t burstLength = blockLen * BYTE_32_ALIGN;
    uint64_t actSrcStride = srcStride;
    uint32_t actDstStride = dstStride * BYTE_32_ALIGN;
    copy_cbuf_to_gm_align((__gm__ T*)dst, (__cbuf__ T*)src, 0, blockCount, burstLength,
        actSrcStride, actDstStride);
}

// only support VecCore    PIPE_MTE2
// GM -> UB
template <typename T, bool isDataCopyPad = false>
__aicore__ inline void CopyGmToUbufAlign(__ubuf__ T* dst, __gm__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint8_t leftPaddingCount, const uint8_t rightPaddingCount, const uint32_t srcStride,
    const uint32_t dstStride)
{
    ASCENDC_ASSERT((sizeof(T) == B8_BYTE_SIZE || sizeof(T) == B16_BYTE_SIZE || sizeof(T) == B32_BYTE_SIZE || sizeof(T) == B64_BYTE_SIZE),
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported data type of copy from gm to ubuf on this version"); });

    uint32_t burstLength = isDataCopyPad ? blockLen : (blockLen * BYTE_32_ALIGN);
    uint32_t actSrcStride = isDataCopyPad ? srcStride : (srcStride * BYTE_32_ALIGN);
    uint32_t actDstStride = dstStride;
    uint8_t leftPaddingCountT = leftPaddingCount * sizeof(T);
    uint8_t rightPaddingCountT = rightPaddingCount * sizeof(T);
    copy_gm_to_ubuf_align((__ubuf__ uint8_t*)dst, (__gm__ uint8_t*)src, 0, blockCount, burstLength,
            leftPaddingCountT, rightPaddingCountT, actSrcStride, actDstStride);
}


// only support VecCore   PIPE_MTE3
// UB -> GM
template <typename T>
__aicore__ inline void CopyUbufToGmAlign(__gm__ T* dst, __ubuf__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint32_t srcStride, const uint32_t dstStride)
{
    uint32_t burstLength = blockLen * BYTE_32_ALIGN;
    uint32_t actSrcStride = srcStride;
    uint32_t actDstStride = dstStride * BYTE_32_ALIGN;
    copy_ubuf_to_gm_align(dst, src, 0, blockCount, burstLength, (uint32_t)actSrcStride, (uint32_t)actDstStride);
}

/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void DataCopyGM2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    CopyGmToUbufAlign<T, false>(dst, src, intriParams.blockCount, intriParams.blockLen, 0, 0, intriParams.srcStride,
        intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyGM2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    CopyGmToCbufAlign<T, false>(dst, src, intriParams.blockCount, intriParams.blockLen, 0, 0,
        intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
#ifdef ASCENDC_CPU_DEBUG
    DataCopyWithAtomic(dst, src, intriParams);
#endif // ASCENDC_CPU_DEBUG
    CopyUbufToGmAlign(dst, src, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride,
        intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyUB2UBImpl(__ubuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    copy_ubuf_to_ubuf((__ubuf__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount, intriParams.blockLen,
        intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    copy_ubuf_to_cbuf((__cbuf__ void *)dst, (__ubuf__ void *)src, 0, intriParams.blockCount,
     intriParams.blockLen, intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyL12UBImpl(__ubuf__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    copy_cbuf_to_ubuf((__ubuf__ void *)dst, (__cbuf__ void *)src, 0, intriParams.blockCount,
     intriParams.blockLen, intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyL12BTImpl(const uint64_t dst, __cbuf__ T* src, const uint16_t isenableConv,
    const DataCopyParams &intriParams)
{
    if constexpr(std::is_same<T, float>::value || std::is_same<T, int32_t>::value || std::is_same<T, half>::value) {
        // the burst length, destination gap size must be even.
        uint16_t blockLenAlign = AlignUp(intriParams.blockLen, 2);
        uint16_t dstStrideAlign = AlignUp(intriParams.dstStride, 2);
        copy_cbuf_to_bt(dst, src, (bool)isenableConv, intriParams.blockCount, blockLenAlign,
            intriParams.srcStride, dstStrideAlign);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data type of copy from cbuf to bt on this version"); });
    }
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImplAlign32B(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    uint16_t ndNum = intriParams.ndNum;                         // nd矩阵数据
    uint16_t nValue = intriParams.nValue;                       // nd矩阵行数
    uint16_t dValue = intriParams.dValue;                       // nd矩阵列数，要求32B对齐
    uint16_t srcNdMatrixStride = intriParams.srcNdMatrixStride; // 源操作数相邻nd矩阵起始地址间的偏移，要求32B对齐
    uint16_t srcDValue = intriParams.srcDValue;                 // 源操作数同一nd矩阵的相邻行起始地址间的偏移 (切分场景，行地址不连续)，要求32B对齐
    uint16_t dstNzC0Stride = intriParams.dstNzC0Stride;         // 目的nz矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移, 单位：block
    uint16_t dstNzNStride = intriParams.dstNzNStride;           // 目的nz矩阵中，Z型矩阵相邻行起始地址之间的偏移，单位：block
    uint16_t dstNzMatrixStride = intriParams.dstNzMatrixStride; // 目的nz矩阵中，相邻nz矩阵起始地址间的偏移，要求32B对齐

    ASCENDC_DEBUG_ASSERT(((dValue * sizeof(T)) % BYTE_32_ALIGN == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(((srcNdMatrixStride * sizeof(T)) % BYTE_32_ALIGN == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "srcNdMatrixStride should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(((srcDValue * sizeof(T)) % BYTE_32_ALIGN == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(((dstNzMatrixStride * sizeof(T)) % BYTE_32_ALIGN == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));

    uint32_t elementsPerBlock = ONE_BLK_SIZE / sizeof(T);
    uint32_t nBurst = dValue * sizeof(T) / ONE_BLK_SIZE; // 控制连续读的burst个数，当前为连续读一行
    for (uint32_t ndIdx = 0; ndIdx < ndNum; ++ndIdx) {
        for (int i = 0; i < nValue;  ++i) {
            uint32_t offsetDst = ndIdx * dstNzMatrixStride + i * dstNzNStride * elementsPerBlock; // 跳着写时，每次遍历的起始位置
            uint32_t offsetSrc = ndIdx * srcNdMatrixStride + i * srcDValue; // 连续读时，每次遍历的起始位置
            copy_gm_to_cbuf_align(dst + offsetDst, src + offsetSrc, 0, nBurst, ONE_BLK_SIZE, 0, 0, 0, (dstNzC0Stride - 1));
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyL12FBImpl(__fbuf__ T* dst, __cbuf__ T* src, const DataCopyParams &intriParams)
{
    // 该API属于ISASI，并且不支持多次分配内存，只能进行一次AllocTensor操作，然后搬运所有数据到FB。因此直接设置dst = 0确保数据从0开始排布
    dst = 0;

    // ISA/API: Is the burst number in total.
    uint16_t burstNum = intriParams.blockCount;
    // ISA: unit of 64B    API: unit of 32B
    uint16_t burstLen = DivCeil(intriParams.blockLen, 2);
    // ISA/API: unit of 32B
    uint16_t srcGapSize = intriParams.srcStride;
    // ISA: unit of 64B    API: unit of 32B
    uint16_t dstGapSize = DivCeil(intriParams.dstStride, 2);

    copy_cbuf_to_fbuf((__fbuf__ void*)dst, (__cbuf__ void*)src, burstNum, burstLen, srcGapSize, dstGapSize);
}

template <typename T>
__aicore__ inline void DataCopyL12FBImpl(uint64_t dst, __cbuf__ T* src, const DataCopyParams &intriParams, const DataCopyAttrParams& attrParams)
{
    // 该API属于ISASI，并且不支持多次分配内存，只能进行一次AllocTensor操作，然后搬运所有数据到FB。因此直接设置dst = 0确保数据从0开始排布
    dst = 0;
    dst |= static_cast<uint64_t>(attrParams.fixBufPos) << 16;

    // ISA/API: Is the burst number in total.
    uint16_t burstNum = intriParams.blockCount;
    // ISA: unit of 64B    API: unit of 32B
    uint16_t burstLen = DivCeil(intriParams.blockLen, 2);
    // ISA/API: unit of 32B
    uint16_t srcGapSize = intriParams.srcStride;
    // ISA: unit of 64B    API: unit of 32B
    uint16_t dstGapSize = DivCeil(intriParams.dstStride, 2);

    copy_cbuf_to_fbuf((__fbuf__ void*)dst, (__cbuf__ void*)src, burstNum, burstLen, srcGapSize, dstGapSize);
}

template <typename T>
__aicore__ inline void DataCopyL12PTImpl(const uint64_t dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    copy_cbuf_to_pt(dst, (__cbuf__ void *)src, intriParams.blockCount, intriParams.blockLen,
        intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
typename std::enable_if<1 != sizeof(T)>::type __aicore__ inline TransND2NZ(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t high, uint16_t width, T scalar)
{
    struct UnaryRepeatParams intriParams;
    intriParams.srcBlkStride = 1;
    intriParams.dstBlkStride = 1;
    intriParams.srcRepStride = width * sizeof(T) / ONE_BLK_SIZE;
    intriParams.dstRepStride = 1;

    int highBlock = MAX_REPEAT_TIMES;
    int highBlocks = high / highBlock;
    int highTail = high % highBlock;

    uint64_t mask[2];
    mask[0] = (1 << (32 / sizeof(T))) - 1;
    mask[1] = 0;

    int widthFractal = width * sizeof(T) / 32;
    for (int i = 0; i < widthFractal; ++i) {
        for (int j = 0; j < highBlocks; ++j) {
            AddsImpl(dstAddr + i * (32 / sizeof(T)) * high + j * highBlock * (32 / sizeof(T)),
                srcAddr + i * (32 / sizeof(T)) + j * highBlock * width, scalar, mask, highBlock, intriParams);
        }
        if (highTail) {
            AddsImpl(dstAddr + i * (32 / sizeof(T)) * high + highBlocks * highBlock * (32 / sizeof(T)),
                srcAddr + i * (32 / sizeof(T)) + highBlocks * highBlock * width, scalar, mask, highTail, intriParams);
        }
    }
}

template <typename T>
typename std::enable_if<1 == sizeof(T)>::type __aicore__ inline TransND2NZ(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t high, uint16_t width, T scalar)
{
    struct UnaryRepeatParams intriParams;
    intriParams.srcBlkStride = 1;
    intriParams.dstBlkStride = 1;
    intriParams.srcRepStride = width * sizeof(T) / ONE_BLK_SIZE;
    intriParams.dstRepStride = 1;

    int highBlock = MAX_REPEAT_TIMES;
    int highBlocks = high / highBlock;
    int highTail = high % highBlock;

    uint64_t mask = 32;

    int widthFractal = width * sizeof(T) / 32;
    for (int i = 0; i < widthFractal; ++i) {
        for (int j = 0; j < highBlocks; ++j) {
            AddsImpl(dstAddr + i * (32 / sizeof(T)) * high + j * highBlock * (32 / sizeof(T)),
                srcAddr + i * (32 / sizeof(T)) + j * highBlock * width, scalar, mask, highBlock, intriParams);
        }
        if (highTail) {
            AddsImpl(dstAddr + i * (32 / sizeof(T)) * high + highBlocks * highBlock * (32 / sizeof(T)),
                srcAddr + i * (32 / sizeof(T)) + highBlocks * highBlock * width, scalar, mask, highTail, intriParams);
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImplBase(__cbuf__ T* dst, __gm__ T* src, Nd2NzParams& intriParams)
{
    uint16_t ndNum = intriParams.ndNum;
    uint16_t nValue = intriParams.nValue;
    uint16_t dValue = intriParams.dValue;
    uint16_t srcNdMatrixStride = intriParams.srcNdMatrixStride;
    uint16_t srcDValue = intriParams.srcDValue;
    uint16_t dstNzC0Stride = intriParams.dstNzC0Stride;
    uint16_t dstNzNStride = intriParams.dstNzNStride;
    uint16_t dstNzMatrixStride = intriParams.dstNzMatrixStride;

    uint16_t alignedDValueBlockNum = (dValue * sizeof(T) - 1) / 32 + 1;
    uint16_t alignedDValue = alignedDValueBlockNum * 32 / sizeof(T);

    event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    for (int i = 0; i < ndNum; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        __ubuf__ T* nd2nzTempBuf = AscendCUtils::GetTemporaryBufferAddr<T>(TMP_UB_OFFSET, 8 * 1024 / sizeof(T));
        if (((dValue * sizeof(T)) % 32 == 0) && ((srcDValue * sizeof(T)) % 32 == 0)) {
            DataCopyGM2UBImpl(nd2nzTempBuf, src + i * srcNdMatrixStride,
                { nValue, static_cast<uint16_t>(dValue * sizeof(T) / 32),
                static_cast<uint16_t>((srcDValue - dValue) * sizeof(T) / 32), 0 });
            event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
        } else {
            // copy and pad zero
            for (int j = 0; j < nValue; ++j) {
                DataCopyGM2UBImpl(nd2nzTempBuf + j * alignedDValue, src + i * srcNdMatrixStride + j * srcDValue,
                    { 1, static_cast<uint16_t>(alignedDValueBlockNum), 0, 0 });
            }
            if (alignedDValue != dValue) {
                event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

                uint16_t downAlignedDValueBlockNum = dValue * sizeof(T) / 32;
                uint16_t downAlignedDValue = downAlignedDValueBlockNum * 32 / sizeof(T);
                uint64_t mask[2];
                mask[0] = (((uint64_t)1 << (alignedDValue - dValue)) - (uint64_t)1) << (dValue - downAlignedDValue);
                mask[1] = 0;
                DuplicateImpl(nd2nzTempBuf + downAlignedDValue, (T)0, mask, nValue, 1, alignedDValueBlockNum);
            } else {
                event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            }
        }

        __ubuf__ T* nzTempBuf = nd2nzTempBuf + (4 * 1024 / sizeof(T));
        TransND2NZ(nzTempBuf, nd2nzTempBuf, nValue, alignedDValue, (T)0);

        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

        uint16_t widthFractal = alignedDValue * sizeof(T) / 32;
        uint16_t dstStride = dstNzNStride - 1;
        for (int j = 0; j < widthFractal; ++j) {
            DataCopyUB2L1Impl(dst + i * dstNzMatrixStride + j * 32 * dstNzC0Stride / sizeof(T),
                nzTempBuf + j * 32 * nValue / sizeof(T), { nValue, 1, 0, dstStride });
        }
        AscendCUtils::FreeTemporaryBuffer<T>(nd2nzTempBuf);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    uint16_t ndNum = intriParams.ndNum;
    uint16_t nValue = intriParams.nValue;
    uint16_t dValue = intriParams.dValue;
    uint16_t srcNdMatrixStride = intriParams.srcNdMatrixStride;
    uint16_t srcDValue = intriParams.srcDValue;
    uint16_t dstNzC0Stride = intriParams.dstNzC0Stride;
    uint16_t dstNzNStride = intriParams.dstNzNStride;
    uint16_t dstNzMatrixStride = intriParams.dstNzMatrixStride;

    uint16_t countsFor32align = BYTE_32_ALIGN / sizeof(T);
    // 32B 对齐场景，使用 连续读，跳着写 的方式实现
    if ((dValue % countsFor32align == 0) && (srcNdMatrixStride % countsFor32align == 0) &&
        (srcDValue % countsFor32align == 0) && (dstNzMatrixStride % countsFor32align == 0)) {
        return DataCopyGM2L1ND2NZImplAlign32B(dst, src, intriParams);
    }

    // tiling limited 8k, use half, 64*64B
    uint16_t highTiling = 64;  // Byte
    uint16_t witdhTiling = 64; // Byte
    uint16_t highFractal = (nValue * sizeof(T)) / highTiling;
    uint16_t highFractalTail = (nValue * sizeof(T)) % highTiling;
    uint16_t widthFractal = (dValue * sizeof(T)) / witdhTiling;
    uint16_t widthFractalTail = (dValue * sizeof(T)) % witdhTiling;

    Nd2NzParams intriParamsBase { ndNum,
        static_cast<uint16_t>(highTiling / sizeof(T)),
        static_cast<uint16_t>(witdhTiling / sizeof(T)),
        srcNdMatrixStride,
        srcDValue,
        dstNzC0Stride,
        dstNzNStride,
        dstNzMatrixStride };

    for (int i = 0; i < highFractal; ++i) {
        for (int j = 0; j < widthFractal; ++j) {
            DataCopyGM2L1ND2NZImplBase(dst + i * (highTiling / sizeof(T)) * (32 / sizeof(T)) +
                j * witdhTiling * dstNzC0Stride / sizeof(T),
                src + i * highTiling * srcDValue / sizeof(T) + j * witdhTiling / sizeof(T), intriParamsBase);
        }
    }

    // tail
    if (highFractalTail) {
        Nd2NzParams intriParamsBase1 { ndNum,
            static_cast<uint16_t>(highFractalTail / sizeof(T)),
            static_cast<uint16_t>(witdhTiling / sizeof(T)),
            srcNdMatrixStride,
            srcDValue,
            dstNzC0Stride,
            dstNzNStride,
            dstNzMatrixStride };

        for (int j = 0; j < widthFractal; ++j) {
            DataCopyGM2L1ND2NZImplBase(dst + highFractal * (highTiling / sizeof(T)) * (32 / sizeof(T)) +
                j * witdhTiling * dstNzC0Stride / sizeof(T),
                src + highFractal * highTiling * srcDValue / sizeof(T) + j * witdhTiling / sizeof(T), intriParamsBase1);
        }
    }

    if (widthFractalTail) {
        Nd2NzParams intriParamsBase2 { ndNum,
            static_cast<uint16_t>(highTiling / sizeof(T)),
            static_cast<uint16_t>(widthFractalTail / sizeof(T)),
            srcNdMatrixStride,
            srcDValue,
            dstNzC0Stride,
            dstNzNStride,
            dstNzMatrixStride };

        for (int i = 0; i < highFractal; ++i) {
            DataCopyGM2L1ND2NZImplBase(dst + i * (highTiling / sizeof(T)) * (32 / sizeof(T)) +
                widthFractal * witdhTiling * dstNzC0Stride / sizeof(T),
                src + i * highTiling * srcDValue / sizeof(T) + widthFractal * witdhTiling / sizeof(T),
                intriParamsBase2);
        }
    }

    if (highFractalTail && widthFractalTail) {
        Nd2NzParams intriParamsBase2 { ndNum,
            static_cast<uint16_t>(highFractalTail / sizeof(T)),
            static_cast<uint16_t>(widthFractalTail / sizeof(T)),
            srcNdMatrixStride,
            srcDValue,
            dstNzC0Stride,
            dstNzNStride,
            dstNzMatrixStride };

        DataCopyGM2L1ND2NZImplBase(dst + highFractal * (highTiling / sizeof(T)) * (32 / sizeof(T)) +
            widthFractal * witdhTiling * dstNzC0Stride / sizeof(T),
            src + highFractal * highTiling * srcDValue / sizeof(T) + widthFractal * witdhTiling / sizeof(T),
            intriParamsBase2);
    }
}

template <typename T>
__aicore__ inline void DataCopyUB2L1ND2NZImpl(__cbuf__ T* dst, __ubuf__ T* src, const Nd2NzParams& intriParams)
{
    uint16_t ndNum = intriParams.ndNum;                         // nd矩阵数据
    uint16_t nValue = intriParams.nValue;                       // nd矩阵行数
    uint16_t dValue = intriParams.dValue;                       // nd矩阵列数，要求32B对齐
    uint16_t srcNdMatrixStride = intriParams.srcNdMatrixStride; // 源操作数相邻nd矩阵起始地址间的偏移，要求32B对齐
    uint16_t srcDValue = intriParams.srcDValue;                 // 源操作数同一nd矩阵的相邻行起始地址间的偏移 (切分场景，行地址不连续)，要求32B对齐
    uint16_t dstNzC0Stride = intriParams.dstNzC0Stride;         // 目的nz矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移, 单位：block
    uint16_t dstNzNStride = intriParams.dstNzNStride;           // 目的nz矩阵中，Z型矩阵相邻行起始地址之间的偏移，单位：block
    uint16_t dstNzMatrixStride = intriParams.dstNzMatrixStride; // 目的nz矩阵中，相邻nz矩阵起始地址间的偏移，要求32B对齐

    ASCENDC_DEBUG_ASSERT(((dValue * sizeof(T)) % BYTE_32_ALIGN == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(((srcNdMatrixStride * sizeof(T)) % BYTE_32_ALIGN == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "srcNdMatrixStride should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(((srcDValue * sizeof(T)) % BYTE_32_ALIGN == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(((dstNzMatrixStride * sizeof(T)) % BYTE_32_ALIGN == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));

    uint32_t elementsPerBlock = ONE_BLK_SIZE / sizeof(T);
    uint32_t NBlockSize = dValue / elementsPerBlock;
    for (uint32_t ndIdx = 0; ndIdx < ndNum; ++ndIdx) {
        for (int i = 0; i < NBlockSize; ++i) {
            uint32_t offsetDst = ndIdx * dstNzMatrixStride + i * elementsPerBlock * dstNzC0Stride;
            uint32_t offsetSrc = ndIdx * srcNdMatrixStride + i * elementsPerBlock;
            copy_ubuf_to_cbuf(dst + offsetDst, src + offsetSrc, 0, nValue, 1, (srcDValue / elementsPerBlock - 1), (dstNzNStride - 1));
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    CopyCbufToGmAlign(dst, src, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride,
        intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImplBase(__gm__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t high, uint16_t width,
    uint16_t srcNStride, uint16_t dstDStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ub to gm nz2nd"); });
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImpl(__gm__ T* dst, __ubuf__ T* src, const Nz2NdParamsFull& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ub to gm nz2nd"); });
}

template <typename T>
__aicore__ inline void DataCopyPadL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, (uint64_t)true, intriParams);
    }

    copy_cbuf_to_gm_align(dst, src, 0, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyPadL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyExtParams& intriParams)
{
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, (uint64_t)true, intriParams);
    }

    copy_cbuf_to_gm_align(dst, src, 0, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyL12GMNZ2NDImplBase(__gm__ T* dstAddr, __cbuf__ T* srcAddr, uint16_t high, uint16_t width,
    uint16_t srcNStride, uint16_t dstDStride)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::A1>(reinterpret_cast<uint64_t>(srcAddr) % ONE_BLK_SIZE == 0)),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "src address should be 32B aligned \n"));
    const uint16_t highBlock = MAX_REPEAT_TIMES;
    const uint16_t highBlocks = high / highBlock;
    const uint16_t highTail = high % highBlock;
    uint16_t widthElems = BLOCK_CUBE; // b16,b32
    if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        widthElems = ONE_BLK_SIZE / sizeof(T); // b8
    }
    const uint16_t widthFractal = (width + widthElems - 1) / widthElems;

    for (int i = 0; i < widthFractal; ++i) {
        uint16_t computeCount = (i + 1) * widthElems;
        uint16_t leftLen = width >= computeCount ? widthElems : (width - i * widthElems);
        uint16_t srcLeftLen = (sizeof(T) == B32_BYTE_SIZE && leftLen <= DEFAULT_BLK_NUM) ? MIN_BLOCK_LEN : 0;
        for (int j = 0; j < highBlocks; ++j) {
            DataCopyPadL12GMImpl(dstAddr + i * widthElems + j * highBlock * dstDStride,
                srcAddr + i * srcNStride * BLOCK_CUBE + j * highBlock * BLOCK_CUBE,
                { highBlock, static_cast<uint32_t>(leftLen * sizeof(T)), srcLeftLen,
                static_cast<uint32_t>((dstDStride - leftLen) * sizeof(T)), 0 });
        }
        if (highTail) {
            DataCopyPadL12GMImpl(dstAddr + i * widthElems + highBlocks * highBlock * dstDStride,
                srcAddr + i * srcNStride * BLOCK_CUBE + highBlocks * highBlock * BLOCK_CUBE,
                { highTail, static_cast<uint32_t>(leftLen * sizeof(T)), srcLeftLen,
                static_cast<uint32_t>((dstDStride - leftLen) * sizeof(T)), 0 });
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyL12GMNZ2NDImpl(__gm__ T* dst, __cbuf__ T* src, const Nz2NdParamsFull& intriParams)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::A1>(reinterpret_cast<uint64_t>(src)) % ONE_BLK_SIZE == 0),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "src address should be 32B aligned \n"));
    const uint16_t ndNum = intriParams.ndNum;
    const uint16_t nValue = intriParams.nValue;
    const uint16_t dValue = intriParams.dValue;
    const uint16_t srcNdMatrixStride = intriParams.srcNdMatrixStride;
    const uint16_t srcNStride = intriParams.srcNStride;
    const uint16_t dstDStride = intriParams.dstDStride;
    const uint16_t dstNdMatrixStride = intriParams.dstNdMatrixStride;

    for (int i = 0; i < ndNum; ++i) {
        DataCopyL12GMNZ2NDImplBase(dst + i * dstNdMatrixStride, src + i * srcNdMatrixStride * BLOCK_CUBE * BLOCK_CUBE,
            nValue, dValue, srcNStride, dstDStride);
    }
}

/* **************************************************************************************************
 * Copy                                             *
 * ************************************************************************************************* */
// Copy::Level 0 - mask bit mode
template <typename T, bool isSetMask = true>
__aicore__ inline void CopyImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask[2], const uint8_t repeatTime,
    const CopyRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((sizeof(T) == B16_BYTE_SIZE || sizeof(T) == B32_BYTE_SIZE),
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported data type of copy from ubuf to ubuf"); });
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    __VEC_SCOPE__
    {
        RegTensor<T> vreg;
        MaskReg preg = MovePredicate<T>();
        for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(
                vreg, src, repeatParams.srcStride, repeatParams.srcRepeatSize, preg);
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(
                dst, vreg, repeatParams.dstStride, repeatParams.dstRepeatSize, preg);
        }
    }
}

// Copy::Level 0 - mask count mode
template <typename T, bool isSetMask = true>
__aicore__ inline void CopyImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const CopyRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((sizeof(T) == B16_BYTE_SIZE || sizeof(T) == B32_BYTE_SIZE),
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported data type of copy from ubuf to ubuf "); });
    if constexpr (sizeof(T) == B16_BYTE_SIZE || sizeof(T) == B32_BYTE_SIZE) {
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t sreg = static_cast<uint32_t>(mask);
            MaskReg preg = CreatePredicate<T>(sreg);
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(
                    vreg, src, repeatParams.srcStride, repeatParams.srcRepeatSize, preg);
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(
                    dst, vreg, repeatParams.dstStride, repeatParams.dstRepeatSize, preg);
            }
        }
    }
}

/* **************************************************************************************************
 * DataCopy Enhanced                                             *
 * ************************************************************************************************* */

template <typename T, typename U>
__aicore__ inline void DataCopyL12L0CImpl(__cc__ T* dst, __cbuf__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from cbuf to l0c"); });
}

/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */

// ------------  ------------
template <typename T, typename U>
__aicore__ inline void DataCopyL0C2UBImpl(__ubuf__ T* dst, __cc__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from l0c to ubuf"); });
}

template <typename T, typename U>
__aicore__ inline void DataCopyUB2L0CImpl(__cc__ T* dst, __ubuf__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ubuf to l0c"); });
}

template <typename T>
__aicore__ inline void DataCopySliceGm2UBImpl(__ubuf__ T *dst, __gm__ T *src, const DataCopyParams &intriParams)
{
    uint32_t offsetSrc = 0;
    uint32_t offsetDst = 0;
    for (uint32_t i = 0; i < intriParams.blockCount; i++) {
        offsetSrc = offsetSrc + i * (intriParams.blockLen * ONE_BLK_SIZE + intriParams.srcStride);
        offsetDst = offsetDst + i * (intriParams.blockLen * ONE_BLK_SIZE + intriParams.dstStride);
        DataCopyGM2UBImpl(dst + offsetDst / sizeof(T), src + offsetSrc / sizeof(T), {1, intriParams.blockLen, 0, 0});
    }
}

template <typename T>
__aicore__ inline void DataCopySliceUB2GMImpl(__gm__ T *dst, __ubuf__ T *src, const DataCopyParams &intriParams)
{
    uint32_t offsetSrc = 0;
    uint32_t offsetDst = 0;
    for (uint32_t i = 0; i < intriParams.blockCount; i++) {
        offsetSrc = offsetSrc + i * (intriParams.blockLen * ONE_BLK_SIZE + intriParams.srcStride);
        offsetDst = offsetDst + i * (intriParams.blockLen * ONE_BLK_SIZE + intriParams.dstStride);
        DataCopyUB2GMImpl(dst + offsetDst / sizeof(T), src + offsetSrc / sizeof(T), {1, intriParams.blockLen, 0, 0});
    }
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams,
    const DataCopyPadParams& padParams)
{
    if (padParams.isPad) {
        set_pad_val_outtoub(padParams.paddingValue);
    }
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, (uint64_t)true, intriParams);
    }
    copy_gm_to_ubuf_align(dst, src, 0, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
        padParams.rightPadding, (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyExtParams& intriParams,
    const DataCopyPadExtParams<T>& padParams)
{
    if (padParams.isPad) {
        set_pad_val_outtoub(GetScalarBitcodeValue((T)padParams.paddingValue));
    }
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, (uint64_t)true, intriParams);
    }
    copy_gm_to_ubuf_align(dst, src, 0, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
        padParams.rightPadding, (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyPadGm2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams,
    const DataCopyPadParams& padParams)
{
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, (uint64_t)true, intriParams);
    }
    if (padParams.isPad) {
        set_pad_val_outtol1(padParams.paddingValue);
    }
    CopyGmToCbufAlign<T, true>(dst, src, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
        padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyPadGm2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyExtParams& intriParams,
    const DataCopyPadExtParams<T>& padParams)
{
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, (uint64_t)true, intriParams);
    }
    if (padParams.isPad) {
        set_pad_val_outtol1(GetScalarBitcodeValue((T)padParams.paddingValue));
    }
    CopyGmToCbufAlign<T, true>(dst, src, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
#if ASCENDC_CPU_DEBUG
    uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
    ASCENDC_ASSERT((absUbAddr % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "absUbAddr is 0x%lx, which must be 32B aligned", absUbAddr); });
#endif
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, (uint64_t)true, intriParams);
    }
    copy_ubuf_to_gm_align(dst, src, 0, intriParams.blockCount, intriParams.blockLen,
        (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyExtParams& intriParams)
{
#if ASCENDC_CPU_DEBUG
    uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
    ASCENDC_ASSERT((absUbAddr % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "absUbAddr is 0x%lx, which must be 32B aligned", absUbAddr); });
#endif
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, (uint64_t)true, intriParams);
    }
    copy_ubuf_to_gm_align(dst, src, 0, intriParams.blockCount, intriParams.blockLen,
        (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyGM2UBND2NZImpl(__ubuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from gm to ubuf nd2nz"); });
}

template <typename T>
__aicore__ inline void DataCopyPadUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ubuf to cbuf with pad"); });
}

template <typename T>
__aicore__ inline void DataCopyUB2L1Intf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &intriParams)
{
    DataCopyUB2L1Impl((__cbuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), intriParams);
}

template <typename T>
__aicore__ inline void DataCopyUB2L0CIntf(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    DataCopyUB2L0CImpl(
        (__cc__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), intriParams, enhancedParams);
}

#pragma begin_pipe(V)
template <typename T>
__aicore__ inline void DataCopyUB2UBIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &intriParams)
{
    DataCopyUB2UBImpl((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), intriParams);
}
#pragma end_pipe

template <typename T>
__aicore__ inline void DataCopyL12UBIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &intriParams)
{
    DataCopyL12UBImpl((__ubuf__ T *)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), intriParams);
}

template <typename T>
__aicore__ inline void DataCopyL12L0CIntf(const LocalTensor<T> &dst,
    const LocalTensor<T> &src, const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    DataCopyL12L0CImpl(
        (__cc__ T *)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), intriParams, enhancedParams);
}

template <typename T>
__aicore__ inline void DataCopyL0C2UBIntf(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    DataCopyL0C2UBImpl(
        (__ubuf__ T *)dst.GetPhyAddr(), (__cc__ T *)src.GetPhyAddr(), intriParams, enhancedParams);
}

template <typename T>
__aicore__ inline void DataCopyL12BTIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    DataCopyL12BTImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), (uint16_t)0, repeatParams);
}

template <typename T>
__aicore__ inline void DataCopyL12FBIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    DataCopyL12FBImpl((__fbuf__ T *)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), repeatParams);
}

template <typename T>
__aicore__ inline void DataCopyL12PTIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    DataCopyL12PTImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), repeatParams);
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
