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
 * \brief AscendC l210 support data copy api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H

#include "kernel_operator_vec_duplicate_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void DataCopyGM2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    copy_gm_to_ubuf((__ubuf__ void*)dst, (__gm__ void*)src, 0, intriParams.blockCount, intriParams.blockLen,
        intriParams.srcStride, intriParams.dstStride);
}
template <typename T>

__aicore__ inline void DataCopyGM2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    copy_gm_to_cbuf((__cbuf__ void*)dst, (__gm__ void*)src,
        static_cast<uint8_t>(0),
        static_cast<uint16_t>(intriParams.blockCount),
        static_cast<uint16_t>(intriParams.blockLen),
        static_cast<uint16_t>(intriParams.srcStride),
        static_cast<uint16_t>(intriParams.dstStride),
        static_cast<pad_t>(0));
}

template <typename T>
__aicore__ inline void DataCopyUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
#ifdef ASCENDC_CPU_DEBUG
    DataCopyWithAtomic(dst, src, intriParams);
#endif // ASCENDC_CPU_DEBUG
    copy_ubuf_to_gm((__gm__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount, intriParams.blockLen,
        intriParams.srcStride, intriParams.dstStride);
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
    copy_ubuf_to_cbuf((__cbuf__ void *)dst, (__ubuf__ void *)src,
    static_cast<uint8_t>(0),
    static_cast<uint16_t>(intriParams.blockCount),
    static_cast<uint16_t>(intriParams.blockLen),
    static_cast<uint16_t>(intriParams.srcStride),
    static_cast<uint16_t>(intriParams.dstStride));
}

template <typename T>
__aicore__ inline void DataCopyL12UBImpl(__ubuf__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    copy_cbuf_to_ubuf((__ubuf__ void *)dst, (__cbuf__ void *)src,
    static_cast<uint8_t>(0),
    static_cast<uint16_t>(intriParams.blockCount),
    static_cast<uint16_t>(intriParams.blockLen),
    static_cast<uint16_t>(intriParams.srcStride),
    static_cast<uint16_t>(intriParams.dstStride));
}

template <typename T>
__aicore__ inline void DataCopyL12BTImpl(const uint64_t dst, __cbuf__ T *src, const uint16_t isenableConv,
    const DataCopyParams &intriParams)
{
    ASCENDC_ASSERT(false,
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from C1 to C2 on current device"); });
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
__aicore__ inline void __in_pipe__(MTE1) __out_pipe__(MTE1) DataCopyL12L0CIntf(const LocalTensor<T> &dst,
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
__aicore__ inline __in_pipe__(MTE1) __out_pipe__(MTE1) void DataCopyL12BTIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    DataCopyL12BTImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), (uint16_t)0, repeatParams);
}

template <typename T>
__aicore__ inline void DataCopyL12FBImpl(uint64_t dst, __cbuf__ T* src, const DataCopyParams &intriParams)
{
    // 该API属于ISASI，并且不支持多次分配内存，只能进行一次AllocTensor操作，然后搬运所有数据到FB。因此直接设置dst =
    // 0确保数据从0开始排布
    dst = 0;

    // ISA/API: Is the burst number in total.
    uint16_t burstNum = intriParams.blockCount;
    // ISA: unit of 32B    API: unit of 32B
    uint16_t burstLen = intriParams.blockLen;
    // ISA/API: unit of 32B
    uint16_t srcGapSize = intriParams.srcStride;
    // ISA: unit of 128B    API: unit of 32B
    uint16_t dstGapSize = DivCeil(intriParams.dstStride, 4);

    copy_cbuf_to_fbuf((__fbuf__ void *)dst,
        (__cbuf__ void *)src,
        (uint16_t)burstNum,
        (uint16_t)burstLen,
        (uint16_t)srcGapSize,
        (uint16_t)dstGapSize);
}

template <typename T>
__aicore__ inline void DataCopyL12FBImpl(uint64_t dst, __cbuf__ T* src, const DataCopyParams &intriParams, const DataCopyAttrParams& attrParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from cbuf to fb with attr"); });
}

template <typename T>
__aicore__ inline __in_pipe__(FIX) __out_pipe__(FIX) void DataCopyL12FBIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    DataCopyL12FBImpl((__fbuf__ T *)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), repeatParams);
}

template <typename T>
__aicore__ inline void TransND2NZ(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t high, uint16_t width, T scalar)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ub to gm nd2nz on current device"); });
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImplB16(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ub to gm nd2nz on current device"); });
}

__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ half* dst, __gm__ half* src, const Nd2NzParams& intriParams)
{
    DataCopyGM2L1ND2NZImplB16(dst, src, intriParams);
}

__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ int16_t* dst, __gm__ int16_t* src,
    const Nd2NzParams& intriParams)
{
    DataCopyGM2L1ND2NZImplB16(dst, src, intriParams);
}

__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ uint16_t* dst, __gm__ uint16_t* src,
    const Nd2NzParams& intriParams)
{
    DataCopyGM2L1ND2NZImplB16(dst, src, intriParams);
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImplB32(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ub to gm nd2nz on current device"); });
}

__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ float* dst, __gm__ float* src, const Nd2NzParams& intriParams)
{
    DataCopyGM2L1ND2NZImplB32(dst, src, intriParams);
}

__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ int32_t* dst, __gm__ int32_t* src,
    const Nd2NzParams& intriParams)
{
    DataCopyGM2L1ND2NZImplB32(dst, src, intriParams);
}

__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ uint32_t* dst, __gm__ uint32_t* src,
    const Nd2NzParams& intriParams)
{
    DataCopyGM2L1ND2NZImplB32(dst, src, intriParams);
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImplB8(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from gm to l1 nd2nz on current device"); });
}

__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ int8_t* dst, __gm__ int8_t* src, const Nd2NzParams& intriParams)
{
    DataCopyGM2L1ND2NZImplB8(dst, src, intriParams);
}

__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ uint8_t* dst, __gm__ uint8_t* src,
    const Nd2NzParams& intriParams)
{
    DataCopyGM2L1ND2NZImplB8(dst, src, intriParams);
}

template <typename T>
__aicore__ inline void DataCopyUB2L1ND2NZImpl(__cbuf__ T *dst, __ubuf__ T *src, const Nd2NzParams &intriParams)
{
    uint16_t ndNum = intriParams.ndNum;                          // nd矩阵数据
    uint16_t nValue = intriParams.nValue;                        // nd矩阵行数
    uint16_t dValue = intriParams.dValue;                        // nd矩阵列数，要求32B对齐
    uint16_t srcNdMatrixStride = intriParams.srcNdMatrixStride;  // 源操作数相邻nd矩阵起始地址间的偏移，要求32B对齐
    uint16_t srcDValue =
        intriParams.srcDValue;  // 源操作数同一nd矩阵的相邻行起始地址间的偏移 (切分场景，行地址不连续)，要求32B对齐
    uint16_t dstNzC0Stride =
        intriParams.dstNzC0Stride;  // 目的nz矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移, 单位：block
    uint16_t dstNzNStride = intriParams.dstNzNStride;  // 目的nz矩阵中，Z型矩阵相邻行起始地址之间的偏移，单位：block
    uint16_t dstNzMatrixStride =
        intriParams.dstNzMatrixStride;  // 目的nz矩阵中，相邻nz矩阵起始地址间的偏移，要求32B对齐

    ASCENDC_DEBUG_ASSERT(((dValue * sizeof(T)) % ONE_BLK_SIZE == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(
        ((srcNdMatrixStride * sizeof(T)) % ONE_BLK_SIZE == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "srcNdMatrixStride should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(((srcDValue * sizeof(T)) % ONE_BLK_SIZE == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));
    ASCENDC_DEBUG_ASSERT(((dstNzMatrixStride * sizeof(T)) % ONE_BLK_SIZE == 0), KERNEL_LOG_INTERNAL(KERNEL_ERROR, "dValue should be 32B aligned \n"));

    uint32_t elementsPerBlock = ONE_BLK_SIZE / sizeof(T);
    uint32_t NBlockSize = DivCeil(dValue, elementsPerBlock);
    for (uint32_t ndIdx = 0; ndIdx < ndNum; ++ndIdx) {
        for (uint32_t i = 0; i < NBlockSize; ++i) {
            uint32_t offsetDst = ndIdx * dstNzMatrixStride + i * elementsPerBlock * dstNzC0Stride;
            uint32_t offsetSrc = ndIdx * srcNdMatrixStride + i * elementsPerBlock;
            copy_ubuf_to_cbuf(
                dst + offsetDst, src + offsetSrc, 0, nValue, 1, (srcDValue / elementsPerBlock - 1), (dstNzNStride - 1));
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    copy_cbuf_to_gm((__gm__ void*)dst, (__cbuf__ void*)src, (uint8_t)0, (uint16_t)intriParams.blockCount,
        (uint16_t)intriParams.blockLen, (uint16_t)intriParams.srcStride, (uint16_t)intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImplBase(__gm__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t high, uint16_t width,
    uint16_t srcNStride, uint16_t dstDStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ub to gm nz2nd on current device"); });
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImpl(__gm__ T* dst, __ubuf__ T* src, const Nz2NdParamsFull& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ub to gm nz2nd on current device"); });
}

template <typename T>
__aicore__ inline void DataCopyL12GMNZ2NDImplBase(
    __gm__ T *dstAddr, __cbuf__ T *srcAddr, uint16_t high, uint16_t width, uint16_t srcNStride, uint16_t dstDStride)
{
    ASCENDC_DEBUG_ASSERT((TransUBAddr<TPosition::A1>(reinterpret_cast<uint64_t>(srcAddr) % ONE_BLK_SIZE == 0)),
        KERNEL_LOG_INTERNAL(KERNEL_ERROR, "src address should be 32B aligned \n"));
    const uint16_t highBlock = MAX_REPEAT_TIMES;
    const uint16_t highBlocks = high / highBlock;
    const uint16_t highTail = high % highBlock;
    uint16_t widthElems = BLOCK_CUBE;  // b16,b32
    if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        widthElems = ONE_BLK_SIZE / sizeof(T);  // b8
    }
    const uint16_t widthFractal = (width + widthElems - 1) / widthElems;

    for (int i = 0; i < widthFractal; ++i) {
        uint16_t computeCount = (i + 1) * widthElems;
        uint16_t leftLen = width >= computeCount ? widthElems : (width - i * widthElems);
        uint16_t srcLeftLen = (sizeof(T) == B32_BYTE_SIZE && leftLen <= DEFAULT_BLK_NUM) ? MIN_BLOCK_LEN : 0;
        for (int j = 0; j < highBlocks; ++j) {
            copy_cbuf_to_gm(dstAddr + i * widthElems + j * highBlock * dstDStride,
                srcAddr + i * srcNStride * BLOCK_CUBE + j * highBlock * BLOCK_CUBE,
                0,
                highBlock,
                static_cast<int32_t>(leftLen * sizeof(T)) / ONE_BLK_SIZE,
                srcLeftLen,
                static_cast<int32_t>((dstDStride - leftLen) * sizeof(T)) / ONE_BLK_SIZE);
        }
        if (highTail) {
            copy_cbuf_to_gm(dstAddr + i * widthElems + highBlocks * highBlock * dstDStride,
                srcAddr + i * srcNStride * BLOCK_CUBE + highBlocks * highBlock * BLOCK_CUBE,
                0,
                highTail,
                static_cast<int32_t>(leftLen * sizeof(T)) / ONE_BLK_SIZE,
                srcLeftLen,
                static_cast<int32_t>((dstDStride - leftLen) * sizeof(T)) / ONE_BLK_SIZE);
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
__aicore__ inline void CopyImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask[2], uint8_t repeatTime,
    const CopyRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ubuf to ubuf on current device"); });
}

// Copy::Level 0 - mask count mode
template <typename T, bool isSetMask = true>
__aicore__ inline void CopyImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTime,
    const CopyRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ubuf to ubuf on current device"); });
}

/* **************************************************************************************************
 * DataCopy Enhanced                                             *
 * ************************************************************************************************* */

template <typename T, typename U>
__aicore__ inline void DataCopyL12L0CImpl(__cc__ T* dst, __cbuf__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from cbuf to l0c on current device"); });
}

/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */

// ------------  ------------
template <typename T, typename U>
__aicore__ inline void DataCopyL0C2UBImpl(__ubuf__ T* dst, __cc__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from l0c to ubuf on current device"); });
}

template <typename T, typename U>
__aicore__ inline void DataCopyUB2L0CImpl(__cc__ T* dst, __ubuf__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ubuf to l0c on current device"); });
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
    const uint16_t rightPadding, const uint16_t leftPadding,
    const T paddingValue = 0)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from gm to ubuf with pad on current device"); });
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams,
    const DataCopyPadParams& padParams)
{
    if (padParams.isPad) {
        set_mov_pad_val(GetScalarBitcodeValue(padParams.paddingValue));
    }
    if constexpr (sizeof(T) > 4) {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
            "unsupported dtype for data copy from global to local in on current device"); });
    }
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, true, intriParams);
    }

    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        copy_gm_to_ubuf_pad_b8((__ubuf__ void*)dst, (__gm__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, padParams.leftPadding, padParams.rightPadding);
    } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_gm_to_ubuf_pad_b16((__ubuf__ void*)dst, (__gm__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, padParams.leftPadding, padParams.rightPadding);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_gm_to_ubuf_pad_b32((__ubuf__ void*)dst, (__gm__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, padParams.leftPadding, padParams.rightPadding);
    }
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyExtParams& intriParams,
    const DataCopyPadExtParams<T>& padParams)
{
    if (padParams.isPad) {
        set_mov_pad_val(GetScalarBitcodeValue(padParams.paddingValue));
    }
    if constexpr (sizeof(T) > 4) {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
            "unsupported dtype for data copy from global to local on current device"); });
    }
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, true, intriParams);
    }

    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        copy_gm_to_ubuf_pad_b8((__ubuf__ void*)dst, (__gm__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, padParams.leftPadding, padParams.rightPadding);
    } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_gm_to_ubuf_pad_b16((__ubuf__ void*)dst, (__gm__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, padParams.leftPadding, padParams.rightPadding);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_gm_to_ubuf_pad_b32((__ubuf__ void*)dst, (__gm__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, padParams.leftPadding, padParams.rightPadding);
    }
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
#if ASCENDC_CPU_DEBUG
    uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
    ASCENDC_ASSERT((absUbAddr % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "absUbAddr is 0x%lx, which must be 32B aligned", absUbAddr); });
#endif
    if constexpr (sizeof(T) > B32_BYTE_SIZE) {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
            "unsupported dtype for data copy from global to local on current device"); });
    }
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, true, intriParams);
    }

    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        copy_ubuf_to_gm_pad_b8((__gm__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, 0, 0);
    } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_ubuf_to_gm_pad_b16((__gm__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, 0, 0);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_ubuf_to_gm_pad_b32((__gm__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, 0, 0);
    }
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyExtParams& intriParams)
{
#if ASCENDC_CPU_DEBUG
    uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
    ASCENDC_ASSERT((absUbAddr % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "absUbAddr is 0x%lx, which must be 32B aligned", absUbAddr); });
#endif
    if constexpr (sizeof(T) > B32_BYTE_SIZE) {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR,
            "unsupported dtype for data copy from global to local on current device"); });
    }
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, true, intriParams);
    }

    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        copy_ubuf_to_gm_pad_b8((__gm__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, 0, 0);
    } else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_ubuf_to_gm_pad_b16((__gm__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, 0, 0);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_ubuf_to_gm_pad_b32((__gm__ void*)dst, (__ubuf__ void*)src, 0, intriParams.blockCount,
            intriParams.blockLen, intriParams.srcStride, intriParams.dstStride, 0, 0);
    }
}

template <typename T>
__aicore__ inline void DataCopyGM2UBND2NZImpl(__ubuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from gm to ubuf nd2nz on current device"); });
}

template <typename T>
__aicore__ inline void DataCopyPadUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ubuf to cbuf with pad on current device"); });
}

template <typename T>
__aicore__ inline __in_pipe__(FIX) __out_pipe__(FIX) void DataCopyL12PTIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from cbuf to pt"); });
}

template <typename T>
__aicore__ inline void DataCopyPadGm2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams,
    const DataCopyPadParams& padParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported DataCopyPadGm2L1"); });
}

template <typename T>
__aicore__ inline void DataCopyPadGm2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyExtParams& intriParams,
    const DataCopyPadExtParams<T>& padParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported DataCopyPadGm2L1"); });
}

template <typename T>
__aicore__ inline void DataCopyPadL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported DataCopyPadL12GM"); });
}

template <typename T>
__aicore__ inline void DataCopyPadL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyExtParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported DataCopyPadL12GM"); });
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
