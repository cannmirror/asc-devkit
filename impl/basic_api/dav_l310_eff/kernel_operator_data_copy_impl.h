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
 * \brief AscendC l310 eff support data copy api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H

namespace AscendC {
/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */

constexpr uint8_t BYTE_32_ALIGN = 32;          // in unit of 32 bytes

// only support CubeCore   PIPE_MTE2
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

template <typename T>
__aicore__ inline void DataCopyGM2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from GM to UB on this version"); });
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
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to GM on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyUB2UBImpl(__ubuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to UB on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to L1 on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyL12UBImpl(__ubuf__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy L1 to UB on this version"); });
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
__aicore__ inline void DataCopyL12FBImpl(uint64_t dst, __cbuf__ T* src, const DataCopyParams& intriParams)
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
__aicore__ inline void DataCopyGM2L1ND2NZImpl(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported DataCopyGM2L1ND2NZ");}
    );
}

template <typename T>
__aicore__ inline void DataCopyL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    CopyCbufToGmAlign(dst, src, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride,
        intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImpl(__gm__ T* dst, __ubuf__ T* src, const Nz2NdParamsFull& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to GM NZ2ND on this version"); });
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
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to UB on this version"); });
}

// Copy::Level 0 - mask count mode
template <typename T, bool isSetMask = true>
__aicore__ inline void CopyImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const CopyRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to UB on this version"); });
}

/* **************************************************************************************************
 * DataCopy Enhanced                                             *
 * ************************************************************************************************* */

template <typename T, typename U>
__aicore__ inline void DataCopyL12L0CImpl(__cc__ T* dst, __cbuf__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy L1 to L0C on this version"); });
}

/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */

// ------------  ------------
template <typename T, typename U>
__aicore__ inline void DataCopyL0C2UBImpl(__ubuf__ T* dst, __cc__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy L0C to UB on this version"); });
}

template <typename T, typename U>
__aicore__ inline void DataCopyUB2L0CImpl(__cc__ T* dst, __ubuf__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to L0C on this version"); });
}

template <typename T>
__aicore__ inline void DataCopySliceGm2UBImpl(__ubuf__ T *dst, __gm__ T *src, const DataCopyParams &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy GM to UB on this version"); });
}

template <typename T>
__aicore__ inline void DataCopySliceUB2GMImpl(__gm__ T *dst, __ubuf__ T *src, const DataCopyParams &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to GM on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyExtParams& intriParams,
    const DataCopyPadExtParams<T>& padParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy GM to UB on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams,
    const DataCopyPadParams& padParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy GM to UB on this version"); });
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
        set_pad_val_outtol1(GetScalarBitcodeValue(padParams.paddingValue));
    }
    CopyGmToCbufAlign<T, true>(dst, src, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, intriParams.srcStride, intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to GM on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyExtParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to GM on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyGM2UBND2NZImpl(__ubuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy GM to UB ND2NZ on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyPadUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to L1 on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyUB2L1Intf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to L1 on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyUB2L0CIntf(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to L0C on this version"); });
}

#pragma begin_pipe(V)
template <typename T>
__aicore__ inline void DataCopyUB2UBIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy UB to UB on this version"); });
}
#pragma end_pipe

template <typename T>
__aicore__ inline void DataCopyL12UBIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &intriParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy L1 to UB on this version"); });
}

template <typename T>
__aicore__ inline void __in_pipe__(MTE1) __out_pipe__(MTE1) DataCopyL12L0CIntf(const LocalTensor<T> &dst,
    const LocalTensor<T> &src, const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy L1 to L0C on this version"); });
}

template <typename T>
__aicore__ inline void DataCopyL0C2UBIntf(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const DataCopyParams &intriParams, const DataCopyEnhancedParams &enhancedParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy L0C to UB on this version"); });
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
    DataCopyL12FBImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), repeatParams);
}

template <typename T>
__aicore__ inline void DataCopyL12PTIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    DataCopyL12PTImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), repeatParams);
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
