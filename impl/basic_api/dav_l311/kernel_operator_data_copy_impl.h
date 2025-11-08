/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_data_copy_impl.h
 * \brief AscendC l311 support data copy api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
#include "kernel_operator_common_impl.h"
namespace AscendC {
/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */
// all input params: C220 version, not C310 version, thus need conversion by *32
constexpr uint8_t BYTE_32_ALIGN = 32;          // in unit of 32 bytes

// only support VecCore    PIPE_MTE2
// GM -> UB: copy_gm_to_ubuf_align_v2: support changing padding value
template <typename T>
__aicore__ inline void CopyGmToUbufAlignV2(__ubuf__ T* dst, __gm__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint8_t leftPaddingCount, const uint8_t rightPaddingCount, const uint32_t srcStride,
    const uint32_t dstStride, const bool isDataCopyPad, const bool isPad = false, const uint8_t sid = 0)
{
    // DataCopy is in unit of 32 bytes, DataCopyPad is in unit of 1byte / 32bytes
    uint32_t unitOfBytes = (isDataCopyPad) ? 1 : BYTE_32_ALIGN;
    uint32_t burstLength = blockLen * unitOfBytes;
    uint32_t padLength = leftPaddingCount * sizeof(T) + rightPaddingCount * sizeof(T);
    uint64_t srcStride310 = srcStride * unitOfBytes + burstLength;                 // GM  DataCopy:32Bytes, DataCopyPad:1Byte
    uint32_t dstStride310 = dstStride * BYTE_32_ALIGN + burstLength + padLength;   // UB  DataCopy:32Bytes, DataCopyPad:32Bytes
    // gm stride should be 32B aligned
    dstStride310 = DivCeil(dstStride310, BYTE_32_ALIGN) * BYTE_32_ALIGN;

    if constexpr (sizeof(T) == B64_BYTE_SIZE) {
        copy_gm_to_ubuf_align_v2((__ubuf__ uint32_t*)dst, (__gm__ uint32_t*)src, sid, blockCount, burstLength,
            leftPaddingCount * 2, rightPaddingCount * 2, isPad, srcStride310, dstStride310);
    }  else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_gm_to_ubuf_align_v2((__ubuf__ uint32_t*)dst, (__gm__ uint32_t*)src, sid, blockCount, burstLength,
            leftPaddingCount, rightPaddingCount, isPad, srcStride310, dstStride310);
    }  else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_gm_to_ubuf_align_v2((__ubuf__ uint16_t*)dst, (__gm__ uint16_t*)src, sid, blockCount, burstLength,
            leftPaddingCount, rightPaddingCount, isPad, srcStride310, dstStride310);
    }  else if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)dst, (__gm__ uint8_t*)src, sid, blockCount, burstLength,
            leftPaddingCount, rightPaddingCount, isPad, srcStride310, dstStride310);
    } else {
        ASSERT(false && "unsupported data type of copy from gm to ubuf on current device");
    }
}

// only support VecCore   PIPE_MTE3
// UB -> GM: copy_ubuf_to_gm_align_v2: Note: switch the order between srcStride and dstStride!!!!
template <typename T>
__aicore__ inline void CopyUbufToGmAlignV2(__gm__ T* dst, __ubuf__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint32_t srcStride, const uint32_t dstStride, const bool isDataCopyPad)
{
    // DataCopy is in unit of 32 bytes, DataCopyPad is in unit of 1byte / 32bytes
    uint32_t unitOfBytes = (isDataCopyPad) ? 1 : BYTE_32_ALIGN;
    uint32_t burstLength = blockLen * unitOfBytes;
    uint32_t srcStride310 = srcStride * BYTE_32_ALIGN + burstLength;   // UB   DataCopy:32Bytes, DataCopyPad:32Bytes
    uint64_t dstStride310 = dstStride * unitOfBytes + burstLength;     // GM   DataCopy:32Bytes, DataCopyPad:1Byte
    // srcstride should be 32B aligned
    srcStride310 = DivCeil(srcStride310, BYTE_32_ALIGN) * BYTE_32_ALIGN;
    copy_ubuf_to_gm_align_v2(dst, src, 0, blockCount, burstLength, dstStride310,
        srcStride310);
}

// only support CubeCore   PIPE_MTE2
// GM -> L1: copy_gm_to_cbuf_align_v2
template <typename T>
__aicore__ inline void CopyGmToCbufAlignV2(__cbuf__ T* dst, __gm__ T* src, const uint32_t blockCount,
    const uint32_t blockLen, const uint8_t leftPadding, const uint8_t rightPadding, const bool isPad,
    const uint32_t srcStride, const uint32_t dstStride, const bool isDataCopyPad, const uint8_t sid = 0)
{
    uint32_t unitOfBytes = (isDataCopyPad) ? 1 : BYTE_32_ALIGN;
    uint32_t burstLength = blockLen * unitOfBytes;
    uint32_t padLength = leftPadding * sizeof(T) + rightPadding * sizeof(T);
    uint64_t actSrcStride = srcStride * unitOfBytes + burstLength;
    uint32_t actDstStride = AlignUp(dstStride * BYTE_32_ALIGN + burstLength + padLength, BYTE_32_ALIGN);

    if constexpr (sizeof(T) == B64_BYTE_SIZE) {
        copy_gm_to_cbuf_align_v2((__cbuf__ uint32_t*)dst, (__gm__ uint32_t*)src, (uint8_t)sid, blockCount, burstLength,
            leftPadding * 2, rightPadding * 2, isPad, actSrcStride, actDstStride);
    }  else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        copy_gm_to_cbuf_align_v2((__cbuf__ uint32_t*)dst, (__gm__ uint32_t*)src, (uint8_t)sid, blockCount, burstLength,
            leftPadding, rightPadding, isPad, actSrcStride, actDstStride);
    }  else if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        copy_gm_to_cbuf_align_v2((__cbuf__ uint16_t*)dst, (__gm__ uint16_t*)src, (uint8_t)sid, blockCount, burstLength,
            leftPadding, rightPadding, isPad, actSrcStride, actDstStride);
    }  else if constexpr (sizeof(T) == B8_BYTE_SIZE) {
        copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t*)dst, (__gm__ uint8_t*)src, (uint8_t)sid, blockCount, burstLength,
            leftPadding, rightPadding, isPad, actSrcStride, actDstStride);
    } else {
        ASSERT(false && "unsupported data type of copy from gm to cbuf on current device");
    }
}

// only support CubeCore   PIPE_MTE3
// L1 -> GM: copy_cbuf_to_gm_align_v2
template <typename T>
__aicore__ inline void CopyCbufToGmAlignV2(__gm__ T* dst, __cbuf__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint32_t srcStride, const uint32_t dstStride, const uint8_t sid = 0)
{
    uint32_t burstLength = blockLen * BYTE_32_ALIGN;
    uint64_t actSrcStride = srcStride * BYTE_32_ALIGN + burstLength;
    uint32_t actDstStride = dstStride * BYTE_32_ALIGN + burstLength;
    copy_cbuf_to_gm_align_v2((__gm__ T*)dst, (__cbuf__ T*)src, sid, blockCount, burstLength,
        actDstStride, actSrcStride);
}

template <typename T>
__aicore__ inline void DataCopyGM2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    CopyGmToUbufAlignV2(dst, src, intriParams.blockCount, intriParams.blockLen, 0, 0, intriParams.srcStride,
        intriParams.dstStride, false, intriParams.sid);
}

template <typename T>
__aicore__ inline void DataCopyGM2L1Impl(__cbuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams)
{
    CopyGmToCbufAlignV2(dst, src, intriParams.blockCount, intriParams.blockLen, 0, 0, false,
        intriParams.srcStride, intriParams.dstStride, false, intriParams.sid);
}

template <typename T>
__aicore__ inline void DataCopyUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
#ifdef ASCENDC_CPU_DEBUG
    DataCopyWithAtomic(dst, src, intriParams);
#endif // ASCENDC_CPU_DEBUG
    CopyUbufToGmAlignV2(dst, src, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride,
        intriParams.dstStride, false);
}

template <typename T>
__aicore__ inline void DataCopyUB2UBImpl(__ubuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    // 当前#1018实现有问题，需要手动拼1017的config值
    uint64_t config = ((uint64_t)intriParams.dstStride << 48) +
        (((uint64_t)intriParams.srcStride) << 32) +
        (((uint64_t)intriParams.blockLen) << 16) +
        (((uint64_t)intriParams.blockCount) << 4);
    copy_ubuf_to_ubuf((__ubuf__ void*)dst, (__ubuf__ void*)src, config);
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
        ASSERT(false && "unsupported data type of copy from cbuf to bt on this version");
    }
}

template <typename T>
__aicore__ inline void DataCopyL12FBImpl(__fbuf__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
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
__aicore__ inline void DataCopyL12FBImpl(uint64_t dst, __cbuf__ T* src,
    const DataCopyParams& intriParams, const DataCopyAttrParams& attrParams)
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
__aicore__ inline void TransND2NZ(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t high, uint16_t width, T scalar)
{
    ASSERT(false && "unsupported data copy from ub to gm nd2nz on this version");
}

template <typename T>
__aicore__ inline void DataCopyGM2L1ND2NZImplB16(__cbuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    if constexpr (g_gm_overflow_check) {
            __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
            AscendCUtils::CheckGmMemOverflowNd2Nz(src, workSpace, true, intriParams);
    }

    uint16_t loop2DstStride = intriParams.dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
    uint16_t loop3DstStride = intriParams.dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
    uint16_t loop4DstStride = static_cast<uint16_t>(intriParams.dstNzMatrixStride * sizeof(T) / ONE_BLOCK_SIZE);

    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(intriParams.ndNum);            // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);   // CCE: store parameters for ND2NZ DMA instructions

    uint64_t loop1SrcStride = intriParams.srcDValue * sizeof(T);
    uint64_t loop4SrcStride = intriParams.srcNdMatrixStride * sizeof(T);
    copy_gm_to_cbuf_multi_nd2nz((__cbuf__ half *)dst, (__gm__ half *)src, 0, loop1SrcStride,
     intriParams.nValue, intriParams.dValue, loop4SrcStride, false, false);
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
    if constexpr (g_gm_overflow_check) {
            __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
            AscendCUtils::CheckGmMemOverflowNd2Nz(src, workSpace, true, intriParams);
    }

    uint16_t loop2DstStride = intriParams.dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
    uint16_t loop3DstStride = intriParams.dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
    uint16_t loop4DstStride = static_cast<uint16_t>(intriParams.dstNzMatrixStride * sizeof(T) / ONE_BLOCK_SIZE);

    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(intriParams.ndNum);            // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);   // CCE: store parameters for ND2NZ DMA instructions

    uint64_t loop1SrcStride = intriParams.srcDValue * sizeof(T);
    uint64_t loop4SrcStride = intriParams.srcNdMatrixStride * sizeof(T);
    copy_gm_to_cbuf_multi_nd2nz((__cbuf__ float *)dst, (__gm__ float *)src, 0, loop1SrcStride,
     intriParams.nValue, intriParams.dValue, loop4SrcStride, false, false);
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
    if constexpr (g_gm_overflow_check) {
            __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
            AscendCUtils::CheckGmMemOverflowNd2Nz(src, workSpace, true, intriParams);
    }

    uint16_t loop2DstStride = intriParams.dstNzNStride;  // loop2_dst_stride = dst_nz_n_stride
    uint16_t loop3DstStride = intriParams.dstNzC0Stride; // loop3_dst_stride = dst_nz_c0_Stride
    uint16_t loop4DstStride = static_cast<uint16_t>(intriParams.dstNzMatrixStride * sizeof(T) / ONE_BLOCK_SIZE);

    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(intriParams.ndNum);            // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);   // CCE: store parameters for ND2NZ DMA instructions

    uint64_t loop1SrcStride = intriParams.srcDValue * sizeof(T);
    uint64_t loop4SrcStride = intriParams.srcNdMatrixStride * sizeof(T);
    copy_gm_to_cbuf_multi_nd2nz((__cbuf__ int8_t *)dst, (__gm__ int8_t *)src, 0, loop1SrcStride,
     intriParams.nValue, intriParams.dValue, loop4SrcStride, false, false);
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
__aicore__ inline void DataCopyL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    CopyCbufToGmAlignV2(dst, src, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride,
        intriParams.dstStride);
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImplBase(__gm__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t high, uint16_t width,
    uint16_t srcNStride, uint16_t dstDStride)
{
    ASSERT(false && "unsupported data copy from ub to gm nz2nd on this version");
}

template <typename T>
__aicore__ inline void DataCopyUB2GMNZ2NDImpl(__gm__ T* dst, __ubuf__ T* src, const Nz2NdParamsFull& intriParams)
{
    ASSERT(false && "unsupported data copy from ub to gm nz2nd on this version");
}

template <typename T>
__aicore__ inline void DataCopyL12GMNZ2NDImpl(__gm__ T* dst, __cbuf__ T* src, const Nz2NdParamsFull& intriParams)
{
    ASSERT(false && "unsupported data copy from cbuf to gm with nz2nd");
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
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported data type of copy from ubuf to ubuf on this version"); });
    __ubuf__ uint8_t *maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)maskBuf) = mask[0];
    *((__ubuf__ uint64_t*)maskBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            MaskReg preg;
            plds(preg, ((__ubuf__ uint32_t *)maskBuf), 0, US);
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(
                    vreg, src, repeatParams.srcStride, repeatParams.srcRepeatSize, preg);
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(
                    dst, vreg, repeatParams.dstStride, repeatParams.dstRepeatSize, preg);
            }
        }
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            MaskReg preg;
            plds(preg, ((__ubuf__ uint32_t *)maskBuf), 0, US);
            for (uint16_t i = 0; i < (uint16_t)repeatTime; ++i) {
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(
                    vreg, src, repeatParams.srcStride, repeatParams.srcRepeatSize, preg);
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(
                    dst, vreg, repeatParams.dstStride, repeatParams.dstRepeatSize, preg);
            }
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(maskBuf);
}

// Copy::Level 0 - mask count mode
template <typename T, bool isSetMask = true>
__aicore__ inline void CopyImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint64_t mask, const uint8_t repeatTime,
    const CopyRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((sizeof(T) == B16_BYTE_SIZE || sizeof(T) == B32_BYTE_SIZE),
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported data type of copy from ubuf to ubuf on this version"); });
    if constexpr (sizeof(T) == B16_BYTE_SIZE || sizeof(T) == B32_BYTE_SIZE) {
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            uint32_t sreg = (uint32_t)mask;
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
    ASSERT(false && "unsupported data copy from cbuf to l0c on this version");
}

/* **************************************************************************************************
 * DataCopy                                             *
 * ************************************************************************************************* */

// ------------  ------------
template <typename T, typename U>
__aicore__ inline void DataCopyL0C2UBImpl(__ubuf__ T* dst, __cc__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from l0c to ubuf on this version"); });
}

template <typename T, typename U>
__aicore__ inline void DataCopyUB2L0CImpl(__cc__ T* dst, __ubuf__ U* src, const DataCopyParams& intriParams,
    const DataCopyEnhancedParams& enhancedParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported data copy from ubuf to l0c on this version"); });
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
__aicore__ inline void DataCopyPadGm2UBInternal(__ubuf__ T* dst, __gm__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint32_t srcStride, const uint32_t dstStride,
    const uint8_t leftPadding, const uint8_t rightPadding, const bool isPad)
{
    CopyGmToUbufAlignV2(dst, src, blockCount, blockLen, leftPadding,
        rightPadding, srcStride, dstStride, true, isPad);
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyExtParams& intriParams,
    const DataCopyPadExtParams<T>& padParams)
{
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, (uint64_t)true, intriParams);
    }
    if (padParams.isPad) {
        set_pad_val_outtoub(GetScalarBitcodeValue(padParams.paddingValue));
    }
    DataCopyPadGm2UBInternal(dst, src, intriParams.blockCount, intriParams.blockLen, intriParams.srcStride,
        intriParams.dstStride, padParams.leftPadding, padParams.rightPadding, padParams.isPad);
}

template <typename T>
__aicore__ inline void DataCopyPadGm2UBImpl(__ubuf__ T* dst, __gm__ T* src, const DataCopyParams& intriParams,
    const DataCopyPadParams& padParams)
{
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(src, workSpace, true, (uint64_t)true, intriParams);
    }
    if (padParams.isPad) {
        set_pad_val_outtoub(padParams.paddingValue);
    }
    DataCopyPadGm2UBInternal(dst, src, intriParams.blockCount, (uint32_t)intriParams.blockLen,
        (uint32_t)intriParams.srcStride, (uint32_t)intriParams.dstStride, padParams.leftPadding,
        padParams.rightPadding, padParams.isPad);
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
    CopyGmToCbufAlignV2(dst, src, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
        padParams.rightPadding, padParams.isPad, intriParams.srcStride, intriParams.dstStride, true, intriParams.sid);
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
    CopyGmToCbufAlignV2(dst, src, intriParams.blockCount, intriParams.blockLen, padParams.leftPadding,
            padParams.rightPadding, padParams.isPad, intriParams.srcStride, intriParams.dstStride, true);
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMInternal(__gm__ T* dst, __ubuf__ T* src, const uint16_t blockCount,
    const uint32_t blockLen, const uint32_t dstStride, const uint32_t srcStride)
{
    CopyUbufToGmAlignV2(dst, src, blockCount, blockLen, srcStride, dstStride, true);
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
#if __CCE_KT_TEST__
    uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
    ASCENDC_ASSERT((absUbAddr % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "absUbAddr is 0x%lx, which must be 32B aligned", absUbAddr); });
#endif
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, (uint64_t)true, intriParams);
    }
    DataCopyPadUB2GMInternal(dst, src, intriParams.blockCount, (uint32_t)intriParams.blockLen,
        (uint32_t)intriParams.dstStride, (uint32_t)intriParams.srcStride);
}

template <typename T>
__aicore__ inline void DataCopyPadUB2GMImpl(__gm__ T* dst, __ubuf__ T* src, const DataCopyExtParams& intriParams)
{
#if __CCE_KT_TEST__
    uint64_t absUbAddr = (uint8_t*)src - (uint8_t*)(GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN));
    ASCENDC_ASSERT((absUbAddr % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "absUbAddr is 0x%lx, which must be 32B aligned", absUbAddr); });
#endif
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, (uint64_t)true, intriParams);
    }
    DataCopyPadUB2GMInternal(dst, src, intriParams.blockCount, intriParams.blockLen,
        intriParams.dstStride, intriParams.srcStride);
}

template <typename T>
__aicore__ inline void DataCopyPadL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyParams& intriParams)
{
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, (uint64_t)true, intriParams);
    }

    uint64_t srcStride = AlignUp(intriParams.srcStride * BYTE_32_ALIGN + intriParams.blockLen, BYTE_32_ALIGN);
    uint32_t dstStride = intriParams.blockLen + intriParams.dstStride;

    copy_cbuf_to_gm_align_v2(dst, src, intriParams.sid, intriParams.blockCount, intriParams.blockLen, dstStride,
        srcStride);
}

template <typename T>
__aicore__ inline void DataCopyPadL12GMImpl(__gm__ T* dst, __cbuf__ T* src, const DataCopyExtParams& intriParams)
{
    if constexpr (g_gm_overflow_check) {
        __gm__ uint8_t* workSpace = GetSysWorkSpacePtr();
        AscendCUtils::CheckGmMemOverflowNormal(dst, workSpace, false, (uint64_t)true, intriParams);
    }

    uint64_t srcStride = AlignUp(intriParams.srcStride * BYTE_32_ALIGN + intriParams.blockLen, BYTE_32_ALIGN);
    uint32_t dstStride = intriParams.blockLen + intriParams.dstStride;

    copy_cbuf_to_gm_align_v2(dst, src, 0, intriParams.blockCount, intriParams.blockLen, dstStride, srcStride);
}

template <typename T>
__aicore__ inline void DataCopyGM2UBND2NZImpl(__ubuf__ T* dst, __gm__ T* src, const Nd2NzParams& intriParams)
{
    ASSERT(false && "unsupported data copy from gm to ubuf nd2nz on this version");
}

template <typename T>
__aicore__ inline void DataCopyPadUB2L1Impl(__cbuf__ T* dst, __ubuf__ T* src, const DataCopyParams& intriParams)
{
    ASSERT(false && "unsupported data copy from ubuf to cbuf with pad on this version");
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
__aicore__ inline __in_pipe__(FIX) __out_pipe__(FIX) void DataCopyL12FBIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    DataCopyL12FBImpl((__fbuf__ T *)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), repeatParams);
}

template <typename T>
__aicore__ inline __in_pipe__(FIX) __out_pipe__(FIX) void DataCopyL12PTIntf(
    const LocalTensor<T> &dst, const LocalTensor<T> &src, const DataCopyParams &repeatParams)
{
    DataCopyL12PTImpl((uint64_t)dst.GetPhyAddr(), (__cbuf__ T *)src.GetPhyAddr(), repeatParams);
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DATA_COPY_IMPL_H
