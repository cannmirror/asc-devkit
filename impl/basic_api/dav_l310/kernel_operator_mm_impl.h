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
 * \file kernel_operator_mm_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_MM_IMPL_H
#define ASCENDC_MODULE_OPERATOR_MM_IMPL_H
#include "kernel_struct_mm.h"
namespace AscendC {
/* **************************************************************************************************
 * LoadData 2dv2                                             *
 * ************************************************************************************************* */

/****************************L1 To L0A******************************************/
template <typename T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    if constexpr (B8_BYTE_SIZE == sizeof(T)) {
        load_cbuf_to_ca((__ca__ int8_t*)dst, (__cbuf__ int8_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.srcStride,
            loadDataParam.dstStride, false);
    } else if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        load_cbuf_to_ca((__ca__ int16_t*)dst, (__cbuf__ int16_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.srcStride,
            loadDataParam.dstStride, false);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/****************************L1 To L0B******************************************/
template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T* dst, __cbuf__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    if constexpr (B8_BYTE_SIZE == sizeof(T)) {
        load_cbuf_to_cb((__cb__ int8_t *)dst, (__cbuf__ int8_t *)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.srcStride,
            loadDataParam.dstStride, false);
    } else if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        load_cbuf_to_cb((__cb__ int16_t *)dst, (__cbuf__ int16_t *)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.srcStride,
            loadDataParam.dstStride, false);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/****************************GM To L0A******************************************/

template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T* dst, __gm__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    if constexpr (B8_BYTE_SIZE == sizeof(T)) {
        load_gm_to_ca_2dv2((__ca__ int8_t*)dst, (__gm__ int8_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        load_gm_to_ca_2dv2((__ca__ int16_t*)dst, (__gm__ int16_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else if constexpr (B32_BYTE_SIZE == sizeof(T)) {
        load_gm_to_ca_2dv2((__ca__ int32_t*)dst, (__gm__ int32_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/****************************GM To L0B******************************************/
template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T* dst, __gm__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    if constexpr (B8_BYTE_SIZE == sizeof(T)) {
        load_gm_to_cb_2dv2((__cb__ int8_t*)dst, (__gm__ int8_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        load_gm_to_cb_2dv2((__cb__ int16_t*)dst, (__gm__ int16_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else if constexpr (B32_BYTE_SIZE == sizeof(T)) {
        load_gm_to_cb_2dv2((__cb__ int32_t*)dst, (__gm__ int32_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/****************************GM To L1******************************************/
template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T* dst, __gm__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    if constexpr (B8_BYTE_SIZE == sizeof(T)) {
        load_gm_to_cbuf_2dv2((__cbuf__ int8_t*)dst, (__gm__ int8_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        load_gm_to_cbuf_2dv2((__cbuf__ int16_t*)dst, (__gm__ int16_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else if constexpr (B32_BYTE_SIZE == sizeof(T)) {
        load_gm_to_cbuf_2dv2((__cbuf__ int32_t*)dst, (__gm__ int32_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.srcStride, loadDataParam.dstStride, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.sid);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/* **************************************************************************************************
 * LoadData 2dv1                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LoadData 2dv1 is not supported!"); });
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T* dst, __cbuf__ T* src, const LoadData2DParams& loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LoadData 2dv1 is not supported!"); });
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LoadData 2dv1 is not supported!"); });
}

template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LoadData 2dv1 is not supported!"); });
}

template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T* dst, __gm__ T* src, const LoadData2DParams& loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LoadData 2dv1 is not supported!"); });
}

/* **************************************************************************************************
 * LoadData 2d with transpose                                             *
 * ************************************************************************************************* */

template <typename T>
__aicore__ inline void LoadData2DL12L0ATransposeCal(__ca__ T * dst, __cbuf__ T * src,
    const LoadData2dTransposeParams& loadDataParam)
{
    if constexpr (B8_BYTE_SIZE == sizeof(T)) {
        load_cbuf_to_ca((__ca__ int8_t*)dst, (__cbuf__ int8_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.srcStride,
            loadDataParam.dstStride, true);
    } else if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        load_cbuf_to_ca((__ca__ int16_t*)dst, (__cbuf__ int16_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.srcStride,
            loadDataParam.dstStride, true);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BTransposeCal(__cb__ T * dst, __cbuf__ T * src,
    const LoadData2dTransposeParams& loadDataParam)
{
    if constexpr (B8_BYTE_SIZE == sizeof(T)) {
        load_cbuf_to_cb((__cb__ int8_t*)dst, (__cbuf__ int8_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.srcStride,
            loadDataParam.dstStride, true);
    } else if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        load_cbuf_to_cb((__cb__ int16_t*)dst, (__cbuf__ int16_t*)src, loadDataParam.mStartPosition,
            loadDataParam.kStartPosition, loadDataParam.mStep, loadDataParam.kStep, loadDataParam.srcStride,
            loadDataParam.dstStride, true);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
    img2colv2_cbuf_to_ca(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
        loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
        loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
        loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
        loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
    img2colv2_cbuf_to_cb(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
        loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
        loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
        loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
        loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
}

__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ half* dst, __cbuf__ half* src,
    const LoadData3DParamsV2<bfloat16_t>& loadDataParams)
{
    img2colv2_cbuf_to_ca(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
        loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
        loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
        loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
        loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
}

__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ half* dst, __cbuf__ half* src,
    const LoadData3DParamsV2<bfloat16_t>& loadDataParams)
{
    img2colv2_cbuf_to_cb(dst, src, loadDataParams.kExtension, loadDataParams.mExtension, loadDataParams.kStartPt,
        loadDataParams.mStartPt, loadDataParams.strideW, loadDataParams.strideH, loadDataParams.filterW,
        loadDataParams.filterH, loadDataParams.dilationFilterW, loadDataParams.dilationFilterH,
        loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose,
        loadDataParams.fMatrixCtrl, loadDataParams.channelSize);
}

/* **************************************************************************************************
 * LoadData 3dv2Pro                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12L0ACal(__ca__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    img2colv2_cbuf_to_ca(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
        loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
        loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
        loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
        loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W, loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H,
        loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
        loadDataParams.channelSize);
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    img2colv2_cbuf_to_cb(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
        loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
        loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
        loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
        loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W, loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H,
        loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
        loadDataParams.channelSize);
}

template <typename T>
__aicore__ inline void LoadData3DV2L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_ASSERT((false),
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 from A1/B1 to UB"); });
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ uint8_t* dst, __cbuf__ uint8_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 load uint8_t dtype from A1/B1 to B2");
    });
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ int8_t* dst, __cbuf__ int8_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 load int8_t dtype from A1/B1 to B2");
    });
}

/* **************************************************************************************************
 * Mmad                                             *
 * ************************************************************************************************* */
template <typename T, typename U, typename S>
__aicore__ inline void MmadCal(__cc__ T* c, __ca__ U* a, __cb__ S* b, const MmadParams& mmadParams)
{
    uint64_t config = 0;
    config |= (((uint64_t)mmadParams.m & 0xfff) << 0);
    config |= (((uint64_t)mmadParams.k & 0xfff) << 12);
    config |= (((uint64_t)mmadParams.n & 0xfff) << 24);
    config |= (((uint64_t)mmadParams.fmOffset & 0xff) << 36);
    config |= (((uint64_t)mmadParams.smaskBufferAddr & 0x7f) << 44);
    config |= (((uint64_t)mmadParams.unitFlag & 0x3) << 55);
    config |= (((uint64_t)mmadParams.s16s8rightShift & 0x1) << 57);
    config |= (((uint64_t)mmadParams.s16s8subDtype & 0x1) << 59);
    config |= (((uint64_t)mmadParams.isWeightOffset & 0x1) << 60);
    config |= (((uint64_t)mmadParams.gemvCtrl & 0x1) << 61);
    config |= (((uint64_t)mmadParams.cmatrixSource & 0x1) << 62);
    config |= (((uint64_t)mmadParams.cmatrixInitVal & 0x1) << 63);
    mad(c, a, b, config);
}

template <typename T, typename U, typename S>
__aicore__ inline void MmadCal(__cc__ T* c, __ca__ U* a, __cb__ S* b, uint64_t bias,
    const MmadParams& mmadParams, bool cmatrixSource)
{
    ASCENDC_ASSERT((cmatrixSource == mmadParams.cmatrixSource), {
        KERNEL_LOG(KERNEL_ERROR, "Mmad cmatrixSource param config error");
    });
    uint64_t config = 0;
    config |= (((uint64_t)mmadParams.m & 0xfff) << 0);
    config |= (((uint64_t)mmadParams.k & 0xfff) << 12);
    config |= (((uint64_t)mmadParams.n & 0xfff) << 24);
    config |= (((uint64_t)mmadParams.fmOffset & 0xff) << 36);
    config |= (((uint64_t)mmadParams.smaskBufferAddr & 0x7f) << 44);
    config |= (((uint64_t)mmadParams.unitFlag & 0x3) << 55);
    config |= (((uint64_t)mmadParams.s16s8rightShift & 0x1) << 57);
    config |= (((uint64_t)mmadParams.s16s8subDtype & 0x1) << 59);
    config |= (((uint64_t)mmadParams.isWeightOffset & 0x1) << 60);
    config |= (((uint64_t)mmadParams.gemvCtrl & 0x1) << 61);
    config |= (((uint64_t)mmadParams.cmatrixSource & 0x1) << 62);
    config |= (((uint64_t)mmadParams.cmatrixInitVal & 0x1) << 63);
    mad(c, a, b, bias, config);
}

/* **************************************************************************************************
 * MmadFix                                          *
 * ************************************************************************************************* */
template <typename T, typename U, typename S>
__aicore__ inline void MmadFixCal(__cbuf__ T* c, __cbuf__ U* a, __cb__ S* b,
    const MmadFixParams& mmadFixParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported MmadFixCal"); });
}

template <typename T, typename U, typename S>
__aicore__ inline void MmadFixCal(__cbuf__ T* c, __cbuf__ U* a, __cb__ S* b, uint64_t bias,
    const MmadFixParams& mmadFixParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported MmadFixCal"); });
}

/* **************************************************************************************************
 * ConvFix                                          *
 * ************************************************************************************************* */
template <typename T, typename U, typename S>
__aicore__ inline void ConvFixCal(__cbuf__ T* c, __cbuf__ U* a, __cb__ S* b,
    const ConvFixParams& convFixParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported ConvFixCal"); });
}

template <typename T, typename U, typename S>
__aicore__ inline void ConvFixCal(__cbuf__ T* c, __cbuf__ U* a, __cb__ S* b, uint64_t bias,
    const ConvFixParams& convFixParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported ConvFixCal"); });
}

__aicore__ inline void SetMatrixParaImpl(uint64_t config)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetMatrixPara on this version");
    });
}

__aicore__ inline void SetMatrixParaImpl(const MatrixParams &params)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetMatrixPara on this version");
    });
}

__aicore__ inline void SetCubeStrideParaImpl(uint64_t config)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetCubeStridePara on this version");
    });
}

__aicore__ inline void SetCubeStrideParaImpl(const CubeStrideParams &params)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetCubeStridePara on this version");
    });
}

__aicore__ inline void MmadSpCal(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, const MmadParams &mmadParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "not support mmadSp calculate"); });
}

__aicore__ inline void LoadDataWithSparseCal(const LocalTensor<int8_t> &dst, const LocalTensor<int8_t> &src,
    const LocalTensor<uint8_t> &idx, const LoadData2dParams &loadDataParam)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "not support LoadDataWithSparse calculate");
    });
}

__aicore__ inline void LoadUnzipIndexCal(const GlobalTensor<int8_t>& src, uint32_t numOfIndexTabEntry)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "not support LoadUnzipIndex");
    });
}
/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
__aicore__ inline void Load3DSetFMatrixCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    uint64_t regFMatrix = 0;
    regFMatrix |= uint64_t(l1W & 0xFFFF);

    uint32_t l1HShiftBit = 16;
    regFMatrix |= uint64_t(l1H & 0xFFFF) << l1HShiftBit;

    uint32_t padNumber = 4;
    uint32_t padListShiftBit = 8;
    uint32_t padListShiftBase = 32;
    for (uint32_t i = 0; i < padNumber; i++) {
        regFMatrix |= uint64_t(padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
    }
    set_fmatrix(regFMatrix);
}

__aicore__ inline void Load3DSetFMatrixBCal(uint16_t l1H, uint16_t l1W, const uint8_t padList[4])
{
    uint64_t regFMatrix = 0;
    regFMatrix |= (uint64_t)l1W;

    uint32_t l1HShiftBit = 16;
    regFMatrix |= (uint64_t)l1H << l1HShiftBit;
    uint32_t padNumber = 4;
    uint32_t padListShiftBit = 8;
    uint32_t padListShiftBase = 32;
    for (uint32_t i = 0; i < padNumber; i++) {
        regFMatrix |= uint64_t(padList[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
    }
    set_fmatrix_b(regFMatrix);
}

template <typename T>
__aicore__ inline void Load3DSetPaddingCal(const T padValue)
{
    uint16_t paddingValue = 0;
    uint16_t padValueShiftBit = 8;

    if constexpr (sizeof(T) == B16_BYTE_SIZE) {
        paddingValue = (uint16_t)GetScalarBitcodeValue((T)padValue);
    } else if constexpr (sizeof(T) == B32_BYTE_SIZE) {
        paddingValue = (uint32_t)GetScalarBitcodeValue((T)padValue);
    } else {
        paddingValue = (((uint16_t)padValue) << padValueShiftBit) | (uint16_t)padValue;
    }
    set_padding(paddingValue);
}

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV1L12L0ACal(__ca__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v1 from l1 to l0a"); });
}

template <typename T>
__aicore__ inline void LoadData3DV1L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v1 from l1 to l0b"); });
}

template <typename T>
__aicore__ inline void LoadData3DV1L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV1<T>& loadDataParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v1 from l1 to ubuf"); });
}

/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData3DV2L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2<T>& loadDataParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 from l1 to ubuf"); });
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ int8_t* dst, __cbuf__ int8_t* src,
    const LoadData3DParamsV2<int8_t>& loadDataParams)
{
    ASCENDC_ASSERT((false),
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 load int8_t dtype from l1 to cb"); });
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ uint8_t* dst, __cbuf__ uint8_t* src,
    const LoadData3DParamsV2<uint8_t>& loadDataParams)
{
    ASCENDC_ASSERT((false),
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 load uint8_t dtype from l1 to cb"); });
}

/* **************************************************************************************************
 * BroadCastVecToMM                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void BroadCastVecToMMCal(__cc__ T* dst, __ubuf__ T* src, const int32_t blockCount,
    const uint8_t blockLen, const uint8_t srcGap, const uint8_t dstGap)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported broadcast from ub to l0c"); });
}

/* **************************************************************************************************
 * InitL1Buffer                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL1BufferCal(__cbuf__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        T tmpValue = initConstValueParams.initValue;
        // 由于set_l0_set_value_h只支持half类型，故需要将其它类型的数据的二进制用half
        half initValue = *(half*)(&tmpValue);
        set_l0_set_value_h(initValue);
        int64_t config = 0;
        config |= (uint64_t)initConstValueParams.repeatTimes;
        config |= (uint64_t)initConstValueParams.blockNum << 16;
        config |= (uint64_t)initConstValueParams.dstGap << 32;
        set_l1_2d((__cbuf__ half*)dst, config);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported the data type!"); });
    }
}
/* **************************************************************************************************
 * InitL0ANzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0ANzMatrixCal(__ca__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        T tmpValue = initConstValueParams.initValue;
        // 由于set_l0_set_value_h只支持half类型，故需要将其它类型的数据的二进制用half
        half initValue = *(half*)(&tmpValue);
        set_l0_set_value_h(initValue);
        int64_t config = 0;
        config |= (uint64_t)initConstValueParams.repeatTimes;
        config |= (uint64_t)initConstValueParams.blockNum << 16;
        config |= (uint64_t)initConstValueParams.dstGap << 32;
        set_l0a_2d((__ca__ half*)dst, config);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported the data type!"); });
    }
}

/* **************************************************************************************************
 * InitL0BNzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0BNzMatrixCal(__cb__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    if constexpr (B16_BYTE_SIZE == sizeof(T)) {
        T tmpValue = initConstValueParams.initValue;
        // 由于set_l0_set_value_h只支持half类型，故需要将其它类型的数据的二进制用half
        half initValue = *(half*)(&tmpValue);
        set_l0_set_value_h(initValue);
        int64_t config = 0;
        config |= (uint64_t)initConstValueParams.repeatTimes;
        config |= (uint64_t)initConstValueParams.blockNum << 16;
        config |= (uint64_t)initConstValueParams.dstGap << 32;
        set_l0b_2d((__cb__ half*)dst, config);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported the data type!"); });
    }
}

/* **************************************************************************************************
 * SetLoadDataRepeat                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataRepeatCal(const LoadDataRepeatParam& repeatParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetLoadDataRepeat");
    });
}

/* **************************************************************************************************
 * SetLoadDataBoundary                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataBoundaryCal(uint32_t boundaryValue)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetLoadDataBoundary");
    });
}

/* **************************************************************************************************
 * LoadImageToLocalCal                                            *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadImageToLocalCal(__cbuf__ T *dst, const LoadImageToLocalParams &loadDataParams)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "unsupported LoadImageToLocal"); });
}

/* **************************************************************************************************
 * LoadDataUnzip                                            *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadDataUnzipToL1Cal(__cbuf__ T *dst, __gm__ T *src)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "unsupported LoadDataUnzip");
    });
}

template <typename T>
__aicore__ inline void LoadDataUnzipToL0BCal(__cb__ T *dst, __gm__ T *src)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "unsupported LoadDataUnzip");
    });
}

template <typename T>
__aicore__ inline void LoadDataUnzipToL0ACal(__ca__ T *dst, __gm__ T *src)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "unsupported LoadDataUnzip");
    });
}

__aicore__ inline void SetLoad2DsetMte2QTable0FlagCal(const Mte2QTableParam& qTableParam)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetMte2QTable0Flag");
    });
}

__aicore__ inline void SetLoad2DsetMte2QTable1FlagCal(const Mte2QTableParam& qTableParam)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetMte2QTable1Flag");
    });
}

__aicore__ inline void SetMte2SrcParaFlagCal(uint64_t kStride)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetMte2SrcParaFlag");
    });
}

template <typename T>
__aicore__ inline void Load3DSetPaddingCal(const LoadDataPaddingParam<T> &param)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetLoadDataPaddingValue");
    });
}

template <typename T>
__aicore__ inline void Load3DSetPaddingBCal(const LoadDataPaddingParam<T> &param)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetLoadDataPaddingValue");
    });
}

__aicore__ inline void SetLoadDataRepeatBCal(const LoadDataRepeatParam& repeatParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetLoadDataRepeat");
    });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_MM_IMPL_H
