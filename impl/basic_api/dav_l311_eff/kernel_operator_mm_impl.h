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

// **** cube isabit begin ****
#define CUBE_REG_BIAS_VAL_0 (0b1)
#define CUBE_REG_BIAS_VAL_REAL (0b0)
#define CUBE_REG_BIAS_NO_BRC (0b0)
#define CUBE_REG_BIAS_EN_BRC (0b1)
#define MATMUL_REG_EN_GEMV (0b0)
#define MATMUL_REG_DIS_GEMV (0b1)
#define CUBE_REG_WINO_MODE_1 (0b1)

// **** cube isabit end ****

// **** fixp isabit begin ****
// prequant
#define FIXP_REG_PREQUANT_NONE                (0b00000)
#define FIXP_REG_PREQUANT_REQ8_INT_VEC        (0b00010)
#define FIXP_REG_PREQUANT_REQ8_INT_SCALAR     (0b00011)
#define FIXP_REG_PREQUANT_DEQF16_VEC          (0b00100)
#define FIXP_REG_PREQUANT_DEQF16_SCALAR       (0b00101)
#define FIXP_REG_PREQUANT_DEQS16_INT_VEC      (0b00110)
#define FIXP_REG_PREQUANT_DEQS16_INT_SCALAR   (0b00111)
#define FIXP_REG_PREQUANT_DEQS32_INT_VEC      (0b01000)
#define FIXP_REG_PREQUANT_DEQS32_INT_SCALAR   (0b01001)
#define FIXP_REG_PREQUANT_REQ4_INT_VEC        (0b01010)
#define FIXP_REG_PREQUANT_REQ4_INT_SCALAR     (0b01011)

// pre acitivation, 除了LUT外，pre和post的bit表示无差别
#define FIXP_REG_NO_RELU     (0b000)
#define FIXP_REG_NORMAL_RELU (0b001)
#define FIXP_REG_SCALAR_RELU (0b010) // Leakyrelu
#define FIXP_REG_VECTOR_RELU (0b011) // Prelu
#define FIXP_REG_LUT         (0b100)

// postquant
#define FIXP_REG_PREQUANT_NONE                   (0b00000)
#define FIXP_REG_POSTQUANT_SHIFT2S16_POST_SCALAR (0b00001)
#define FIXP_REG_POSTQUANT_SHIFT2S16_POST_VEC    (0b00010)
#define FIXP_REG_POSTQUANT_SHIFT2S8_POST_SCALAR  (0b00011)
#define FIXP_REG_POSTQUANT_SHIFT2S8_POST_VEC     (0b00100)
#define FIXP_REG_POSTQUANT_SHIFT2S4_POST_SCALAR  (0b00101)
#define FIXP_REG_POSTQUANT_SHIFT2S4_POST_VEC     (0b00110)

// pre/post clipRelu
#define FIXP_REG_NO_CLIPRELU (0b0)
#define FIXP_REG_CLIPRELU    (0b1)

// eltwise
#define FIXP_REG_ELT_NONE   (0b000)
#define FIXP_REG_ELT_OP_ADD (0b001)
#define FIXP_REG_ELT_OP_SUB (0b100)
#define FIXP_REG_ELT_OP_MUL (0b011)
#define FIXP_REG_ELT_OP_MAX (0b100)

#define FIXP_REG_ELT_NONE_BRC (0b0)
#define FIXP_REG_ELT_BRC (0b1)

// anti-quant
#define FIXP_REG_NO_ANTIQUANT (0b0)
#define FIXP_REG_ANTIQUANT_EN (0b1)

#define FIXP_REG_ANTIQUANT_S8_INT_SCALAR      (0b0000)
#define FIXP_REG_ANTIQUANT_S8_INT_VEC         (0b0001)
#define FIXP_REG_ANTIQUANT_S4_INT_SCALAR      (0b0010)
#define FIXP_REG_ANTIQUANT_S4_INT_VEC         (0b0011)
#define FIXP_REG_ANTIQUANT_U8_INT_SCALAR      (0b0100)
#define FIXP_REG_ANTIQUANT_U8_INT_VEC         (0b0101)
#define FIXP_REG_ANTIQUANT_S16_INT_SCALAR     (0b0110)
#define FIXP_REG_ANTIQUANT_S16_INT_VEC        (0b0111)

// bitmask
#define FIXP_REG_BITMASK_NONE  (0b000)
#define FIXP_REG_BITMASK_1_LSB (0b001)
#define FIXP_REG_BITMASK_2_LSB (0b010)
#define FIXP_REG_BITMASK_3_LSB (0b011)
#define FIXP_REG_BITMASK_4_LSB (0b100)

// dualsrc
#define FIXP_REG_DIS_DUAL_SRC (0b0)
#define FIXP_REG_EN_DUAL_SRC (0b1)

// compact
#define FIXP_REG_DIS_COMPACT (0b0)
#define FIXP_REG_EN_COMPACT (0b1)
// **** fixp isabit end ****

// **** clear isabit begin ****
#define CLEAR_REG_BIAS_BIT      (0xFFFFFFFFFFFFFFFC) // [1:0]
#define CLEAR_REG_PREQUANT_BIT  (0xFFFFFFFFFFFFFF83) // [6:2]
#define CLEAR_REG_RELU_BIT      (0xFFFFFFFFFFFFFC7F) // [9:7]
#define CLEAR_REG_POSTQUANT_BIT (0xFFFFFFFFFFFF83FF) // [14:10]
#define CLEAR_REG_CLIPRELU_BIT  (0xFFFFFFFFFFFBFFFF) // [18]
#define CLEAR_REG_ELT_BIT       (0xFFFFFFFFFF8FFFFF) // [22:20]
#define CLEAR_REG_ANTIQUANT_BIT (0xFFFFFFFFFF7FFFFF) // [23]
#define CLEAR_REG_ELT_BRC_BIT   (0xFFFFFFFFFEFFFFFF) // [24]
#define CLEAR_REG_BITMASK_BIT   (0xFFFFFFFF8EFFFFFF) // [30:28]
#define CLEAR_REG_DUAL_SRC_BIT  (0xFFFFFFFBFFFFFFFF) // [34]
#define CLEAR_REG_COMPACT_BIT   (0xFFFFFFF7FFFFFFFF) // [35]
#define CLEAR_REG_INSTR_ID_BIT  (0xFFFFFF9FFFFFFFFF) // [38:37]
#define CLEAR_REG_CIN_SIZE_BIT  (0x0000FFFFFFFFFFFF) // [63:48]

#define CLEAR_REG_ELT_SRC_ADDR_BIT  (0xFFFFFFFF0000FFFF) // [31:16]

// 版本差异修改bit位置
#define CUBE_XT_POS_BIAS_EN 0
#define CUBE_XT_POS_BIAS_BRCMODE 1
#define CUBE_XT_POS_PREQ 2
#define CUBE_XT_POS_PREQ 2
#define CUBE_XT_POS_PREQ_ACT 7
#define CUBE_XT_POS_POSTQ 10
// #define CUBE_XT_POS_POSTQ_ACT 15 这个应该不支持
#define CUBE_XT_POS_PREQ_CLIP 18
// #define CUBE_XT_POS_POSTQ_CLIP 19 这个应该不支持
#define CUBE_XT_POS_ELT 20
#define CUBE_XT_POS_ELT_ANTIQ 23
#define CUBE_XT_POS_ELT_BRC 24
#define CUBE_XT_POS_BITMASK 28
#define CUBE_XT_POS_LOOPENHANCE 31
#define CUBE_XT_POS_LOOPENHANCE_CH_MERGE 32
#define CUBE_XT_POS_FIXP_C0_PAD 33
#define CUBE_XT_POS_DUAL_SRC 34
#define CUBE_XT_POS_COMPACT 35
#define CUBE_XT_POS_DEPEND_EN 36
#define CUBE_XT_POS_INSTR_ID 37
#define CUBE_XT_POS_WINO_MODE 46
#define CUBE_XT_POS_CIN_SIZE 48

#define MATMUL_XT_POS_TRANS 25
#define MATMUL_XT_POS_GEMV 31
#define MATMUL_XT_POS_N_SIZE 48

typedef __bf16 bfloat16_t;
namespace AscendC {
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
 * LoadData 2dv2                                             *
 * ************************************************************************************************* */

__aicore__ inline void SetLoad2DsetMte2QTable0FlagCal(const Mte2QTableParam& qTableParam)
{
    set_mte2_qtable0(qTableParam.qTable);
}

__aicore__ inline void SetLoad2DsetMte2QTable1FlagCal(const Mte2QTableParam& qTableParam)
{
    set_mte2_qtable1(qTableParam.qTable);
}

__aicore__ inline void SetMte2SrcParaFlagCal(uint64_t kStride)
{
    set_mte2_src_para(kStride);
}

/****************************L1 To L0A******************************************/
template <typename T>
__aicore__ inline void LoadData2DL12L0ACal(__ca__ T* dst, __cbuf__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LoadData A1 To A2 is not supported!"); });
}

/****************************L1 To L0B******************************************/
template <typename T>
__aicore__ inline void LoadData2DL12L0BCal(__cb__ T* dst, __cbuf__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    if constexpr (1 == sizeof(T)) {
        load_cbuf_to_cb((__cb__ int8_t *)dst, (__cbuf__ int8_t *)src, loadDataParam.mStartPosition, loadDataParam.kStartPosition, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.srcStride, loadDataParam.dstStride, false);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/****************************GM To L0A******************************************/
template <typename T>
__aicore__ inline void LoadData2DGM2L0ACal(__ca__ T* dst, __gm__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "LoadData GM To A2 is not supported!"); });
}

/****************************GM To L0B******************************************/
template <typename T>
__aicore__ inline void LoadData2DGM2L0BCal(__cb__ T* dst, __gm__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    if (loadDataParam.kStep != 1) {
        set_mte2_src_para(uint64_t(loadDataParam.srcStride));
    }
    if constexpr (1 == sizeof(T)) {
        load_gm_to_cb_2dv2((__cb__ int8_t*)dst, (__gm__ int8_t*)src, loadDataParam.mStartPosition, loadDataParam.kStartPosition, loadDataParam.dstStride,
            loadDataParam.mStep, loadDataParam.kStep, loadDataParam.sid, loadDataParam.qmode);
    } else if constexpr (2 == sizeof(T)) {
        load_gm_to_cb_2dv2((__cb__ half*)dst, (__gm__ half*)src, loadDataParam.mStartPosition, loadDataParam.kStartPosition, loadDataParam.dstStride,
            loadDataParam.mStep, loadDataParam.kStep, loadDataParam.sid, loadDataParam.qmode);
    } else if constexpr (4 == sizeof(T)) {
        load_gm_to_cb_2dv2((__cb__ float*)dst, (__gm__ float*)src, loadDataParam.mStartPosition, loadDataParam.kStartPosition, loadDataParam.dstStride,
            loadDataParam.mStep, loadDataParam.kStep, loadDataParam.sid, loadDataParam.qmode);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/****************************GM To L1******************************************/
template <typename T>
__aicore__ inline void LoadData2DGM2L1Cal(__cbuf__ T* dst, __gm__ T* src, const LoadData2DParamsV2& loadDataParam)
{
    if (loadDataParam.kStep != 1) {
        set_mte2_src_para(uint64_t(loadDataParam.srcStride));
    }
    if constexpr (1 == sizeof(T)) {
        load_gm_to_cbuf_2dv2((__cbuf__ int8_t*)dst, (__gm__ int8_t*)src, loadDataParam.mStartPosition, loadDataParam.kStartPosition, loadDataParam.dstStride,
            loadDataParam.mStep, loadDataParam.kStep, loadDataParam.sid, loadDataParam.qmode);
    } else if constexpr (2 == sizeof(T)) {
        load_gm_to_cbuf_2dv2((__cbuf__ half*)dst, (__gm__ half*)src, loadDataParam.mStartPosition, loadDataParam.kStartPosition, loadDataParam.dstStride,
            loadDataParam.mStep, loadDataParam.kStep, loadDataParam.sid, loadDataParam.qmode);
    } else if constexpr (4 == sizeof(T)) {
        load_gm_to_cbuf_2dv2((__cbuf__ float*)dst, (__gm__ float*)src, loadDataParam.mStartPosition, loadDataParam.kStartPosition, loadDataParam.dstStride,
            loadDataParam.mStep, loadDataParam.kStep, loadDataParam.sid, loadDataParam.qmode);
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
    }
}

/* **************************************************************************************************
 * LoadData 2d with transpose                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LoadData2DL12L0ATransposeCal(__ca__ T * dst, __cbuf__ T * src, const LoadData2dTransposeParams& loadDataParam)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "the position is not supported in LoadDataWithTranspose!"); });
}

template <typename T>
__aicore__ inline void LoadData2DL12L0BTransposeCal(__cb__ T * dst, __cbuf__ T * src, const LoadData2dTransposeParams& loadDataParam)
{
    if constexpr (1 == sizeof(T)) {
        load_cbuf_to_cb((__cb__ int8_t*)dst, (__cbuf__ int8_t*)src, loadDataParam.mStartPosition, loadDataParam.kStartPosition, loadDataParam.mStep,
            loadDataParam.kStep, loadDataParam.srcStride, loadDataParam.dstStride, true);
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
    if (IsSameType<T, int8_t>::value) {
        img2colv2_cbuf_to_ca(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
            loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
            loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
            loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W, loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H,
            loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
            loadDataParams.channelSize);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    if (IsSameType<T, int8_t>::value) {
        img2colv2_cbuf_to_cb(dst, src, loadDataParams.extConfig, loadDataParams.extConfig >> LOAD_M_EXTENSION,
            loadDataParams.extConfig >> LOAD_K_START_POSITION, loadDataParams.extConfig >> LOAD_M_START_POSITION,
            loadDataParams.filterConfig, loadDataParams.filterConfig >> LOAD_STRIDE_H,
            loadDataParams.filterConfig >> LOAD_FILTER_W, loadDataParams.filterConfig >> LOAD_FILTER_H,
            loadDataParams.filterConfig >> LOAD_DILATION_FILTER_W, loadDataParams.filterConfig >> LOAD_DILATION_FILTER_H,
            loadDataParams.filterSizeW, loadDataParams.filterSizeH, loadDataParams.enTranspose, loadDataParams.fMatrixCtrl,
            loadDataParams.channelSize);
    }
}

template <typename T>
__aicore__ inline void LoadData3DV2L12UBCal(__ubuf__ T* dst, __cbuf__ T* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_ASSERT((false),
                   { KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 from A1/B1 to UB on this version"); });
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ uint8_t* dst, __cbuf__ uint8_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 load uint8_t dtype from A1/B1 to B2 on this version");
    });
}

template <>
__aicore__ inline void LoadData3DV2L12L0BCal(__cb__ int8_t* dst, __cbuf__ int8_t* src,
    const LoadData3DParamsV2Pro& loadDataParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported loaddata_3d_v2 load int8_t dtype from A1/B1 to B2 on this version");
    });
}

/* **************************************************************************************************
 * Mmad                                             *
 * ************************************************************************************************* */
template <typename T, typename U, typename S>
__aicore__ inline void MmadCal(__cbuf__ T* c, __cbuf__ U* a, __cb__ S* b, const MmadParams& mmadParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported Mmad on this version");
    });
}

template <typename T, typename U, typename S>
__aicore__ inline void MmadCal(__cbuf__ T* c, __cbuf__ U* a, __cb__ S* b, uint64_t bias,
    const MmadParams& mmadParams, bool cmatrixSource)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported Mmad on this version");
    });
}


inline __aicore__ uint8_t GetQuantPreMode(QuantMode_t quantPre)
{
    uint8_t quantPreMode = 0;
    switch (quantPre) {
        case QuantMode_t::NoQuant:
            quantPreMode = 0;
            break;
        case QuantMode_t::VREQ8:
            quantPreMode = 2;
            break;
        case QuantMode_t::REQ8:
            quantPreMode = 3;
            break;
        case QuantMode_t::VDEQF16:
            quantPreMode = 4;
            break;
        case QuantMode_t::DEQF16:
            quantPreMode = 5;
            break;
        case QuantMode_t::VREQ4:
            quantPreMode = 10;
            break;
        case QuantMode_t::REQ4:
            quantPreMode = 11;
            break;
        case QuantMode_t::VDEQS16:
            quantPreMode = 6;
            break;
        case QuantMode_t::DEQS16:
            quantPreMode = 7;
            break;
        default:
            quantPreMode = 0;
            break;
    }
    return quantPreMode;
}

inline __aicore__ uint8_t GetQuantPostMode(QuantMode_post quantPost)
{
    uint8_t quantPostMode = 0;
    switch (quantPost) {
        case QuantMode_post::NoConv:
            quantPostMode = 0;
            break;
        case QuantMode_post::VSHIFT2S4_POST:
            quantPostMode = 6;
            break;
        case QuantMode_post::SHIFT2S4_POST:
            quantPostMode = 5;
            break;
        case QuantMode_post::VSHIFT2S8_POST:
            quantPostMode = 4;
            break;
        case QuantMode_post::SHIFT2S8_POST:
            quantPostMode = 3;
            break;
        case QuantMode_post::VSHIFT2S16_POST:
            quantPostMode = 2;
            break;
        case QuantMode_post::SHIFT2S16_POST:
            quantPostMode = 1;
            break;
        default:
            quantPostMode = 0;
            break;
    }
    return quantPostMode;
}

/* **************************************************************************************************
 * MmadFix                                          *
 * ************************************************************************************************* */
template <typename T, typename U, typename S>
__aicore__ inline void MmadFixCal(__cbuf__ T* c, __cbuf__ U* a, __cb__ S* b,
    const MmadFixParams& mmadFixParams)
{
    uint64_t config = 0;
    config |= ((static_cast<uint64_t>(mmadFixParams.biasMatrixInitVal) & 0x1) << 0);
    config |= ((static_cast<uint64_t>(mmadFixParams.biasMatrixBroadcast) & 0x1) << 1);
    config |= ((static_cast<uint64_t>(GetQuantPreMode(mmadFixParams.preQuantMode)) & 0x1f) << 2);
    config |= ((static_cast<uint64_t>(mmadFixParams.preReluMode) & 0x7) << 7);
    config |= ((static_cast<uint64_t>(GetQuantPostMode(mmadFixParams.postQuantMode)) & 0x1f) << 10);
    config |= ((static_cast<uint64_t>(mmadFixParams.preClipReluMode) & 0x1) << 18);
    config |= ((static_cast<uint64_t>(mmadFixParams.eltwiseOp) & 0x7) << 20);
    config |= ((static_cast<uint64_t>(mmadFixParams.eltwiseAntiqEnable) & 0x1) << 23);
    config |= ((static_cast<uint64_t>(mmadFixParams.eltwiseBroadcastEnable) & 0x1) << 24);
    config |= ((static_cast<uint64_t>(mmadFixParams.lsbMask) & 0x7) << 28);
    config |= ((static_cast<uint64_t>(mmadFixParams.gemvCtrl) & 0x1) << 31);
    config |= ((static_cast<uint64_t>(mmadFixParams.dependEnable) & 0x1) << 36);
    config |= ((static_cast<uint64_t>(mmadFixParams.instrId) & 0x3) << 37);
    config |= ((static_cast<uint64_t>(mmadFixParams.n) & 0xffff) << 48);
    matmul((__cbuf__ void*)c, a, b, (__cbuf__  int32_t *)get_imm(0), 0, config);
}

template <typename T, typename U, typename S>
__aicore__ inline void MmadFixCal(__cbuf__ T* c, __cbuf__ U* a, __cb__ S* b, uint64_t bias,
    const MmadFixParams& mmadFixParams)
{
    uint64_t config = 0;
    config |= (((uint64_t)mmadFixParams.biasMatrixInitVal & 0x1) << 0);
    config |= (((uint64_t)mmadFixParams.biasMatrixBroadcast & 0x1) << 1);
    config |= (((uint64_t)GetQuantPreMode(mmadFixParams.preQuantMode) & 0x1f) << 2);
    config |= (((uint64_t)mmadFixParams.preReluMode & 0x7) << 7);
    config |= (((uint64_t)GetQuantPostMode(mmadFixParams.postQuantMode) & 0x1f) << 10);
    config |= (((uint64_t)mmadFixParams.preClipReluMode & 0x1) << 18);
    config |= (((uint64_t)mmadFixParams.eltwiseOp & 0x7) << 20);
    config |= (((uint64_t)mmadFixParams.eltwiseAntiqEnable & 0x1) << 23);
    config |= (((uint64_t)mmadFixParams.eltwiseBroadcastEnable & 0x1) << 24);
    config |= (((uint64_t)mmadFixParams.lsbMask & 0x7) << 28);
    config |= (((uint64_t)mmadFixParams.gemvCtrl & 0x1) << 31);
    config |= (((uint64_t)mmadFixParams.dependEnable & 0x1) << 36);
    config |= (((uint64_t)mmadFixParams.instrId & 0x3) << 37);
    config |= (((uint64_t)mmadFixParams.n & 0xffff) << 48);
    if (mmadFixParams.biasMatrixBroadcast) {
        // enable bias broadcast, bias store in C2
        matmul((__cbuf__ void*)c, a, b, (__cbuf__  int32_t *)get_imm(0), bias, config);
    } else {
        // disable bias broadcast, bias store in C1
        matmul((__cbuf__ void*)c, a, b, (__cbuf__  int32_t*)bias, (uint64_t)0, config);
    }
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
    set_matrix_para(config);
}

__aicore__ inline void SetMatrixParaImpl(const MatrixParams &params)
{
    uint64_t config = 0;
    config = (((uint64_t)params.m & 0xffff) << 0) |
             (((uint64_t)params.k & 0xffff) << 16);
    set_matrix_para(config);
}

__aicore__ inline void SetCubeStrideParaImpl(uint64_t config)
{
    set_cube_stride_para(config);
}

__aicore__ inline void SetCubeStrideParaImpl(const CubeStrideParams &params)
{
    uint64_t config = 0;
    config = (((uint64_t)params.loop4SrcStride & 0xffff) << 0) |
             (((uint64_t)params.loop4DstStride & 0xffff) << 16);
    set_cube_stride_para(config);
}

/* **************************************************************************************************
 * LoadData Register                                           *
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
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported right SetFmatrix");
    }
    );
}

template <typename T>
__aicore__ inline void Load3DSetPaddingCal(const LoadDataPaddingParam<T> &param)
{
    uint64_t config = 0;
    config |= 0xFFFFFFFF & (uint64_t)param.padValue;
    config |= (uint64_t)param.quantSch << 32;
    set_padding(config);
}

template <typename T>
__aicore__ inline void Load3DSetPaddingBCal(const LoadDataPaddingParam<T> &param)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported right SetLoadDataPaddingValue"); });
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
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupported broadcast from ub to l0c in Ascend910B1"); });
}

/* **************************************************************************************************
 * InitL1Buffer                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL1BufferCal(__cbuf__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    if constexpr (2 == sizeof(T)) {
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
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "InitConstValue unsupported the position A2!"); });
}

/* **************************************************************************************************
 * InitL0BNzMatrix                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void InitL0BNzMatrixCal(__cb__ T *dst, const InitConstValueParams<T> &initConstValueParams)
{
    ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "InitConstValue unsupported the position B2!"); });
}

/* **************************************************************************************************
 * SetLoadDataRepeat                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataRepeatCal(const LoadDataRepeatParam& repeatParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetLoadDataRepeat!");
    }
    );
}

/* **************************************************************************************************
 * SetLoadDataBoundary                                             *
 * ************************************************************************************************* */
__aicore__ inline void SetLoadDataBoundaryCal(uint32_t boundaryValue)
{
    ASCENDC_ASSERT(false, {
        KERNEL_LOG(KERNEL_ERROR, "unsupported SetLoadDataBoundary!");
    }
    );
}

__aicore__ inline void SetLoadDataRepeatBCal(const LoadDataRepeatParam& repeatParams)
{
    ASCENDC_ASSERT((false), {
        KERNEL_LOG(KERNEL_ERROR, "unsupported Right SetLoadDataRepeat");
    });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_MM_IMPL_H
