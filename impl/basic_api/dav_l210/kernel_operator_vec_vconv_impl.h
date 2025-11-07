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
 * \file kernel_operator_vec_vconv_impl.h
 * \brief AscendC l210 support vector cast memory base api.
 */

#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H

#include "kernel_utils.h"
#include "kernel_operator.h"

namespace AscendC {

#define REGISTER_CAST_LV2_NOT_SUPPORTED(round_str)                                                       \
    template <typename T, typename U>                                                      \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ T *dst, __ubuf__ U *src, \
                                                         const uint32_t count)                        \
    {                                                                                                    \
        ASCENDC_ASSERT((false), {                                                                        \
            KERNEL_LOG(KERNEL_ERROR,                                                                     \
                       "round_str from U to T not supported on this device!");              \
        });                                                                                              \
    }

    REGISTER_CAST_LV2_NOT_SUPPORTED(CastNone);
    REGISTER_CAST_LV2_NOT_SUPPORTED(CastRint);
    REGISTER_CAST_LV2_NOT_SUPPORTED(CastRound);
    REGISTER_CAST_LV2_NOT_SUPPORTED(CastFloor);
    REGISTER_CAST_LV2_NOT_SUPPORTED(CastCeil);
    REGISTER_CAST_LV2_NOT_SUPPORTED(CastTrunc);
    REGISTER_CAST_LV2_NOT_SUPPORTED(CastOdd);

// 32bit->16bit/16bit->8bit
// vcvt
#define VCVT_TO_HALF(DST_TYPE, SRC_TYPE, DST_REG, SRC_REG,                                                \
                     dst_size, src_size, round_str, round_mode)                                           \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ DST_TYPE *dst,                          \
                                                         __ubuf__ SRC_TYPE *src, const uint32_t count) \
    {                                                                                                     \
        __VEC_SCOPE__                                                                                     \
        {                                                                                                 \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##dst_size(count); i++)                        \
            {                                                                                             \
                vector_##SRC_REG input0;                                                                  \
                vector_##SRC_REG input1;                                                                  \
                vector_##SRC_REG input_even;                                                              \
                vector_##SRC_REG input_odd;                                                               \
                vector_##DST_REG output;                                                                  \
                vector_address idx0 = vag_b##src_size(ELE_CNT_B##dst_size);                               \
                vector_address idx1 = vag_b##dst_size(ELE_CNT_B##dst_size);                               \
                                                                                                          \
                vector_bool pg = vpd_b##dst_size();                                                       \
                                                                                                          \
                vld(input0, (__ubuf__ SRC_TYPE *)src, idx0, NORM);                                        \
                vld(input1, (__ubuf__ SRC_TYPE *)src + ELE_CNT_B##src_size, idx0, NORM);                  \
                vdintlv(input_even, input_odd, input0, input1);                                           \
                vscvt(output, input_even, PART_EVEN);                                                     \
                vscvt(output, input_odd, PART_ODD, MODE_MERGING);                                         \
                vst(output, dst, idx1, NORM_B##dst_size, pg);                                             \
            }                                                                                             \
        }                                                                                                 \
    }

    VCVT_TO_HALF(uint16_t, uint32_t, u16, u32, 16, 32, CastNone, ROUND_R);
    VCVT_TO_HALF(int16_t, uint32_t, s16, u32, 16, 32, CastNone, ROUND_R);
    VCVT_TO_HALF(uint16_t, int32_t, u16, s32, 16, 32, CastNone, ROUND_R);
    VCVT_TO_HALF(int16_t, int32_t, s16, s32, 16, 32, CastNone, ROUND_R);
    VCVT_TO_HALF(uint8_t, uint16_t, u8, u16, 8, 16, CastNone, ROUND_R);
    VCVT_TO_HALF(uint8_t, int16_t, u8, s16, 8, 16, CastNone, ROUND_R);

// vfcvt
#define VFCVT_TO_HALF(DST_TYPE, SRC_TYPE, DST_REG, SRC_REG,                                               \
                      dst_size, src_size, round_str, round_mode, cast_func)                               \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ DST_TYPE *dst,                          \
                                                         __ubuf__ SRC_TYPE *src, const uint32_t count) \
    {                                                                                                     \
        __VEC_SCOPE__                                                                                     \
        {                                                                                                 \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##dst_size(count); i++)                        \
            {                                                                                             \
                vector_##SRC_REG input0;                                                                  \
                vector_##SRC_REG input1;                                                                  \
                vector_##SRC_REG input_even;                                                              \
                vector_##SRC_REG input_odd;                                                               \
                vector_##DST_REG output;                                                                  \
                vector_address idx0 = vag_b##src_size(ELE_CNT_B##dst_size);                               \
                vector_address idx1 = vag_b##dst_size(ELE_CNT_B##dst_size);                               \
                                                                                                          \
                vector_bool pg = vpd_b##dst_size();                                                       \
                                                                                                          \
                vld(input0, (__ubuf__ SRC_TYPE *)src, idx0, NORM);                                        \
                vld(input1, (__ubuf__ SRC_TYPE *)src + ELE_CNT_B##src_size, idx0, NORM);                  \
                vdintlv(input_even, input_odd, input0, input1);                                           \
                cast_func(output, input_even, round_mode, PART_EVEN);                                     \
                cast_func(output, input_odd, round_mode, PART_ODD, MODE_MERGING);                         \
                vst(output, dst, idx1, NORM_B##dst_size, pg);                                             \
            }                                                                                             \
        }                                                                                                 \
    }

    VFCVT_TO_HALF(half, float, f16, f32, 16, 32, CastRint, ROUND_R, vfcvt);
    VFCVT_TO_HALF(half, float, f16, f32, 16, 32, CastOdd, ROUND_O, vfcvt);
    VFCVT_TO_HALF(half, float, f16, f32, 16, 32, CastNone, ROUND_R, vfcvt);

    VFCVT_TO_HALF(int16_t, float, s16, f32, 16, 32, CastRint, ROUND_R, vsfcvt);
    VFCVT_TO_HALF(int16_t, float, s16, f32, 16, 32, CastTrunc, ROUND_Z, vsfcvt);

    VFCVT_TO_HALF(int8_t, half, s8, f16, 8, 16, CastRint, ROUND_R, vsfcvt);
    VFCVT_TO_HALF(int8_t, half, s8, f16, 8, 16, CastRound, ROUND_A, vsfcvt);
    VFCVT_TO_HALF(int8_t, half, s8, f16, 8, 16, CastFloor, ROUND_F, vsfcvt);
    VFCVT_TO_HALF(int8_t, half, s8, f16, 8, 16, CastCeil, ROUND_C, vsfcvt);
    VFCVT_TO_HALF(int8_t, half, s8, f16, 8, 16, CastTrunc, ROUND_Z, vsfcvt);
    VFCVT_TO_HALF(int8_t, half, s8, f16, 8, 16, CastNone, ROUND_R, vsfcvt);

    VFCVT_TO_HALF(uint8_t, half, u8, f16, 8, 16, CastRint, ROUND_R, vsfcvt);
    VFCVT_TO_HALF(uint8_t, half, u8, f16, 8, 16, CastRound, ROUND_A, vsfcvt);
    VFCVT_TO_HALF(uint8_t, half, u8, f16, 8, 16, CastFloor, ROUND_F, vsfcvt);
    VFCVT_TO_HALF(uint8_t, half, u8, f16, 8, 16, CastCeil, ROUND_C, vsfcvt);
    VFCVT_TO_HALF(uint8_t, half, u8, f16, 8, 16, CastTrunc, ROUND_Z, vsfcvt);
    VFCVT_TO_HALF(uint8_t, half, u8, f16, 8, 16, CastNone, ROUND_R, vsfcvt);

// 16bit->32bit/8bit->16bit(except f162s32)
#define VCVT_TO_DOUBLE(DST_TYPE, SRC_TYPE, DST_REG, SRC_REG,                                              \
                       dst_size, src_size, round_str, round_mode, cast_func)                              \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ DST_TYPE *dst,                          \
                                                         __ubuf__ SRC_TYPE *src, const uint32_t count) \
    {                                                                                                     \
        __VEC_SCOPE__                                                                                     \
        {                                                                                                 \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##src_size(count); i++)                        \
            {                                                                                             \
                vector_##SRC_REG input;                                                                   \
                vector_##DST_REG output0;                                                                 \
                vector_##DST_REG output1;                                                                 \
                vector_##DST_REG output_even;                                                             \
                vector_##DST_REG output_odd;                                                              \
                vector_address idx0 = vag_b##src_size(ELE_CNT_B##src_size);                               \
                vector_address idx1 = vag_b##dst_size(ELE_CNT_B##src_size);                               \
                vector_bool p0 = vpd_b##src_size();                                                       \
                vector_bool p1;                                                                           \
                vector_bool p2;                                                                           \
                vld(input, (__ubuf__ SRC_TYPE *)src, idx0, NORM);                                         \
                cast_func(output_even, input, PART_EVEN);                                                 \
                cast_func(output_odd, input, PART_ODD);                                                   \
                vintlv(output0, output1, output_even, output_odd);                                        \
                punpack(p1, p0, LOWER);                                                                   \
                punpack(p2, p0, HIGHER);                                                                  \
                vst(output0, dst, idx1, NORM_B##dst_size, p1);                                            \
                vst(output1, dst + ELE_CNT_B##dst_size, idx1, NORM_B##dst_size, p2);                      \
            }                                                                                             \
        }                                                                                                 \
    }

    VCVT_TO_DOUBLE(float, half, f32, f16, 32, 16, CastNone, ROUND_R, vfcvt);
    VCVT_TO_DOUBLE(float, int16_t, f32, s16, 32, 16, CastNone, ROUND_R, vfcvt);
    VCVT_TO_DOUBLE(half, uint8_t, f16, u8, 16, 8, CastNone, ROUND_R, vfcvt);
    VCVT_TO_DOUBLE(half, int8_t, f16, s8, 32, 16, CastNone, ROUND_R, vfcvt);

    VCVT_TO_DOUBLE(uint16_t, uint8_t, u16, u8, 16, 8, CastNone, ROUND_R, vcvt);
    VCVT_TO_DOUBLE(int16_t, int8_t, s16, s8, 16, 8, CastNone, ROUND_R, vcvt);
    VCVT_TO_DOUBLE(uint32_t, uint16_t, u32, u16, 32, 16, CastNone, ROUND_R, vcvt);
    VCVT_TO_DOUBLE(uint32_t, int16_t, u32, s16, 32, 16, CastNone, ROUND_R, vcvt);
    VCVT_TO_DOUBLE(int32_t, int16_t, s32, s16, 32, 16, CastNone, ROUND_R, vcvt);

#define F16_TO_S32(round_str, round_mode)                                                             \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ int32_t *dst,                       \
                                                         __ubuf__ half *src, const uint32_t count) \
    {                                                                                                 \
        __VEC_SCOPE__                                                                                 \
        {                                                                                             \
            for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); i++)                            \
            {                                                                                         \
                vector_f16 input;                                                                     \
                vector_s32 output0;                                                                   \
                vector_s32 output1;                                                                   \
                vector_s32 output_even;                                                               \
                vector_s32 output_odd;                                                                \
                vector_address idx0 = vag_b16(ELE_CNT_B16);                                           \
                vector_address idx1 = vag_b32(ELE_CNT_B16);                                           \
                                                                                                      \
                vector_bool p0 = vpd_b16();                                                           \
                vector_bool p1;                                                                       \
                vector_bool p2;                                                                       \
                vld(input, (__ubuf__ half *)src, idx0, NORM);                                         \
                vfcvt(output_even, input, round_mode, PART_EVEN);                                     \
                vfcvt(output_odd, input, round_mode, PART_ODD);                                       \
                vintlv(output0, output1, output_even, output_odd);                                    \
                punpack(p1, p0, LOWER);                                                               \
                punpack(p2, p0, HIGHER);                                                              \
                vst(output0, dst, idx1, NORM_B32, p1);                                                \
                vst(output1, dst + ELE_CNT_B32, idx1, NORM_B32, p2);                                  \
            }                                                                                         \
        }                                                                                             \
    }

    F16_TO_S32(CastRint, ROUND_R);
    F16_TO_S32(CastRound, ROUND_A);
    F16_TO_S32(CastFloor, ROUND_F);
    F16_TO_S32(CastCeil, ROUND_C);
    F16_TO_S32(CastTrunc, ROUND_Z);

#define US32_TO_U8(SRC_TYPE, SRC_REG)                                                                \
    __aicore__ inline void CastIntrinsicsImplCastNone(__ubuf__ uint8_t *dst, __ubuf__ SRC_TYPE *src, \
                                                      const uint32_t count)                       \
    {                                                                                                \
        __VEC_SCOPE__                                                                                \
        {                                                                                            \
            for (uint16_t i = 0; i <= get_vloopn_bound_b8(count); i++)                            \
            {                                                                                        \
                vector_bool pg = vpd_b8();                                                           \
                vector_##SRC_REG input0;                                                             \
                vector_##SRC_REG input1;                                                             \
                vector_##SRC_REG input2;                                                             \
                vector_##SRC_REG input3;                                                             \
                vector_u8 output;                                                                    \
                vector_address idx0 = vag_b32(ELE_CNT_B8);                                           \
                vector_address idx1 = vag_b8(ELE_CNT_B8);                                            \
                                                                                                     \
                vld(input0, (__ubuf__ SRC_TYPE *)src, idx0, NORM);                                   \
                vld(input1, (__ubuf__ SRC_TYPE *)src + ELE_CNT_B32, idx0, NORM);                     \
                vld(input2, (__ubuf__ SRC_TYPE *)src + 2 * ELE_CNT_B32, idx0, NORM);                 \
                vld(input3, (__ubuf__ SRC_TYPE *)src + 3 * ELE_CNT_B32, idx0, NORM);                 \
                                                                                                     \
                vdintlv(input0, input1, input0, input1);                                             \
                vdintlv(input2, input3, input2, input3);                                             \
                vdintlv(input0, input2, input0, input2);                                             \
                vdintlv(input1, input3, input1, input3);                                             \
                                                                                                     \
                vscvt(output, input0, PART_P0);                                                      \
                vscvt(output, input1, PART_P1, MODE_MERGING);                                        \
                vscvt(output, input2, PART_P2, MODE_MERGING);                                        \
                vscvt(output, input3, PART_P3, MODE_MERGING);                                        \
                vst(output, dst, idx1, NORM_B8, pg);                                                 \
            }                                                                                        \
        }                                                                                            \
    }

    US32_TO_U8(uint32_t, u32);
    US32_TO_U8(int32_t, s32);

#define US8_TO_US32(DST_TYPE, SRC_TYPE, DST_REG, SRC_REG)                                             \
    __aicore__ inline void CastIntrinsicsImplCastNone(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, \
                                                      const uint32_t count)                        \
    {                                                                                                 \
        __VEC_SCOPE__                                                                                 \
        {                                                                                             \
            for (uint16_t i = 0; i <= get_vloopn_bound_b8(count); i++)                             \
            {                                                                                         \
                vector_bool p0 = vpd_b8();                                                            \
                vector_##SRC_REG input;                                                               \
                vector_##DST_REG output0;                                                             \
                vector_##DST_REG output1;                                                             \
                vector_##DST_REG output2;                                                             \
                vector_##DST_REG output3;                                                             \
                vector_address idx0 = vag_b8(ELE_CNT_B8);                                             \
                vector_address idx1 = vag_b32(ELE_CNT_B8);                                            \
                                                                                                      \
                vld(input, (__ubuf__ SRC_TYPE *)src, idx0, NORM);                                     \
                                                                                                      \
                vcvt(output0, input, PART_P0);                                                        \
                vcvt(output1, input, PART_P1);                                                        \
                vcvt(output2, input, PART_P2);                                                        \
                vcvt(output3, input, PART_P3);                                                        \
                                                                                                      \
                vintlv(output0, output2, output0, output2);                                           \
                vintlv(output1, output3, output1, output3);                                           \
                vintlv(output0, output1, output0, output1);                                           \
                vintlv(output2, output3, output2, output3);                                           \
                                                                                                      \
                vector_bool p1, p2, p3, p4, p5, p6;                                                   \
                punpack(p1, p0, LOWER);                                                               \
                punpack(p2, p0, HIGHER);                                                              \
                punpack(p3, p1, LOWER);                                                               \
                punpack(p4, p1, HIGHER);                                                              \
                punpack(p5, p2, LOWER);                                                               \
                punpack(p6, p2, HIGHER);                                                              \
                                                                                                      \
                vst(output0, dst, idx1, NORM_B32, p3);                                                \
                vst(output1, dst + ELE_CNT_B32, idx1, NORM_B32, p4);                                  \
                vst(output2, dst + 2 * ELE_CNT_B32, idx1, NORM_B32, p5);                              \
                vst(output3, dst + 3 * ELE_CNT_B32, idx1, NORM_B32, p6);                              \
            }                                                                                         \
        }                                                                                             \
    }

    US8_TO_US32(uint32_t, uint8_t, u32, u8);
    US8_TO_US32(int32_t, int8_t, s32, s8);

#define F32_TO_S32(round_str, round_mode)                                                              \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ int32_t *dst,                        \
                                                         __ubuf__ float *src, const uint32_t count) \
    {                                                                                                  \
        __VEC_SCOPE__                                                                                  \
        {                                                                                              \
            for (uint16_t i = 0; i <= get_vloopn_bound_b32(count); i++)                             \
            {                                                                                          \
                vector_bool pg = vpd_b32();                                                            \
                vector_f32 input;                                                                      \
                vector_s32 output;                                                                     \
                                                                                                       \
                uint32_t idx = vag_b32(ELE_CNT_B32);                                                   \
                vld(input, src, idx, NORM);                                                            \
                vsfcvt(output, input, pg, round_mode);                                                 \
                vst(output, dst, idx, NORM_B32, pg);                                                   \
            }                                                                                          \
        }                                                                                              \
    }

    F32_TO_S32(CastRint, ROUND_R);
    F32_TO_S32(CastRound, ROUND_A);
    F32_TO_S32(CastFloor, ROUND_F);
    F32_TO_S32(CastCeil, ROUND_C);
    F32_TO_S32(CastTrunc, ROUND_Z);

#define EQUAL_CAST(DST_TYPE, SRC_TYPE, DST_REG, SRC_REG, SIZE, CAST_FUNC)                             \
    __aicore__ inline void CastIntrinsicsImplCastNone(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, \
                                                      const uint32_t count)                        \
    {                                                                                                 \
        __VEC_SCOPE__                                                                                 \
        {                                                                                             \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##SIZE(count); i++)                        \
            {                                                                                         \
                vector_bool pg = vpd_b##SIZE();                                                       \
                vector_##SRC_REG input;                                                               \
                vector_##DST_REG output;                                                              \
                                                                                                      \
                uint32_t idx = vag_b##SIZE(ELE_CNT_B##SIZE);                                          \
                vld(input, src, idx, NORM);                                                           \
                CAST_FUNC(output, input, pg);                                                         \
                vst(output, dst, idx, NORM_B##SIZE, pg);                                              \
            }                                                                                         \
        }                                                                                             \
    }

    EQUAL_CAST(int16_t, half, s16, f16, 16, vsfcvt);
    EQUAL_CAST(half, int16_t, f16, s16, 16, vfcvt);
    EQUAL_CAST(float, int32_t, f32, s32, 32, vfcvt);

    // s32->f16
    __aicore__ inline void CastIntrinsicsImplCastNone(__ubuf__ half *dst, __ubuf__ int32_t *src,
                                                      const uint32_t count)
    {
        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i <= get_vloopn_bound_b16(count); i++)
            {
                vector_s32 input0;
                vector_s32 input1;
                vector_f32 input2;
                vector_f32 input3;
                vector_f32 input_even;
                vector_f32 input_odd;
                vector_f16 output;
                vector_address idx0 = vag_b32(ELE_CNT_B16);
                vector_address idx1 = vag_b16(ELE_CNT_B16);

                vector_bool p0 = pge_b32(PAT_ALL);
                vector_bool p1 = vpd_b16();

                vld(input0, src, idx0, NORM);
                vld(input1, src + ELE_CNT_B32, idx0, NORM);
                vfcvt(input2, input0, p0);
                vfcvt(input3, input1, p0);
                vdintlv(input_even, input_odd, input2, input3);
                vfcvt(output, input_even, ROUND_R, PART_EVEN);
                vfcvt(output, input_odd, ROUND_R, PART_ODD, MODE_MERGING);
                vst(output, dst, idx1, NORM_B16, p1);
            }
        }
    }

#define TRUNCATE(round_str, round_mode, DATA_TYPE, REG_TYPE, SIZE)                                         \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ DATA_TYPE *dst,                          \
                                                         __ubuf__ DATA_TYPE *src, const uint32_t count) \
    {                                                                                                      \
        __VEC_SCOPE__                                                                                      \
        {                                                                                                  \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##SIZE(count); i++)                             \
            {                                                                                              \
                vector_bool pg = vpd_b##SIZE();                                                            \
                vector_##REG_TYPE input;                                                                   \
                vector_##REG_TYPE output;                                                                  \
                                                                                                           \
                uint32_t idx = vag_b##SIZE(ELE_CNT_B##SIZE);                                               \
                vld(input, src, idx, NORM);                                                                \
                vtrc(output, input, round_mode, pg);                                                       \
                vst(output, dst, idx, NORM_B32, pg);                                                       \
            }                                                                                              \
        }                                                                                                  \
    }

    TRUNCATE(CastRint, ROUND_R, float, f32, 32);
    TRUNCATE(CastRound, ROUND_A, float, f32, 32);
    TRUNCATE(CastFloor, ROUND_F, float, f32, 32);
    TRUNCATE(CastCeil, ROUND_C, float, f32, 32);
    TRUNCATE(CastTrunc, ROUND_Z, float, f32, 32);
    TRUNCATE(CastRint, ROUND_R, half, f16, 16);
    TRUNCATE(CastRound, ROUND_A, half, f16, 16);
    TRUNCATE(CastFloor, ROUND_F, half, f16, 16);
    TRUNCATE(CastCeil, ROUND_C, half, f16, 16);
    TRUNCATE(CastTrunc, ROUND_Z, half, f16, 16);

    // Cast::Level 2
    template <typename T, typename U>
    __aicore__ inline void CastImpl(__ubuf__ T *dst, __ubuf__ U *src, const RoundMode &roundMode,
                                    const uint32_t count)
    {
        switch (roundMode)
        {
        case RoundMode::CAST_RINT:
            CastIntrinsicsImplCastRint(dst, src, count);
            break;
        case RoundMode::CAST_FLOOR:
            CastIntrinsicsImplCastFloor(dst, src, count);
            break;
        case RoundMode::CAST_CEIL:
            CastIntrinsicsImplCastCeil(dst, src, count);
            break;
        case RoundMode::CAST_ROUND:
            CastIntrinsicsImplCastRound(dst, src, count);
            break;
        case RoundMode::CAST_TRUNC:
            CastIntrinsicsImplCastTrunc(dst, src, count);
            break;
        case RoundMode::CAST_ODD:
            CastIntrinsicsImplCastOdd(dst, src, count);
            break;
        case RoundMode::CAST_NONE:
            CastIntrinsicsImplCastNone(dst, src, count);
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
        }
    }

    template <typename T, typename U, bool isVecDeq, bool halfBlock>
    __aicore__ inline void CastDeqImpl(__ubuf__ T *dst, __ubuf__ U *src,
                                       const uint32_t count)
    {
        ASCENDC_ASSERT((false), "CastDeq is not supported on this device");
    }

    template <typename T, typename U>
    __aicore__ inline void AddReluCastImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1,
                                           const uint32_t count)
    {
        ASCENDC_ASSERT((false), "AddReluCast is not supported on this device");
    }

    template <typename T, typename U>
    __aicore__ inline void SubReluCastImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1,
                                           const uint32_t count)
    {
        ASCENDC_ASSERT((false), "SubReluCast is not supported on this device");
    }

    __aicore__ inline void SetDeqScaleImpl(float scale, int16_t offset, bool signMode)
    {
        ASCENDC_ASSERT((false), "SetDeqScale is not supported on this device");
    }

    template <typename T>
    __aicore__ inline void SetDeqScaleImpl(const LocalTensor<T> &vdeq, const VdeqInfo &vdeqInfo)
    {
        ASCENDC_ASSERT((false), "SetDeqScale is not supported on this device");
    }

    template <typename T>
    __aicore__ inline void SetDeqScaleImpl(T config)
    {
        ASCENDC_ASSERT((false), "SetDeqScale is not supported on this device");
    }

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
