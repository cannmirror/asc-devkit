/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_CUBE_DATAMOVE_IMPL_H
#define IMPL_C_API_INSTR_IMPL_NPU_ARCH_3510_CUBE_DATAMOVE_IMPL_H

#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_l0c2l1_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_l12l0a_mx_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_l13d_rpt_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_l13d_fmatrix_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_l0c_copy_prequant_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_gm2l1_loop_size_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_gm2l1_loop1_stride_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_gm2l1_loop2_stride_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_gm2l1_pad_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_l12l0a_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_l12l0b_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_gm2l1_align_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_gm2l1_dn2nz_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_gm2l1_nd2nz_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_gm2l1_impl.h"

// ==========asc_copy_l0c2l1===========
// half  float
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ half* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size,
                                            uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre,
                                            uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre,
                                            bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post,
                                            bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg,
                                            bool c0_pad_en)
{
    asc_copy_l0c2l1_impl(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode,
        quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op,
        eltwise_antq_cfg, c0_pad_en);
}

// int8_t  float
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ int8_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size,
                                            uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre,
                                            uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre,
                                            bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post,
                                            bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg,
                                            bool c0_pad_en)
{
    asc_copy_l0c2l1_impl(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode,
        quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op,
        eltwise_antq_cfg, c0_pad_en);
}

// uint8_t  float
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ uint8_t* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size,
                                            uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre,
                                            uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre,
                                            bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post,
                                            bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg,
                                            bool c0_pad_en)
{
    asc_copy_l0c2l1_impl(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode,
        quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op,
        eltwise_antq_cfg, c0_pad_en);
}

// float  float
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ float* dst, __cc__ float* src, uint16_t n_size, uint16_t m_size,
                                            uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre,
                                            uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre,
                                            bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post,
                                            bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg,
                                            bool c0_pad_en)
{
    asc_copy_l0c2l1_impl(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode,
        quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op,
        eltwise_antq_cfg, c0_pad_en);
}

// half int32_t
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ half* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size,
                                            uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre,
                                            uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre,
                                            bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post,
                                            bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg,
                                            bool c0_pad_en)
{
    asc_copy_l0c2l1_impl(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode,
        quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op,
        eltwise_antq_cfg, c0_pad_en);
}

// int8_t int32_t
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size,
                                            uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre,
                                            uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre,
                                            bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post,
                                            bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg,
                                            bool c0_pad_en)
{
    asc_copy_l0c2l1_impl(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode,
        quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op,
        eltwise_antq_cfg, c0_pad_en);
}

// uint8_t int32_t
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size,
                                            uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre,
                                            uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre,
                                            bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post,
                                            bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg,
                                            bool c0_pad_en)
{
    asc_copy_l0c2l1_impl(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode,
        quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op,
        eltwise_antq_cfg, c0_pad_en);
}

// int32_t int32_t
__aicore__ inline void asc_copy_l0c2l1(__cbuf__ int32_t* dst, __cc__ int32_t* src, uint16_t n_size, uint16_t m_size,
                                            uint32_t dst_stride, uint16_t src_stride, uint8_t clip_relu_pre,
                                            uint8_t unit_flag_mode, uint64_t quant_pre, uint8_t relu_pre,
                                            bool channel_split, bool nz2nd_en, uint64_t quant_post, uint8_t relu_post,
                                            bool clip_relu_post, uint8_t eltwise_op, uint8_t eltwise_antq_cfg,
                                            bool c0_pad_en)
{
    asc_copy_l0c2l1_impl(dst, src, n_size, m_size, dst_stride, src_stride, clip_relu_pre, unit_flag_mode,
        quant_pre, relu_pre, channel_split, nz2nd_en, quant_post, relu_post, clip_relu_post, eltwise_op,
        eltwise_antq_cfg, c0_pad_en);
}

__aicore__ inline void asc_copy_l12l0a_mx(uint64_t dst, __cbuf__ fp8_e8m0_t* src, uint16_t x_start_pos,
    uint16_t y_start_pos, uint8_t x_step, uint8_t y_step, uint16_t src_stride, uint16_t dst_stride)
{
    asc_copy_l12l0a_mx_impl(dst, src, x_start_pos, y_start_pos, x_step, y_step, src_stride, dst_stride);
}

__aicore__ inline void asc_set_l0c_copy_prequant(uint64_t config)
{
    asc_set_l0c_copy_prequant_impl(config);
}

__aicore__ inline void asc_set_gm2l1_loop_size(uint64_t loop1_size, uint64_t loop2_size)
{
    asc_set_gm2l1_loop_size_impl(loop1_size, loop2_size);
}

__aicore__ inline void asc_set_gm2l1_loop1_stride(uint64_t loop1_src_stride, uint64_t loop1_dst_stride)
{
    asc_set_gm2l1_loop1_stride_impl(loop1_src_stride, loop1_dst_stride);
}

__aicore__ inline void asc_set_gm2l1_loop2_stride(uint64_t loop2_src_stride, uint64_t loop2_dst_stride)
{
    asc_set_gm2l1_loop2_stride_impl(loop2_src_stride, loop2_dst_stride);
}

__aicore__ inline void asc_set_gm2l1_pad(uint32_t pad_val)
{
    asc_set_gm2l1_pad_impl(pad_val);
}

// ==========asc_set_l13d_rpt==========
__aicore__ inline void asc_set_l13d_rpt(asc_load3d_v2_config& config)
{
    asc_set_l13d_rpt_impl(config);
}

// ==========asc_set_l13d_fmatrix==========
__aicore__ inline void asc_set_l13d_fmatrix(asc_l13d_fmatrix_config& config)
{
    asc_set_l13d_fmatrix_impl(config);
}

// ==========asc_copy_l12l0a==========
__aicore__ inline void asc_copy_l12l0a(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ float8_e4m3_t* dst, __cbuf__ float8_e4m3_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ float8_e4m3_t* dst, __cbuf__ float8_e4m3_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ float8_e5m2_t* dst, __cbuf__ float8_e5m2_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ float8_e5m2_t* dst, __cbuf__ float8_e5m2_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ half* dst, __cbuf__ half* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ half* dst, __cbuf__ half* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ float* dst, __cbuf__ float* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ float* dst, __cbuf__ float* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ hifloat8_t* dst, __cbuf__ hifloat8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ hifloat8_t* dst, __cbuf__ hifloat8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ int32_t* dst, __cbuf__ int32_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ int32_t* dst, __cbuf__ int32_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ int8_t* dst, __cbuf__ int8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ int8_t* dst, __cbuf__ int8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ int16_t* dst, __cbuf__ int16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ int16_t* dst, __cbuf__ int16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a(__ca__ uint16_t* dst, __cbuf__ uint16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0a_sync(__ca__ uint16_t* dst, __cbuf__ uint16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0a_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

// ==========asc_copy_l12l0b==========
__aicore__ inline void asc_copy_l12l0b(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ float8_e4m3_t* dst, __cbuf__ float8_e4m3_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ float8_e4m3_t* dst, __cbuf__ float8_e4m3_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ float8_e5m2_t* dst, __cbuf__ float8_e5m2_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ float8_e5m2_t* dst, __cbuf__ float8_e5m2_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ half* dst, __cbuf__ half* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ half* dst, __cbuf__ half* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ float* dst, __cbuf__ float* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ float* dst, __cbuf__ float* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ hifloat8_t* dst, __cbuf__ hifloat8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ hifloat8_t* dst, __cbuf__ hifloat8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ int32_t* dst, __cbuf__ int32_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ int8_t* dst, __cbuf__ int8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ uint8_t* dst, __cbuf__ uint8_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b(__cb__ int16_t* dst, __cbuf__ int16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ int16_t* dst, __cbuf__ int16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}
__aicore__ inline void asc_copy_l12l0b(__cb__ uint16_t* dst, __cbuf__ uint16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

__aicore__ inline void asc_copy_l12l0b_sync(__cb__ uint16_t* dst, __cbuf__ uint16_t* src, uint16_t step_k, uint16_t step_m, uint16_t pos_k,
                                      uint16_t pos_m, uint8_t stride_w, uint8_t stride_h, uint8_t w_k, uint8_t h_k, uint8_t dilation_w,
                                      uint8_t dilation_h, bool filter_w, bool filter_h, bool transpose, bool fmatrix_ctrl, uint16_t size_channel)
{
    asc_copy_l12l0b_sync_impl(dst, src, step_k, step_m, pos_k, pos_m, stride_w, stride_h, w_k, h_k, dilation_w, dilation_h, filter_w, filter_h, transpose, fmatrix_ctrl, size_channel);
}

// ==========asc_copy_gm2l1_align==========
__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ half* dst, __gm__ half* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ half* dst, __gm__ half* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ float* dst, __gm__ float* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ float* dst, __gm__ float* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ uint16_t* dst, __gm__ uint16_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                      uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_align_sync(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                                                uint8_t right_padding_count, bool data_select_bit, uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride)
{
    asc_copy_gm2l1_align_sync_impl(dst, src, burst_num, burst_len, left_padding_count, right_padding_count, data_select_bit, l2_cache_ctl, burst_src_stride, burst_dst_stride);
}

// ==========asc_copy_gm2l1_dn2nz==========
__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ float8_e4m3_t* dst, __gm__ float8_e4m3_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ float8_e4m3_t* dst, __gm__ float8_e4m3_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ float8_e5m2_t* dst, __gm__ float8_e5m2_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ float8_e5m2_t* dst, __gm__ float8_e5m2_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ half* dst, __gm__ half* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ half* dst, __gm__ half* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ float* dst, __gm__ float* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ float* dst, __gm__ float* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ uint16_t* dst, __gm__ uint16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ uint16_t* dst, __gm__ uint16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_dn2nz_sync(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_dn2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

// ==========asc_copy_gm2l1_nd2nz==========
__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ bfloat16_t* dst, __gm__ bfloat16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ float8_e4m3_t* dst, __gm__ float8_e4m3_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ float8_e4m3_t* dst, __gm__ float8_e4m3_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ float8_e5m2_t* dst, __gm__ float8_e5m2_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ float8_e5m2_t* dst, __gm__ float8_e5m2_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ half* dst, __gm__ half* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ half* dst, __gm__ half* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ float* dst, __gm__ float* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ float* dst, __gm__ float* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ hifloat8_t* dst, __gm__ hifloat8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ int16_t* dst, __gm__ int16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ int32_t* dst, __gm__ int32_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ int8_t* dst, __gm__ int8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ uint16_t* dst, __gm__ uint16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ uint16_t* dst, __gm__ uint16_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ uint32_t* dst, __gm__ uint32_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                            uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

__aicore__ inline void asc_copy_gm2l1_nd2nz_sync(__cbuf__ uint8_t* dst, __gm__ uint8_t* src, uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                                uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)
{
    asc_copy_gm2l1_nd2nz_sync_impl(dst, src, loop1_src_stride, l2_cache_ctl, n_value, d_value, loop4_src_stride, smallc0_en);
}

// ==========asc_copy_gm2l1==========
__aicore__ inline void asc_copy_gm2l1(__cbuf__ void* dst, __gm__ void* src, uint32_t n_burst, uint32_t len_burst, uint8_t pad_func_mode,
                                    uint64_t src_stride, uint32_t dst_stride)
{
    asc_copy_gm2l1_impl(dst, src, n_burst, len_burst, pad_func_mode, src_stride, dst_stride);
}

__aicore__ inline void asc_copy_gm2l1_sync(__cbuf__ void* dst, __gm__ void* src, uint32_t n_burst, uint32_t len_burst, uint8_t pad_func_mode,
                                        uint64_t src_stride, uint32_t dst_stride)
{
    asc_copy_gm2l1_sync_impl(dst, src, n_burst, len_burst, pad_func_mode, src_stride, dst_stride);
}

#endif