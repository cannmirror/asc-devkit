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

#include "instr_impl/npu_arch_3510/cube_datamove_impl/copy_matrix_cc_to_cbuf_impl/asc_fixpipe_l0c2l1_deqf16_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/copy_matrix_cc_to_cbuf_impl/asc_fixpipe_l0c2l1_f322bf16_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/copy_matrix_cc_to_cbuf_impl/asc_fixpipe_l0c2l1_f322f16_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/copy_matrix_cc_to_cbuf_impl/asc_fixpipe_l0c2l1_qf322b8_pre_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/copy_matrix_cc_to_cbuf_impl/asc_fixpipe_l0c2l1_req8_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/copy_matrix_cc_to_cbuf_impl/asc_fixpipe_l0c2l1_vdeqf16_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/copy_matrix_cc_to_cbuf_impl/asc_fixpipe_l0c2l1_vqf322b8_pre_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/copy_matrix_cc_to_cbuf_impl/asc_fixpipe_l0c2l1_vreq8_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_copy_l12l0a_mx_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_l13d_rpt_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_l13d_fmatrix_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_l0c_copy_prequant_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_gm2l1_loop_size_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_gm2l1_loop1_stride_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_gm2l1_loop2_stride_impl.h"
#include "instr_impl/npu_arch_3510/cube_datamove_impl/asc_set_gm2l1_pad_impl.h"

// ==========asc_fixpipe_l0c2l1_deqf16===========
__aicore__ inline void asc_fixpipe_l0c2l1_deqf16(__cbuf__ half* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                 uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                 uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                 bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16_sync(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                 uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                 uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                 bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                                 uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                 uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                 bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                 uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                 uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                 bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                 uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                 uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                 bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                 uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                 uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                 bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_deqf16_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_deqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

// ==========asc_fixpipe_l0c2l1_f322bf16===========
__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16(__cbuf__ half* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                   uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                   uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                   bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                  channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16_sync(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                        uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                        uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                        bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                       relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                   uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                   uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                   bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                  channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                        uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                        uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                        bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                       relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                   uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                   uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                   bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                  channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                        uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                        uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                        bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                       relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                   uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                   uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                   bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                  channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                        uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                        uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                        bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                       relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                   uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                   uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                   bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                  channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                        uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                        uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                        bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                       relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                   uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                   uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                   bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                  channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322bf16_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                        uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                        uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                        bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322bf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                       relu_pre, channel_split, nd2nz_en);
}

// ==========asc_fixpipe_l0c2l1_f322f16===========
__aicore__ inline void asc_fixpipe_l0c2l1_f322f16(__cbuf__ half* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                  uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                  uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                  bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16_sync(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                  uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                  uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                  bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                                  uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                  uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                  bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                  uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                  uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                  bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                  uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                  uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                  bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                  uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                  uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                  bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_f322f16_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_f322f16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

// ==========asc_fixpipe_l0c2l1_qf322b8_pre===========
__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre_sync(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                           uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                           uint16_t src_stride, uint8_t uint_flag_mode,
                                                           uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                          relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                           uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                           uint16_t src_stride, uint8_t uint_flag_mode,
                                                           uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                          relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                           uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                           uint16_t src_stride, uint8_t uint_flag_mode,
                                                           uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                          relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                           uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                           uint16_t src_stride, uint8_t uint_flag_mode,
                                                           uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                          relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                           uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                           uint16_t src_stride, uint8_t uint_flag_mode,
                                                           uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                          relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                      uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                      uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                      bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                     relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_qf322b8_pre_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                           uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                           uint16_t src_stride, uint8_t uint_flag_mode,
                                                           uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_qf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                          relu_pre, channel_split, nd2nz_en);
}

// ==========asc_fixpipe_l0c2l1_req8===========
__aicore__ inline void asc_fixpipe_l0c2l1_req8(__cbuf__ half* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                               uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                               uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                               bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                              channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8_sync(__cbuf__ half* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                    uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                    uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                    bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                   relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                               uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                               uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                               bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                              channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                    uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                    uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                    bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                   relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                               uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                               uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                               bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                              channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                    uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                    uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                    bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                   relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                               uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                               uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                               bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                              channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                    uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                    uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                    bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                   relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                               uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                               uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                               bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                              channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                    uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                    uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                    bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                   relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                               uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                               uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                               bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                              channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_req8_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                    uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                    uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                    bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_req8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                   relu_pre, channel_split, nd2nz_en);
}

// ==========asc_fixpipe_l0c2l1_vdeqf16===========
__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16(__cbuf__ half* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                  uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                  uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                  bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16_sync(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                  uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                  uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                  bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                                  uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                  uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                  bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                  uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                  uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                  bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                  uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                  uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                  bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                  uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                  uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                  bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                                 channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vdeqf16_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vdeqf16_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

// ==========asc_fixpipe_l0c2l1_vqf322b8_pre===========
__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre_sync(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                            uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                            uint16_t src_stride, uint8_t uint_flag_mode,
                                                            uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                           relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                            uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                            uint16_t src_stride, uint8_t uint_flag_mode,
                                                            uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                           relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                            uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                            uint16_t src_stride, uint8_t uint_flag_mode,
                                                            uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                           relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                            uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                            uint16_t src_stride, uint8_t uint_flag_mode,
                                                            uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                           relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                            uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                            uint16_t src_stride, uint8_t uint_flag_mode,
                                                            uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                           relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                       uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                       uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                       bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                      relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vqf322b8_pre_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                            uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                            uint16_t src_stride, uint8_t uint_flag_mode,
                                                            uint8_t relu_pre, bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vqf322b8_pre_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                           relu_pre, channel_split, nd2nz_en);
}

// ==========asc_fixpipe_l0c2l1_vreq8===========
__aicore__ inline void asc_fixpipe_l0c2l1_vreq8(__cbuf__ half* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                               channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8_sync(__cbuf__ half* dst, __cc__ float* src, uint8_t sid,
                                                     uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                     uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                     bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                    relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid, uint16_t n_size,
                                                uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                               channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8_sync(__cbuf__ int8_t* dst, __cc__ float* src, uint8_t sid,
                                                     uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                     uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                     bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                    relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                                uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                               channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8_sync(__cbuf__ half* dst, __cc__ int32_t* src, uint8_t sid,
                                                     uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                     uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                     bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                    relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                               channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8_sync(__cbuf__ int16_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                     uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                     uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                     bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                    relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid, uint16_t n_size,
                                                uint16_t m_size, uint32_t dst_stride_dst_d, uint16_t src_stride,
                                                uint8_t uint_flag_mode, uint8_t relu_pre, bool channel_split,
                                                bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                               channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8_sync(__cbuf__ int8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                     uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                     uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                     bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                    relu_pre, channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode, relu_pre,
                               channel_split, nd2nz_en);
}

__aicore__ inline void asc_fixpipe_l0c2l1_vreq8_sync(__cbuf__ uint8_t* dst, __cc__ int32_t* src, uint8_t sid,
                                                     uint16_t n_size, uint16_t m_size, uint32_t dst_stride_dst_d,
                                                     uint16_t src_stride, uint8_t uint_flag_mode, uint8_t relu_pre,
                                                     bool channel_split, bool nd2nz_en)
{
    asc_copy_l0c2l1_vreq8_sync_impl(dst, src, sid, n_size, m_size, dst_stride_dst_d, src_stride, uint_flag_mode,
                                    relu_pre, channel_split, nd2nz_en);
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
#endif