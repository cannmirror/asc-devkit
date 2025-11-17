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
 * \file ascc_ast_kernel_stub.cpp
 * \brief
 */

#include "ascc_ast_kernel_stub.h"

namespace Ascc {
static const char* CLANG_TOOLING_KERNEL_STUB = R"(
#ifndef __INCLUDE_KERNEL_OPERATOR_STUB_H__
#define __INCLUDE_KERNEL_OPERATOR_STUB_H__
using uint32_t = unsigned int;
using int32_t = signed int;
using uint8_t = unsigned char;
struct bfloat16_t { unsigned short data; };
struct half { unsigned short data; };
static constexpr int32_t g_coreType = 0;
#define __global__ __attribute__((global))
#define __aicore__ __attribute__((annotate("device")))
typedef enum {
    PIPE_S = 0,
    PIPE_V,
    PIPE_M,
    PIPE_MTE1,
    PIPE_MTE2,
    PIPE_MTE3,
    PIPE_ALL,
    PIPE_MTE4 = 7,
    PIPE_MTE5 = 8,
    PIPE_V2 = 9,
    PIPE_FIX = 10
} pipe_t;
typedef enum {
    EVENT_ID0 = 0,
    EVENT_ID1,
    EVENT_ID2,
    EVENT_ID3,
    EVENT_ID4,
    EVENT_ID5,
    EVENT_ID6,
    EVENT_ID7
} event_t;
typedef enum {
    inc = 0,
    dec = 1
} addr_cal_mode_t;
typedef enum {
    VALUE_INDEX = 0,
    INDEX_VALUE = 1,
    ONLY_VALUE = 2,
    ONLY_INDEX = 3
} Order_t;
typedef enum {
    NoQuant = 0,
    F322F16 = 1,
    VQF322HIF8_PRE = 2,
    QF322HIF8_PRE = 3,
    VQF322HIF8_PRE_HYBRID = 4,
    QF322HIF8_PRE_HYBRID = 5,
    AttachF16Mul = 6,
    VREQ8 = 8,
    REQ8 = 9,
    VDEQF16 = 10,
    DEQF16 = 11,
    VSHIFTS322S16 = 12,
    SHIFTS322S16 = 13,
    VQF322FP8_PRE = 12,
    QF322FP8_PRE = 13,
    VQF322F32_PRE = 14,
    QF322F32_PRE = 15,
    F322BF16 = 16,
    VQF162B8_PRE = 17,
    QF162B8_PRE = 18,
    VQF162S4_PRE = 19,
    QF162S4_PRE = 20,
    VREQ4 = 21,
    REQ4 = 22,
    VQF322B8_PRE = 23,
    QF322B8_PRE = 24,
    VQF322S4_PRE = 25,
    QF322S4_PRE = 26,
    VDEQS16 = 27,
    DEQS16 = 28,
    VQF162S16_PRE = 29,
    QF162S16_PRE = 30,
    VQF322F16_PRE = 31,
    QF322F16_PRE = 32,
    VQF322BF16_PRE = 33,
    QF322BF16_PRE = 34,
    VQS322BF16_PRE = 35,
    QS322BF16_PRE = 36
} QuantMode_t;
typedef enum {
    VA0 = 0,
    VA1,
    VA2,
    VA3,
    VA4,
    VA5,
    VA6,
    VA7
} ub_addr8_t;
#define SIGABRT 6
int raise(int sig);
#define __WORKGROUP_LOCAL__
#define __BLOCK_LOCAL__
#define __VEC_SCOPE__
#define __gm__
#define __cbuf__
#define __ubuf__
#define __cc__
#define __ca__
#define __cb__
#define __fbuf__
#define __sync_noalias__
#define __sync_alias__
#define __check_sync_alias__
#define __sync_out__
#define __sync_in__
#define __inout_pipe__(...)
#define __in_pipe__(...)
#define __out_pipe__(...)
#define __ssbuf__
#define __simt_callee__
#define __simt_vf__
#define GM_ADDR uint8_t*
int32_t __asccPCC__(uint32_t block, void *, void *);
int32_t __asccCC__(uint32_t block, void *, void *);
#endif // __INCLUDE_KERNEL_OPERATOR_STUB_H__
)";

const char* GetAstKernelStub()
{
    return CLANG_TOOLING_KERNEL_STUB;
}
}
