/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file asc_bf16_impl.h
 * \brief
 */
#ifndef IMPL_SIMT_API_ASC_BF16_IMPL_H
#define IMPL_SIMT_API_ASC_BF16_IMPL_H

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_BF16_IMPL__
#warning "asc_bf16_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "asc_bf16.h" and use public functions or variables defined in interface header files."
#endif

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool __hisnan(bfloat16_t x)
{
    return __isnan(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool __hisinf(bfloat16_t x)
{
    return __isinf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __habs(bfloat16_t x)
{
    uint16_t bits = *(uint16_t*)&x;
    bits &= 0x7FFF;
    return *(bfloat16_t*)&bits;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __hfma(bfloat16_t x, bfloat16_t y, bfloat16_t z)
{
    return __fma(x, y, z);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __hmax(bfloat16_t x, bfloat16_t y)
{
    if (__hisnan(x)) {
        return y;
    } else if (__hisnan(y)) {
        return x;
    }
    return __max(x, y);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __hmin(bfloat16_t x, bfloat16_t y)
{
    if (__hisnan(x)) {
        return y;
    } else if (__hisnan(y)) {
        return x;
    }
    return __min(x, y);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hcos(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = cosf(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2cos(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = cosf(tmp1);
    tmp2 = cosf(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hsin(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = sinf(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2sin(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = sinf(tmp1);
    tmp2 = sinf(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t htanh(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = tanhf(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2tanh(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = tanhf(tmp1);
    tmp2 = tanhf(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hexp(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = expf(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2exp(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = expf(tmp1);
    tmp2 = expf(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hexp2(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = exp2f(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2exp2(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = exp2f(tmp1);
    tmp2 = exp2f(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hexp10(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = exp10f(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2exp10(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = exp10f(tmp1);
    tmp2 = exp10f(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hlog(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = logf(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2log(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = logf(tmp1);
    tmp2 = logf(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hlog2(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = log2f(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2log2(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = log2f(tmp1);
    tmp2 = log2f(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hlog10(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = log10f(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2log10(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = log10f(tmp1);
    tmp2 = log10f(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hsqrt(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = sqrtf(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2sqrt(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = sqrtf(tmp1);
    tmp2 = sqrtf(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hrsqrt(bfloat16_t x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = 1.0f / sqrtf(tmp);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2rsqrt(bfloat16x2_t x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = 1.0f / sqrtf(tmp1);
    tmp2 = 1.0f / sqrtf(tmp2);
    bfloat16_t bftmp1 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    bfloat16_t bftmp2 = __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {bftmp1, bftmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hrcp(bfloat16_t x)
{
    return (bfloat16_t)1.0 / x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2rcp(bfloat16x2_t x)
{
    bfloat16_t tmp1 = (bfloat16_t)1.0 / x.x;
    bfloat16_t tmp2 = (bfloat16_t)1.0 / x.y;
    return {tmp1, tmp2};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hfloor(bfloat16_t x)
{
    return __floorf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2floor(bfloat16x2_t x)
{
    bfloat16_t tmp1 = __floorf(x.x);
    bfloat16_t tmp2 = __floorf(x.y);
    return {tmp1, tmp2};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hrint(bfloat16_t x)
{
    return __rintf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2rint(bfloat16x2_t x)
{
 	bfloat16_t tmp1 = __rintf(x.x);
 	bfloat16_t tmp2 = __rintf(x.y);
 	return {tmp1, tmp2};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t hceil(bfloat16_t x)
{
    return __ceilf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2ceil(bfloat16x2_t x)
{
 	bfloat16_t tmp1 = __ceilf(x.x);
 	bfloat16_t tmp2 = __ceilf(x.y);
 	return {tmp1, tmp2};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t htrunc(bfloat16_t x)
{
    if (x > (bfloat16_t)0) {
        return __floorf(x);
    } else {
        return __ceilf(x);
    }
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t h2trunc(bfloat16x2_t x)
{
    bfloat16_t tmp1 = x.x;
    bfloat16_t tmp2 = x.y;
    if (x.x > (bfloat16_t)0) {
        tmp1 = __floorf(tmp1);
    } else {
        tmp1 = __ceilf(tmp1);
    }
    if (x.y > (bfloat16_t)0) {
        tmp2 = __floorf(tmp2);
    } else {
        tmp2 = __ceilf(tmp2);
    }
    x = {tmp1, tmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __float2bfloat16(const float x) {
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __float2bfloat16_rn(const float x) {
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __float2bfloat16_rz(const float x) {
    return __cvt_bfloat16_t<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __float2bfloat16_rd(const float x) {
    return __cvt_bfloat16_t<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __float2bfloat16_ru(const float x) {
    return __cvt_bfloat16_t<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __float2bfloat16_rna(const float x) {
    return __cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __half2bfloat16_rn(const half x) {
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __half2bfloat16_rz(const half x) {
    return __cvt_bfloat16_t<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __half2bfloat16_rd(const half x) {
    return __cvt_bfloat16_t<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __half2bfloat16_ru(const half x) {
    return __cvt_bfloat16_t<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __half2bfloat16_rna(const half x) {
    return __cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __bfloat162half_rn(const bfloat16_t x) {
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __bfloat162half_rz(const bfloat16_t x) {
    return __cvt_half<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __bfloat162half_rd(const bfloat16_t x) {
    return __cvt_half<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __bfloat162half_ru(const bfloat16_t x) {
    return __cvt_half<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __bfloat162half_rna(const bfloat16_t x) {
    return __cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline float __bfloat162float(const bfloat16_t x) {
    union Data {
        bfloat16_t bf;
        unsigned int i;
    };
    union Data d = {.bf = x};
    unsigned int u = d.i << 16;
    union Data2 {
        float f;
        unsigned int i;
    };
    union Data2 d2 = {.i = u};
    return d2.f;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __bfloat162uint_rn(const bfloat16_t x) {
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    return __cvt_uint32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __bfloat162uint_rz(const bfloat16_t x) {
    float f = __cvt_float<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
    return __cvt_uint32_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __bfloat162uint_rd(const bfloat16_t x) {
    float f = __cvt_float<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
    return __cvt_uint32_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __bfloat162uint_ru(const bfloat16_t x) {
    float f = __cvt_float<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
    return __cvt_uint32_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __bfloat162uint_rna(const bfloat16_t x) {
    float f = __cvt_float<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
    return __cvt_uint32_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __bfloat162int_rn(const bfloat16_t x) {
    return __cvt_int32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __bfloat162int_rz(const bfloat16_t x) {
    return __cvt_int32_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __bfloat162int_rd(const bfloat16_t x) {
    return __cvt_int32_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __bfloat162int_ru(const bfloat16_t x) {
    return __cvt_int32_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __bfloat162int_rna(const bfloat16_t x) {
    return __cvt_int32_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __bfloat162ull_rn(const bfloat16_t x) {
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_uint64_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __bfloat162ull_rz(const bfloat16_t x) {
    float f = __cvt_float<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_uint64_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __bfloat162ull_rd(const bfloat16_t x) {
    float f = __cvt_float<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_uint64_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __bfloat162ull_ru(const bfloat16_t x) {
    float f = __cvt_float<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_uint64_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __bfloat162ull_rna(const bfloat16_t x) {
    float f = __cvt_float<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_uint64_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __bfloat162ll_rn(const bfloat16_t x) {
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_int64_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __bfloat162ll_rz(const bfloat16_t x) {
    float f = __cvt_float<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_int64_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __bfloat162ll_rd(const bfloat16_t x) {
    float f = __cvt_float<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_int64_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __bfloat162ll_ru(const bfloat16_t x) {
    float f = __cvt_float<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_int64_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __bfloat162ll_rna(const bfloat16_t x) {
    float f = __cvt_float<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(x);
    return __cvt_int64_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __uint2bfloat16_rn(const unsigned int x) {
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __uint2bfloat16_rz(const unsigned int x) {
    return __cvt_bfloat16_t<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __uint2bfloat16_rd(const unsigned int x) {
    return __cvt_bfloat16_t<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __uint2bfloat16_ru(const unsigned int x) {
    return __cvt_bfloat16_t<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __uint2bfloat16_rna(const unsigned int x) {
    return __cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __int2bfloat16_rn(const int x) {
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __int2bfloat16_rz(const int x) {
    return __cvt_bfloat16_t<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __int2bfloat16_rd(const int x) {
    return __cvt_bfloat16_t<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __int2bfloat16_ru(const int x) {
    return __cvt_bfloat16_t<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __int2bfloat16_rna(const int x) {
    return __cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ull2bfloat16_rn(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ull2bfloat16_rz(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ull2bfloat16_rd(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ull2bfloat16_ru(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ull2bfloat16_rna(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ll2bfloat16_rn(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ll2bfloat16_rz(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ll2bfloat16_rd(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ll2bfloat16_ru(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t __ll2bfloat16_rna(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_bfloat16_t<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

#ifndef __NPU_COMPILER_INTERNAL_PURE_SIMT__
#ifndef ASCENDC_CPU_DEBUG
__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_add(__ubuf__ bfloat16_t *address, bfloat16_t val)
{
    atomicAdd(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_add(__ubuf__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicAdd(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_sub(__ubuf__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicSub(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_exch(__ubuf__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicExch(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_max(__ubuf__ bfloat16_t *address, bfloat16_t val)
{
    atomicMax(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_max(__ubuf__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicMax(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_min(__ubuf__ bfloat16_t *address, bfloat16_t val)
{
    atomicMin(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_min(__ubuf__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicMin(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_cas(__ubuf__ bfloat16x2_t *address, bfloat16x2_t compare, bfloat16x2_t val)
{
    return atomicCAS(address, compare, val);
}
#endif

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_add(__gm__ bfloat16_t *address, bfloat16_t val)
{
    atomicAdd(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_add(__gm__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicAdd(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_sub(__gm__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicSub(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_exch(__gm__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicExch(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_max(__gm__ bfloat16_t *address, bfloat16_t val)
{
    atomicMax(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_max(__gm__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicMax(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_min(__gm__ bfloat16_t *address, bfloat16_t val)
{
    atomicMin(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_min(__gm__ bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicMin(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_cas(__gm__ bfloat16x2_t *address, bfloat16x2_t compare, bfloat16x2_t val)
{
    return atomicCAS(address, compare, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_ldcg(__gm__ bfloat16_t* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_ldcg(__gm__ bfloat16x2_t* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((__gm__ int32_t*)address);
    return (bfloat16x2_t&)t;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_ldca(__gm__ bfloat16_t* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_ldca(__gm__ bfloat16x2_t* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((__gm__ int32_t*)address);
    return (bfloat16x2_t&)t;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stcg(__gm__ bfloat16_t* address, bfloat16_t val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stcg(__gm__ bfloat16x2_t* address, bfloat16x2_t val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((__gm__ int32_t*)address, (int32_t&)val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stwt(__gm__ bfloat16_t* address, bfloat16_t val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stwt(__gm__ bfloat16x2_t* address, bfloat16x2_t val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((__gm__ int32_t*)address, (int32_t&)val);
}

#else
__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_add(bfloat16_t *address, bfloat16_t val)
{
    atomicAdd(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_add(bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicAdd(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_sub(bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicSub(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_exch(bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicExch(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_max(bfloat16_t *address, bfloat16_t val)
{
    atomicMax(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_max(bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicMax(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_atomic_min(bfloat16_t *address, bfloat16_t val)
{
    atomicMin(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_min(bfloat16x2_t *address, bfloat16x2_t val)
{
    return atomicMin(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_atomic_cas(bfloat16x2_t *address, bfloat16x2_t compare, bfloat16x2_t val)
{
    return atomicCAS(address, compare, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_ldcg(bfloat16_t* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_ldcg(bfloat16x2_t* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address);
    return (bfloat16x2_t&)t;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16_t asc_ldca(bfloat16_t* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t asc_ldca(bfloat16x2_t* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address);
    return (bfloat16x2_t&)t;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stcg(bfloat16_t* address, bfloat16_t val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stcg(bfloat16x2_t* address, bfloat16x2_t val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address, (int32_t&)val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stwt(bfloat16_t* address, bfloat16_t val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stwt(bfloat16x2_t* address, bfloat16x2_t val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address, (int32_t&)val);
}
#endif

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bfloat16x2_t make_bfloat162(bfloat16_t x, bfloat16_t y)
{
    bfloat16x2_t tmp;
    tmp.x = x;
    tmp.y = y;
    return tmp;
}

#endif

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_BF16_IMPL__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_BF16_IMPL__
#endif

#endif  // IMPL_SIMT_API_ASC_BF16_IMPL_H