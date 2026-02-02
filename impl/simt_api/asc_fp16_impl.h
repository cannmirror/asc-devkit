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
 * \file asc_fp16_impl.h
 * \brief
 */
#ifndef IMPL_SIMT_API_ASC_FP16_IMPL_H
#define IMPL_SIMT_API_ASC_FP16_IMPL_H

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_FP16_IMPL__
#warning "asc_fp16_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "asc_fp16.h" and use public functions or variables defined in interface header files."
#endif
#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)

constexpr uint32_t HALF_INF = 0x7C00;
constexpr uint32_t HALF_NEG_INF = 0xFC00;

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool __hisnan(half x)
{
    return __isnan(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool __hisinf(half x)
{
    return __isinf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __hfma(half x, half y, half z)
{
    return __fma(x, y, z);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __habs(half x)
{
    uint16_t bits = *(uint16_t*)&x;
    bits &= 0x7FFF;
    return *(half*)&bits;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool ispositiveinf(half x)
{
    uint16_t* int_x = (uint16_t*)&x;
    return *int_x == HALF_INF;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool isnegativeinf(half x)
{
    uint16_t* int_x = (uint16_t*)&x;
    return *int_x == HALF_NEG_INF;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __hmax(half x, half y)
{
    if (__hisnan(x)) {
        return y;
    } else if (__hisnan(y)) {
        return x;
    }
    if (ispositiveinf(x)) {
        return x;
    } else if (ispositiveinf(y)) {
        return y;
    }
    if (isnegativeinf(x)) {
        return y;
    } else if (isnegativeinf(y)) {
        return x;
    }
    if (x == (half)0 && y == (half)0) {
        bool sign_bitx = ((uint16_t&)x) & 0x8000;
        bool sign_bity = ((uint16_t&)y) & 0x8000;
        if (sign_bitx) {
            return y;
        } else if (sign_bity) {
            return x;
        }
    }
    if (x > y) {
        return x;
    } else {
        return y;
    }
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __hmin(half x, half y)
{
    if (__hisnan(x)) {
        return y;
    } else if (__hisnan(y)) {
        return x;
    }
    if (isnegativeinf(x)) {
        return x;
    } else if (isnegativeinf(y)) {
        return y;
    }
    if (ispositiveinf(x)) {
        return y;
    } else if (ispositiveinf(y)) {
        return x;
    }
    if (x == (half)0 && y == (half)0) {
        bool sign_bitx = ((uint16_t&)x) & 0x8000;
        bool sign_bity = ((uint16_t&)y) & 0x8000;
        if (sign_bitx) {
            return x;
        } else if (sign_bity) {
            return y;
        }
    }
    if (x < y) {
        return x;
    } else {
        return y;
    }
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hcos(half x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = cosf(tmp);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2cos(half2 x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = cosf(tmp1);
    tmp2 = cosf(tmp2);
    half htmp1 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    half htmp2 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {htmp1, htmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hsin(half x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = sinf(tmp);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2sin(half2 x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = sinf(tmp1);
    tmp2 = sinf(tmp2);
    half htmp1 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    half htmp2 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {htmp1, htmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half htanh(half x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = tanhf(tmp);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2tanh(half2 x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = tanhf(tmp1);
    tmp2 = tanhf(tmp2);
    half htmp1 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    half htmp2 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {htmp1, htmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hexp(half x)
{
    return __expf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hexp2(half x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = powf(2.0f, tmp);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2exp2(half2 x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = powf(2.0f, tmp1);
    tmp2 = powf(2.0f, tmp2);
    half htmp1 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    half htmp2 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {htmp1, htmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hexp10(half x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = powf(10.0f, tmp);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2exp10(half2 x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = powf(10.0f, tmp1);
    tmp2 = powf(10.0f, tmp2);
    half htmp1 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    half htmp2 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {htmp1, htmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hlog(half x)
{
    return __logf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hlog2(half x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = logf(x) / logf(2.0f);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2log2(half2 x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = logf(tmp1) / logf(2.0f);
    tmp2 = logf(tmp2) / logf(2.0f);
    half htmp1 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    half htmp2 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {htmp1, htmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hlog10(half x)
{
    float tmp = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
    tmp = logf(x) / logf(10.0f);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2log10(half2 x)
{
    float tmp1 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.x);
    float tmp2 = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x.y);
    tmp1 = logf(tmp1) / logf(10.0f);
    tmp2 = logf(tmp2) / logf(10.0f);
    half htmp1 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp1);
    half htmp2 = __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(tmp2);
    x = {htmp1, htmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hsqrt(half x)
{
    return __sqrtf(x);
}


__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hrsqrt(half x)
{
    return (half)1.0 / hsqrt(x);
}

#ifndef ASCENDC_CPU_DEBUG
__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2exp(half2 x)
{
    return __expf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2log(half2 x)
{
    return __logf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2sqrt(half2 x)
{
    return __sqrtf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2rsqrt(half2 x)
{
    half tmp1 = (half)1.0 / __sqrtf(x.x);
    half tmp2 = (half)1.0 / __sqrtf(x.y);
    return {tmp1, tmp2};
}
#endif

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hrcp(half x)
{
    return (half)1.0 / x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2rcp(half2 x)
{
    half tmp1 = (half)1.0 / x.x;
    half tmp2 = (half)1.0 / x.y;
    return {tmp1, tmp2};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hfloor(half x)
{
    return __floorf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2floor(half2 x)
{
    half tmp1 = __floorf(x.x);
    half tmp2 = __floorf(x.y);
    return {tmp1, tmp2};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hrint(half x)
{
    return __rintf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2rint(half2 x)
{
    half tmp1 = __rintf(x.x);
    half tmp2 = __rintf(x.y);
    return {tmp1, tmp2};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half hceil(half x)
{
    return __ceilf(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2ceil(half2 x)
{
    half tmp1 = __ceilf(x.x);
    half tmp2 = __ceilf(x.y);
    return {tmp1, tmp2};
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half htrunc(half x)
{
    if (x > (half)0) {
        return __floorf(x);
    } else {
        return __ceilf(x);
    }
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 h2trunc(half2 x)
{
    half tmp1 = x.x;
    half tmp2 = x.y;
    if (x.x > (half)0) {
        tmp1 = __floorf(tmp1);
    } else {
        tmp1 = __ceilf(tmp1);
    }
    if (x.y > (half)0) {
        tmp2 = __floorf(tmp2);
    } else {
        tmp2 = __ceilf(tmp2);
    }
    x = {tmp1, tmp2};
    return x;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __float2half(const float x) {
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __float2half_rn(const float x) {
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __float2half_rz(const float x) {
    return __cvt_half<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __float2half_rd(const float x) {
    return __cvt_half<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __float2half_ru(const float x) {
    return __cvt_half<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __float2half_rna(const float x) {
    return __cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __float2half_ro(const float x) {
    return __cvt_half<ROUND::O, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline float __half2float(const half x) {
    union Data {
        half h;
        unsigned int i;
    };
    union Data d = {.h = x};

    unsigned int sign = ((d.i >> 15U) & 1U);
    unsigned int exponent = ((d.i >> 10U) & 0x1fU);
    if (exponent == 0) {
        return x;
    }
    unsigned int mantissa = ((d.i & 0x3ffU) << 13U);

    if (exponent == 0x1fU) {
        sign = ((mantissa != 0U) ? 0U : sign);
        mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
        exponent = 0xffU;
    } else if (exponent == 0U) {
        if (mantissa != 0U) {
            unsigned int msb;
            exponent = 0x71U;
            do {
                msb = (mantissa & 0x400000U);
                mantissa <<= 1U;
                --exponent;
            } while (msb != 0U);
            mantissa &= 0x7fffffU;
        }
    } else {
        exponent += 0x70U;
    }

    unsigned int u = ((sign << 31) | (exponent << 23U) | mantissa);
    union Data1 {
        float f;
        unsigned int i;
    };
    union Data1 d1{.i = u};
    return d1.f;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __half2uint_rn(const half x) {
    return __cvt_uint32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __half2uint_rz(const half x) {
    return __cvt_uint32_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __half2uint_rd(const half x) {
    return __cvt_uint32_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __half2uint_ru(const half x) {
    return __cvt_uint32_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned int __half2uint_rna(const half x) {
    return __cvt_uint32_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __half2int_rn(const half x) {
    return __cvt_int32_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __half2int_rz(const half x) {
    return __cvt_int32_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __half2int_rd(const half x) {
    return __cvt_int32_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __half2int_ru(const half x) {
    return __cvt_int32_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline int __half2int_rna(const half x) {
    return __cvt_int32_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __half2ull_rn(const half x) {
    float x_fp32 = x;
    return __cvt_uint64_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __half2ull_rz(const half x) {
    float x_fp32 = x;
    return __cvt_uint64_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __half2ull_rd(const half x) {
    float x_fp32 = x;
    return __cvt_uint64_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __half2ull_ru(const half x) {
    float x_fp32 = x;
    return __cvt_uint64_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned long long int __half2ull_rna(const half x) {
    float x_fp32 = x;
    return __cvt_uint64_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __half2ll_rn(const half x) {
    float x_fp32 = x;
    return __cvt_int64_t<ROUND::R, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __half2ll_rz(const half x) {
    float x_fp32 = x;
    return __cvt_int64_t<ROUND::Z, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __half2ll_rd(const half x) {
    float x_fp32 = x;
    return __cvt_int64_t<ROUND::F, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __half2ll_ru(const half x) {
    float x_fp32 = x;
    return __cvt_int64_t<ROUND::C, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline long long int __half2ll_rna(const half x) {
    float x_fp32 = x;
    return __cvt_int64_t<ROUND::A, RoundingSaturation::RS_ENABLE_VALUE>(x_fp32);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __uint2half_rn(const unsigned int x) {
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __uint2half_rz(const unsigned int x) {
    return __cvt_half<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __uint2half_rd(const unsigned int x) {
    return __cvt_half<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __uint2half_ru(const unsigned int x) {
    return __cvt_half<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __uint2half_rna(const unsigned int x) {
    return __cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __int2half_rn(const int x) {
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __int2half_rz(const int x) {
    return __cvt_half<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __int2half_rd(const int x) {
    return __cvt_half<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __int2half_ru(const int x) {
    return __cvt_half<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __int2half_rna(const int x) {
    return __cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(x);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ull2half_rn(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ull2half_rz(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ull2half_rd(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ull2half_ru(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ull2half_rna(const unsigned long long int x) {
    uint64_t y = x;
    float f = __cvt_float<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ll2half_rn(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::R, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ll2half_rz(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::Z, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ll2half_rd(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::F, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ll2half_ru(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::C, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half __ll2half_rna(const long long int x) {
    int64_t y = x;
    float f = __cvt_float<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(y);
    return __cvt_half<ROUND::A, RoundingSaturation::RS_DISABLE_VALUE>(f);
}

#ifndef __NPU_COMPILER_INTERNAL_PURE_SIMT__
#ifndef ASCENDC_CPU_DEBUG
__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_add(__ubuf__ half *address, half val)
{
    atomicAdd(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_add(__ubuf__ half2 *address, half2 val)
{
    return atomicAdd(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_sub(__ubuf__ half2 *address, half2 val)
{
    return atomicSub(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_exch(__ubuf__ half2 *address, half2 val)
{
    return atomicExch(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_max(__ubuf__ half *address, half val)
{
    atomicMax(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_max(__ubuf__ half2 *address, half2 val)
{
    return atomicMax(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_min(__ubuf__ half *address, half val)
{
    atomicMin(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_min(__ubuf__ half2 *address, half2 val)
{
    return atomicMin(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_cas(__ubuf__ half2 *address, half2 compare, half2 val)
{
    return atomicCAS(address, compare, val);
}
#endif

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_add(__gm__ half *address, half val)
{
    atomicAdd(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_add(__gm__ half2 *address, half2 val)
{
    return atomicAdd(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_sub(__gm__ half2 *address, half2 val)
{
    return atomicSub(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_exch(__gm__ half2 *address, half2 val)
{
    return atomicExch(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_max(__gm__ half *address, half val)
{
    atomicMax(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_max(__gm__ half2 *address, half2 val)
{
    return atomicMax(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_min(__gm__ half *address, half val)
{
    atomicMin(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_min(__gm__ half2 *address, half2 val)
{
    return atomicMin(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_cas(__gm__ half2 *address, half2 compare, half2 val)
{
    return atomicCAS(address, compare, val);
}

#ifndef ASCENDC_CPU_DEBUG
__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_ldcg(__gm__ half* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_ldcg(__gm__ half2* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((__gm__ int32_t*)address);
    return (half2&)t;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_ldca(__gm__ half* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_ldca(__gm__ half2* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((__gm__ int32_t*)address);
    return (half2&)t;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stcg(__gm__ half* address, half val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stcg(__gm__ half2* address, half2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((__gm__ int32_t*)address, (int32_t&)val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stwt(__gm__ half* address, half val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stwt(__gm__ half2* address, half2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((__gm__ int32_t*)address, (int32_t&)val);
}
#endif

#else
__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_add(half *address, half val)
{
    atomicAdd(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_add(half2 *address, half2 val)
{
    return atomicAdd(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_sub(half2 *address, half2 val)
{
    return atomicSub(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_exch(half2 *address, half2 val)
{
    return atomicExch(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_max(half *address, half val)
{
    atomicMax(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_max(half2 *address, half2 val)
{
    return atomicMax(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_atomic_min(half *address, half val)
{
    atomicMin(address, val);
    return *address;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_min(half2 *address, half2 val)
{
    return atomicMin(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_atomic_cas(half2 *address, half2 compare, half2 val)
{
    return atomicCAS(address, compare, val);
}

#ifndef ASCENDC_CPU_DEBUG
__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_ldcg(half* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_ldcg(half2* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address);
    return (half2&)t;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_ldca(half* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_ldca(half2* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address);
    return (half2&)t;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stcg(half* address, half val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stcg(half2* address, half2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address, (int32_t&)val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stwt(half* address, half val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void asc_stwt(half2* address, half2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address, (int32_t&)val);
}
#endif
#endif

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_shfl(half var, int32_t src_lane, int32_t width)
{
    return __shfl(var,
        ((warpSize - width) << LANE_MASK_START_POS) | (MAX_OFFSET_OF_MODE << MAX_OFFSET_START_POS) | (src_lane));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_shfl(half2 var, int32_t src_lane, int32_t width)
{
    return __shfl(var,
        ((warpSize - width) << LANE_MASK_START_POS) | (MAX_OFFSET_OF_MODE << MAX_OFFSET_START_POS) | (src_lane));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_shfl_up(half var, uint32_t delta, int32_t width)
{
    return __shfl_up(var,
        ((warpSize - width) << LANE_MASK_START_POS) | (MAX_OFFSET_OF_UP_MODE << MAX_OFFSET_START_POS) | (delta));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_shfl_up(half2 var, uint32_t delta, int32_t width)
{
    return __shfl_up(var,
        ((warpSize - width) << LANE_MASK_START_POS) | (MAX_OFFSET_OF_UP_MODE << MAX_OFFSET_START_POS) | (delta));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_shfl_down(half var, uint32_t delta, int32_t width)
{
    return __shfl_down(var,
                       ((warpSize - width) << LANE_MASK_START_POS) | (MAX_OFFSET_OF_MODE << MAX_OFFSET_START_POS) | (delta));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_shfl_down(half2 var, uint32_t delta, int32_t width)
{
    return __shfl_down(var,
                       ((warpSize - width) << LANE_MASK_START_POS) | (MAX_OFFSET_OF_MODE << MAX_OFFSET_START_POS) | (delta));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_shfl_xor(half var, int32_t lane_mask, int32_t width)
{
    return __shfl_xor(var,
                      ((warpSize - width) << LANE_MASK_START_POS) | (MAX_OFFSET_OF_MODE << MAX_OFFSET_START_POS) | (lane_mask));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 asc_shfl_xor(half2 var, int32_t lane_mask, int32_t width)
{
    return __shfl_xor(var,
                      ((warpSize - width) << LANE_MASK_START_POS) | (MAX_OFFSET_OF_MODE << MAX_OFFSET_START_POS) | (lane_mask));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_reduce_add(half val)
{
    return __reduce_add(val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_reduce_max(half val)
{
    return __reduce_max(val);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline half asc_reduce_min(half val)
{
    return __reduce_min(val);
}

 __SIMT_DEVICE_FUNCTIONS_DECL__ inline half2 make_half2(half x, half y)
{
    half2 tmp;
    tmp.x = x;
    tmp.y = y;
    return tmp;
}
#endif

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_FP16_IMPL__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_ASC_FP16_IMPL__
#endif

#endif  // IMPL_SIMT_API_ASC_FP16_IMPL_H