/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INCLUDE_SIMT_API_MATH_CONSTANTS_H
#define INCLUDE_SIMT_API_MATH_CONSTANTS_H

#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_MATH_CONSTANTS_H__
#endif

#define ASCRT_INF_F                            __builtin_inff()
#define ASCRT_PI_F                             3.141592653589793f
#define ASCRT_PIO2_F                           1.5707963267948966f
#define ASCRT_PIO4_F                           0.7853981633974483f
#define ASCRT_PIO8_F                           0.39269908169872415480783042290994f
#define ASCRT_SCALAR_LN2_F                     0.69314718055994530941723212145818f
#define ASCRT_2OPI_F                           0.63661975f
#define ASCRT_MINUS_PIO2_HI_F                  (-1.5707964f)
#define ASCRT_MINUS_PIO2_LO_F                  4.371139e-8f

#define ASCRT_MAN_BIT_FLOAT_U                  0x7FFFFFU
#define ASCRT_EXP_BIT_FLOAT_U                  0x7F800000U
#define ASCRT_NEG_SIGN_BIT_U                   0x80000000U
#define ASCRT_INF_U                            0x7F800000U

#define ASCRT_MIN_VAL_S                        0x80000000 
#define ASCRT_INT32_INF_S                      2139095040

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_MATH_CONSTANTS_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_MATH_CONSTANTS_H__
#endif

#endif  // INCLUDE_SIMT_API_MATH_CONSTANTS_H