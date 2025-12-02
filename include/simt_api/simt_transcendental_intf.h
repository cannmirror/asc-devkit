/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_C_TRANSCENDENTAL_INTERFACE_H
#define ASCENDC_MODULE_SIMT_C_TRANSCENDENTAL_INTERFACE_H

/**
 * The trigonometric tangent of an angle
 * @param x an angle, in radians.
 * @return the tangent of the x.
 */
__simt_callee__ inline float tanf(float x);

__simt_callee__ inline float tanhf(float x);

/**
 * The trigonometric tangent of an angle*pi
 * @param x an angle, in radians.
 * @return the tangent of the x*PI.
 */
__simt_callee__ inline float tanpif(float x);

__simt_callee__ inline float atanf(float x);

__simt_callee__ inline float atan2f(float y, float x);

__simt_callee__ inline float atanhf(float x);

/**
 * The trigonometric cosine of an angle
 * @param x an angle, in radians.
 * @return the cosine of the x.
 */
__simt_callee__ inline float cosf(float x);

__simt_callee__ inline float coshf(float x);

/**
 * The trigonometric cosine of an angle*pi
 * @param x an angle, in radians.
 * @return the cosine of the x*pi.
 */
__simt_callee__ inline float cospif(float x);

__simt_callee__ inline float asinf(float x);

__simt_callee__ inline float acosf(float x);

__simt_callee__ inline float acoshf(float x);

/**
 * The trigonometric sine of an angle
 * @param x an angle, in radians.
 * @return the sine of the x.
 */
__simt_callee__ inline float sinf(float x);

__simt_callee__ inline float sinhf(float x);

/**
 * The trigonometric sine of an angle
 * @param x an angle, in radians.
 * @return the sine of the x.
 */
__simt_callee__ inline float sinpif(float x);

__simt_callee__ inline float asinhf(float x);

/**
 * The trigonometric sine,cossine of an angle
 * @param x an angle, in radians.
 * @return s the sine of the x.
 *         c the cossin of the x.
 */
__simt_callee__ inline void sincosf(float x, float &s, float &c);

/**
 * The trigonometric sine,cossine of an angle*pi
 * @param x an angle, in radians.
 * @return s the sine of the x*pi.
 *         c the cossin of the x*pi.
 */
__simt_callee__ inline void sincospif(float x, float &s, float &c);

__simt_callee__ inline float exp2f(float x);

__simt_callee__ inline float exp10f(float x);

__simt_callee__ inline float expm1f(float x);


/**
 * The reciprocal of the square root of x
 * @param   x  a float value.
 * @return  the square root of x
 *          if x<0 return Nan
 */
__simt_callee__ inline float rsqrtf(float x);

/**
 * The mantissa,exponent of x
 * @param x  a float value.
 * @return   the mantissa of the x
 *         exp: the exponent of the x
 */
__simt_callee__ inline float frexpf(float x, int &exp);

/**
 * The value of x multiplied by 2 exp.
 * @param x a float value.
 *        exp a int value.
 * @return  x * 2^exp
 */
__simt_callee__ inline float ldexpf(float x, int exp);

/**
 * The square root of (x^2+y^2)
 * @param   x  a float value.
 *          y  a float value.
 * @return square root of (x^2+y^2)
 */
__simt_callee__ inline float hypotf(float x, float y);

 /**
  * The reciprocal of square root of (x^2+y^2)
  * @param   x  a float value.
  *          y  a float value.
  * @return The reciprocal of square root of (x^2+y^2)
  */
__simt_callee__ inline float rhypotf(float x, float y);

/**
 * The square root of (a^2+b^2+c^2)
 * @param a : float value.
 * @param b : float value.
 * @param c : float value.
 * @return The square root of (a^2+b^2+c^2)
 * Special cases:
 *      If any one of a,b,c is ±INF, return INF.
 *      If any one of a,b,c is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c is 0, return 0.
 *      If sqrt(a^2 + b^2 + c^2) overflows, return INF.
 */
__simt_callee__ inline float norm3df(float a, float b, float c);

/**
 * The reciprocal of square root of (a^2+b^2+c^2)
 * @param a : float value.
 * @param b : float value.
 * @param c : float value.
 * @return The reciprocal of square root of (a^2+b^2+c^2)
 * Special cases:
 *      If any one of a,b,c is ±INF, return 0.
 *      If any one of a,b,c is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c is 0, return INF.
 *      If sqrt(a^2 + b^2 + c^2) overflows, return INF.
 */
__simt_callee__ inline float rnorm3df(float a, float b, float c);

/**
 * The square root of (a^2+b^2+c^2+d^2)
 * @param a : float value.
 * @param b : float value.
 * @param c : float value.
 * @param d : float value.
 * @return The square root of (a^2+b^2+c^2+d^2)
 * Special cases:
 *      If any one of a,b,c,d is ±INF, return INF.
 *      If any one of a,b,c,d is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c,d is 0, return 0.
 *      If sqrt(a^2 + b^2 + c^2+ d^2) overflows, return INF.
 */
__simt_callee__ inline float norm4df(float a, float b, float c, float d);

/**
 * The reciprocal of square root of (a^2+b^2+c^2+d^2)
 * @param a : float value.
 * @param b : float value.
 * @param c : float value.
 * @param d : float value.
 * @return The reciprocal of square root of (a^2+b^2+c^2+d^2)
 * Special cases:
 *      If any one of a,b,c,d is ±INF, return 0.
 *      If any one of a,b,c,d is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c,d is 0, return INF.
 *      If sqrt(a^2 + b^2 + c^2 + d^2) overflows,return 0.
 */
__simt_callee__ inline float rnorm4df(float a, float b, float c, float d);

/**
 * The square root of sum of squares of the first N elements in array a
 * @param n : int value.
 * @param x : float array
 * @return The square root of (a[0]^2+...+a[n-1]^2)
 * Special cases:
 *      If any one of a[i] is ±INF, return 0.
 *      If any one of a[i] is NAN and other is not ±INF, return NAN.
 *      If all of a,b,c,d is 0, return 0.
 *      If sqrt(a[0]^2 +... + a[n-1]^2) overflows, return INF.
 *      If n is less than 1, return |a[0]|.
 */
__simt_callee__ inline float normf(int n, float* a);

/**
 * The reciprocal of square root of sum of squares of the first N elements in array a
 * @param n : int value.
 * @param x : float array
 * @return The reciprocal of square root of (a[0]^2+...+a[n-1]^2)
 * Special cases:
 *      If any one of a[i] is ±INF, return 0.
 *      If any one of a[i] is NAN and other is not ±INF, return NAN.
 *      If all of a[i] is 0, return INF.
 *      If sqrt(a[0]^2 +... + a[n-1]^2) overflows, return 0.
 *      If n is less than 1, return 1/|a[0]|.
 */
__simt_callee__ inline float rnormf(int n, float* a);

__simt_callee__ inline float log2f(float x);

__simt_callee__ inline float log10f(float x);

__simt_callee__ inline float log1pf(float x);

__simt_callee__ inline float logbf(float x);

__simt_callee__ inline int32_t ilogbf(float x);

/**
 * calculates a cube root by input x.
 * @param x a value
 * @return cbrt(x)
 * Special cases:
 *      if x is 0, return 0
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return -Inf;
 */
__simt_callee__ inline float cbrtf(float x);

/**
 * calculates reciprocal of the cube root by input x.
 * @param x a value
 * @return rcbrt(x)
 * Special cases:
 *      if x is 0, return Inf
 *      if x is Nan, return Nan;
 *      if x is Inf, return 0;
 *      if x is -Inf, return 0;
 */
__simt_callee__ inline float rcbrtf(float x);

/**
 * Calculate the error function of the input x.
 * @param x a value
 * @return erf(x)
 * Special cases:
 *      if x is 0, return 0
 *      if x is Inf, return 1;
 *      if x is -Inf, return -1;
 *      if x is Nan, return Nan;
 */
__simt_callee__ inline float erff(float x);

/**
 * Calculate the complementary error function of the input x.
 * @param x a value
 * @return erfc(x)
 * Special cases:
 *      if x is Inf, return 2;
 *      if x is -Inf, return +0;
 *      if x is Nan, return Nan;
 */
__simt_callee__ inline float erfcf(float x);

/**
 * Calculate the inverse error function of the input x.
 * @param x a value
 * @return erfinv(x)
 * Special cases:
 *      if x is 0, return 0;
 *      if x is 1, return Inf;
 *      if x is -1, return -Inf;
 *      if x outside [-1, 1], return Nan;
 *      if x is Nan, return Nan;
 */
__simt_callee__ inline float erfinvf(float x);

/**
 * Calculate the inverse complementary error function of the input x.
 * @param x a value
 * @return erfcinv(x)
 * Special cases:
 *      if x is 0, return Inf;
 *      if x is 2, return -Inf;
 *      if x outside [0, 2], return Nan;
 *      if x is Nan, return Nan;
 */
__simt_callee__ inline float erfcinvf(float x);


/**
 * Calculate the scaled complementary error function of the input x.
 * @param x a value
 * @return erfcx(x)
 * Special cases:
 *      if x is -Inf, return Inf;
 *      if x is Inf, return +0;
 *      if x is Nan, return Nan;
 */
__simt_callee__ inline float erfcxf(float x);

/**
 * Calculates gamma value by input x.
 * @param x a value
 * @return tgamma(x)
 * Special cases:
 *      if x is 0, return Inf
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return Nan;
 */
__simt_callee__ inline float tgammaf(float x);

/**
 * Calculates lgamma value by input x.
 * @param x a value
 * @return lgamma(x)
 * Special cases:
 *      if x is 0, return Inf
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return Inf;
 */
__simt_callee__ inline float lgammaf(float x);

/**
 * Calculates CylBesselI0 value by input x.
 * @param x a value
 * @return CylBesselI0(x)
 * Special cases:
 *      if x is 0, return 1;
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return Inf;
 */
__simt_callee__ inline float cyl_bessel_i0f(float x);

/**
 * Calculates CylBesselI1 value by input x.
 * @param x a value
 * @return CylBesselI1(x)
 * Special cases:
 *      if x is 0, return 0;
 *      if x is Nan, return Nan;
 *      if x is Inf, return Inf;
 *      if x is -Inf, return -Inf;
 */
__simt_callee__ inline float cyl_bessel_i1f(float x);

/**
 * Calculate the standard normal cumulative distribution function for input x.
 * @param x a value
 * @return Normcdf(x)
 * Special cases:
 *      if x is Nan, return Nan;
 *      if x is Inf, return 1;
 *      if x is -Inf, return 0;
 */
__simt_callee__ inline float normcdff(float x);

#include "impl/simt_api/simt_transcendental_intf_impl.h"
#endif  // ASCENDC_MODULE_SIMT_TRANSCENDENTAL_INTERFACE_H
