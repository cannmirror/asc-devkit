/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_C_TRANSCENDENTAL_INTERFACE_IMPL_H
#define ASCENDC_MODULE_SIMT_C_TRANSCENDENTAL_INTERFACE_IMPL_H

#include "simt_common_intf_impl.h"

__simt_callee__ inline float tanf(float x)
{
    return AscendC::Simt::TanImpl(x);
}

__simt_callee__ inline float tanhf(float x)
{
    return AscendC::Simt::TanhImpl(x);
}

__simt_callee__ inline float tanpif(float x)
{
    return AscendC::Simt::TanPiImpl(x);
}

__simt_callee__ inline float atanf(float x)
{
    return AscendC::Simt::AtanImpl(x);
}

__simt_callee__ inline float atan2f(float y, float x)
{
    return AscendC::Simt::Atan2Impl(y, x);
}

__simt_callee__ inline float atanhf(float x)
{
    return AscendC::Simt::AtanhImpl(x);
}

__simt_callee__ inline float cosf(float x)
{
    return AscendC::Simt::CosImpl(x);
}

__simt_callee__ inline float coshf(float x)
{
    return AscendC::Simt::CoshImpl(x);
}

__simt_callee__ inline float cospif(float x)
{
    return AscendC::Simt::CospiImpl(x);
}

__simt_callee__ inline float asinf(float x)
{
    return AscendC::Simt::AsinImpl(x);
}

__simt_callee__ inline float acosf(float x)
{
    return AscendC::Simt::AcosImpl(x);
}

__simt_callee__ inline float acoshf(float x)
{
    return AscendC::Simt::AcoshImpl(x);
}

__simt_callee__ inline float sinf(float x)
{
    return AscendC::Simt::SinImpl(x);
}

__simt_callee__ inline float sinhf(float x)
{
    return AscendC::Simt::SinhImpl(x);
}

__simt_callee__ inline float sinpif(float x)
{
    return AscendC::Simt::SinpiImpl(x);
}

__simt_callee__ inline float asinhf(float x)
{
    return AscendC::Simt::AsinhImpl(x);
}

__simt_callee__ inline void sincosf(float x, float &s, float &c)
{
    AscendC::Simt::SinCosImpl(x, s, c);
}

__simt_callee__ inline void sincospif(float x, float &s, float &c)
{
    AscendC::Simt::SinCospiImpl(x, s, c);
}

__simt_callee__ inline float exp2f(float x)
{
    return AscendC::Simt::Exp2Impl(x);
}

__simt_callee__ inline float exp10f(float x)
{
    return AscendC::Simt::Exp10Impl(x);
}

__simt_callee__ inline float expm1f(float x)
{
    return AscendC::Simt::Expm1Impl(x);
}

__simt_callee__ inline float rsqrtf(float x)
{
    return AscendC::Simt::RsqrtImpl(x);
}
__simt_callee__ inline float frexpf(float x, int &exp)
{
    return AscendC::Simt::FrexpImpl(x, exp);
}

__simt_callee__ inline float ldexpf(float x, int exp)
{
    return AscendC::Simt::LdexpImpl(x, exp);
}

__simt_callee__ inline float hypotf(float x, float y)
{
    return AscendC::Simt::HypotImpl(x, y);
}

__simt_callee__ inline float rhypotf(float x, float y)
{
    return AscendC::Simt::RhypotImpl(x, y);
}

__simt_callee__ inline float norm3df(float a, float b, float c)
{
    return AscendC::Simt::Norm3dImpl(a, b, c);
}

__simt_callee__ inline float rnorm3df(float a, float b, float c)
{
    return AscendC::Simt::Rnorm3dImpl(a, b, c);
}

__simt_callee__ inline float norm4df(float a, float b, float c, float d)
{
    return AscendC::Simt::Norm4dImpl(a, b, c, d);
}

__simt_callee__ inline float rnorm4df(float a, float b, float c, float d)
{
    return AscendC::Simt::Rnorm4dImpl(a, b, c, d);
}

__simt_callee__ inline float normf(int n, float* a)
{
    return AscendC::Simt::NormImpl(n, a);
}

__simt_callee__ inline float rnormf(int n, float* a)
{
    return AscendC::Simt::RnormImpl(n, a);
}

__simt_callee__ inline float log2f(float x)
{
    return AscendC::Simt::Log2Impl(x);
}

__simt_callee__ inline float log10f(float x)
{
    return AscendC::Simt::Log10Impl(x);
}

__simt_callee__ inline float log1pf(float x)
{
    return AscendC::Simt::Log1pImpl(x);
}

__simt_callee__ inline float logbf(float x)
{
    return AscendC::Simt::LogbImpl(x);
}

__simt_callee__ inline int32_t ilogbf(float x)
{
    return AscendC::Simt::ILogbImpl(x);
}

__simt_callee__ inline float cbrtf(float x)
{
    return AscendC::Simt::CbrtImpl(x);
}

__simt_callee__ inline float rcbrtf(float x)
{
    return AscendC::Simt::RcbrtImpl(x);
}

__simt_callee__ inline float erff(float x)
{
    return AscendC::Simt::ErfImpl(x);
}

__simt_callee__ inline float erfcf(float x)
{
    return AscendC::Simt::ErfcImpl(x);
}

__simt_callee__ inline float erfinvf(float x)
{
    return AscendC::Simt::ErfinvImpl(x);
}

__simt_callee__ inline float erfcinvf(float x)
{
    return AscendC::Simt::ErfcinvImpl(x);
}

__simt_callee__ inline float erfcxf(float x)
{
    return AscendC::Simt::ErfcxImpl(x);
}

__simt_callee__ inline float tgammaf(float x)
{
    return AscendC::Simt::TgammaImpl(x);
}

__simt_callee__ inline float lgammaf(float x)
{
    return AscendC::Simt::LgammaImpl(x);
}

__simt_callee__ inline float cyl_bessel_i0f(float x)
{
    return AscendC::Simt::CylBesselI0Impl(x);
}

__simt_callee__ inline float cyl_bessel_i1f(float x)
{
    return AscendC::Simt::CylBesselI1Impl(x);
}

__simt_callee__ inline float normcdff(float x)
{
    return AscendC::Simt::NormcdfImpl(x);
}

#endif  // ASCENDC_MODULE_SIMT_TRANSCENDENTAL_INTERFACE_IMPL_H
