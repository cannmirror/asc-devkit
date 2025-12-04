/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_C_BESSEL_INTERFACE_IMPL_H
#define ASCENDC_MODULE_SIMT_C_BESSEL_INTERFACE_IMPL_H

#include "simt_common_intf_impl.h"

__simt_callee__ inline float j0f(float x)
{
    return AscendC::Simt::J0Impl(x);
}

__simt_callee__ inline float j1f(float x)
{
    return AscendC::Simt::J1Impl(x);
}

__simt_callee__ inline float jnf(int n, float x)
{
    return AscendC::Simt::JnImpl(n, x);
}

__simt_callee__ inline float y0f(float x)
{
    return AscendC::Simt::Y0Impl(x);
}

__simt_callee__ inline float y1f(float x)
{
    return AscendC::Simt::Y1Impl(x);
}

__simt_callee__ inline float ynf(int n, float x)
{
    return AscendC::Simt::YnImpl(n, x);
}

#endif  // ASCENDC_MODULE_SIMT_C_BESSEL_INTERFACE_IMPL_H
