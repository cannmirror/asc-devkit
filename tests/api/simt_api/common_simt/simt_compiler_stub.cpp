/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "simt_compiler_stub.h"
#include "kernel_process_lock.h"
#include "kernel_utils.h"
#include "kernel_simt_cpu.h"
#include "stub_def.h"
#include <cmath>

namespace bisheng {
namespace cce {
namespace simt {

int32_t get_block_idx()
{
    return block_idx;
}

int32_t get_block_num()
{
    return block_num;
}

}
}
}

int32_t asc_get_block_idx()
{
    return bisheng::cce::simt::get_block_idx();
}

int32_t asc_get_block_num()
{
    return bisheng::cce::simt::get_block_num();
}

half2 h2exp(half2 x)
{
    half htmp1 = x.x;
    half htmp2 = x.y;
    float tmp1 = htmp1.ToFloat();
    float tmp2 = htmp2.ToFloat();
    tmp1 = exp(tmp1);
    tmp2 = exp(tmp2);
    htmp1 = half(tmp1);
    htmp2 = half(tmp2);
    return {htmp1, htmp2};
}

half2 h2log(half2 x)
{
    half htmp1 = x.x;
    half htmp2 = x.y;
    float tmp1 = htmp1.ToFloat();
    float tmp2 = htmp2.ToFloat();
    tmp1 = log(tmp1);
    tmp2 = log(tmp2);
    htmp1 = half(tmp1);
    htmp2 = half(tmp2);
    return {htmp1, htmp2};
}

half2 h2sqrt(half2 x)
{
    half htmp1 = x.x;
    half htmp2 = x.y;
    float tmp1 = htmp1.ToFloat();
    float tmp2 = htmp2.ToFloat();
    tmp1 = sqrt(tmp1);
    tmp2 = sqrt(tmp2);
    htmp1 = half(tmp1);
    htmp2 = half(tmp2);
    return {htmp1, htmp2};
}

half2 h2rsqrt(half2 x)
{
    half tmp1 = (half)1.0 / __sqrtf(x.x);
    half tmp2 = (half)1.0 / __sqrtf(x.y);
    return {tmp1, tmp2};
}