/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef C_API_INSTR_VECTOR_COMPUTE_SET_DEQSCALE_ASC_2201_SET_DEQ_SCALE_IMPL_H
#define C_API_INSTR_VECTOR_COMPUTE_SET_DEQSCALE_ASC_2201_SET_DEQ_SCALE_IMPL_H

#include "c_api/c_api_interf_util.h"

namespace CApiInternal {

__aicore__ inline void asc_SetDeqScale(half scaleValue)
{
    set_deqscale(scaleValue);
}

__aicore__ inline void asc_SetDeqScale(const DeqScaleConfig config)
{
    set_deqscale(config.val);
}

__aicore__ inline void asc_SetDeqScale(__ubuf__ uint64_t* config)
{
    constexpr uint64_t deqAddr = 5;
    set_deqscale((uint64_t)config >> deqAddr);
}

} // namespace CApiInternal

__aicore__ inline void asc_SetDeqScale(half scaleValue)
{
    CApiInternal::asc_SetDeqScale(scaleValue);
}

__aicore__ inline void asc_SetDeqScale(const DeqScaleConfig config)
{
    CApiInternal::asc_SetDeqScale(config);
}

__aicore__ inline void asc_SetDeqScale(__ubuf__ uint64_t* config)
{
    CApiInternal::asc_SetDeqScale(config);
}

#endif