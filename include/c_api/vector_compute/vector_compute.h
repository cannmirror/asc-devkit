/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef C_API_VECTOR_COMPUTE_H
#define C_API_VECTOR_COMPUTE_H

#include "impl/c_api/instr/vector_compute/vector_compute.h"
#include "../c_api_interf_util.h"

__aicore__ inline void asc_SetCmpMask(__ubuf__ void *src);

__aicore__ inline void asc_SetDeqScale(half scaleValue);

__aicore__ inline void asc_SetDeqScale(DeqScaleConfig config);

__aicore__ inline void asc_SetDeqscale(__ubuf__ uint64_t* config);

__aicore__ inline void asc_SetMaskCount();

__aicore__ inline void asc_SetMaskNorm();

__aicore__ inline void asc_SetVectorMask(uint64_t mask1, uint64_t mask0);

__aicore__ inline int64_t asc_GetAccVal();

__aicore__ inline void asc_GetCmpMask(__ubuf__ void* dst);

__aicore__ inline void asc_GetReduceMaxMinCnt(half& val, int32_t& index);

__aicore__ inline void asc_GetReduceMaxMinCnt(float& val, int32_t& index);

__aicore__ inline int64_t asc_GetRsvdCount();

__aicore__ inline void asc_GetVms4Sr(uint16_t sortedNum[4]);

#endif