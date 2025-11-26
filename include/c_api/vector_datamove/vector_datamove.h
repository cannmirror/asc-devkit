/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef C_API_VECTOR_DATAMOVE_H
#define C_API_VECTOR_DATAMOVE_H

#include "impl/c_api/instr/vector_datamove/vector_datamove.h"

__aicore__ inline void asc_SetMovPadVal(uint64_t val);

__aicore__ inline void asc_copy_gm2ub(__ubuf__ void* dst, __gm__ void* src, uint32_t size);

__aicore__ inline void asc_copy_gm2ub(__ubuf__ void* dst, __gm__ void* src, const asc_copy_config& config);

__aicore__ inline void asc_copy_gm2ub_sync(__ubuf__ void* dst, __gm__ void* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm(__gm__ void* dst, __ubuf__ void* src, uint32_t size);

__aicore__ inline void asc_copy_ub2gm(__gm__ void* dst, __ubuf__ void* src, const asc_copy_config& config);

__aicore__ inline void asc_copy_ub2gm_sync(__gm__ void* dst, __ubuf__ void* src, uint32_t size);

#endif