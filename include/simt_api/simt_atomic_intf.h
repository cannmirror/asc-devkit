/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_C_ATOMIC_INTERFACE_H
#define ASCENDC_MODULE_SIMT_C_ATOMIC_INTERFACE_H
__simt_callee__ inline int32_t asc_atomic_add(__ubuf__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_add(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline float asc_atomic_add(__ubuf__ float *address, float val);

__simt_callee__ inline int32_t asc_atomic_add(__gm__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_add(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline float asc_atomic_add(__gm__ float *address, float val);

__simt_callee__ inline int64_t asc_atomic_add(__gm__ int64_t *address, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_add(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline int32_t asc_atomic_sub(__ubuf__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_sub(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline float asc_atomic_sub(__ubuf__ float *address, float val);

__simt_callee__ inline int32_t asc_atomic_sub(__gm__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_sub(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline float asc_atomic_sub(__gm__ float *address, float val);

__simt_callee__ inline int64_t asc_atomic_sub(__gm__ int64_t *address, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_sub(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline int32_t asc_atomic_exch(__ubuf__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_exch(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline int32_t asc_atomic_exch(__gm__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_exch(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline int64_t asc_atomic_exch(__gm__ int64_t *address, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_exch(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline int32_t asc_atomic_max(__ubuf__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_max(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline float asc_atomic_max(__ubuf__ float *address, float val);

__simt_callee__ inline int32_t asc_atomic_max(__gm__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_max(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline float asc_atomic_max(__gm__ float *address, float val);

__simt_callee__ inline int64_t asc_atomic_max(__gm__ int64_t *address, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_max(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline int32_t asc_atomic_min(__ubuf__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_min(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline float asc_atomic_min(__ubuf__ float *address, float val);

__simt_callee__ inline int32_t asc_atomic_min(__gm__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_min(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline float asc_atomic_min(__gm__ float *address, float val);

__simt_callee__ inline int64_t asc_atomic_min(__gm__ int64_t *address, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_min(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline uint32_t asc_atomic_inc(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline uint32_t asc_atomic_inc(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline uint64_t asc_atomic_inc(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline uint32_t asc_atomic_dec(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline uint32_t asc_atomic_dec(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline uint64_t asc_atomic_dec(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline int32_t asc_atomic_cas(__ubuf__ int32_t *address, int32_t compare, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_cas(__ubuf__ uint32_t *address, uint32_t compare, uint32_t val);

__simt_callee__ inline int32_t asc_atomic_cas(__gm__ int32_t *address, int32_t compare, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_cas(__gm__ uint32_t *address, uint32_t compare, uint32_t val);

__simt_callee__ inline int64_t asc_atomic_cas(__gm__ int64_t *address, int64_t compare, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_cas(__gm__ uint64_t *address, uint64_t compare, uint64_t val);

__simt_callee__ inline int32_t asc_atomic_and(__ubuf__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_and(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline int32_t asc_atomic_and(__gm__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_and(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline int64_t asc_atomic_and(__gm__ int64_t *address, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_and(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline int32_t asc_atomic_or(__ubuf__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_or(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline int32_t asc_atomic_or(__gm__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_or(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline int64_t asc_atomic_or(__gm__ int64_t *address, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_or(__gm__ uint64_t *address, uint64_t val);

__simt_callee__ inline int32_t asc_atomic_xor(__ubuf__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_xor(__ubuf__ uint32_t *address, uint32_t val);

__simt_callee__ inline int32_t asc_atomic_xor(__gm__ int32_t *address, int32_t val);

__simt_callee__ inline uint32_t asc_atomic_xor(__gm__ uint32_t *address, uint32_t val);

__simt_callee__ inline int64_t asc_atomic_xor(__gm__ int64_t *address, int64_t val);

__simt_callee__ inline uint64_t asc_atomic_xor(__gm__ uint64_t *address, uint64_t val);

#include "impl/simt_api/simt_atomic_intf_impl.h"
#endif  // ASCENDC_MODULE_SIMT_ATOMIC_INTERFACE_H
