/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_C_WARP_LEVEL_INTERFACE_H
#define ASCENDC_MODULE_SIMT_C_WARP_LEVEL_INTERFACE_H

#include "simt_utils.h"

__simt_callee__ inline int32_t asc_all(int32_t predicate);

__simt_callee__ inline int32_t asc_any(int32_t predicate);

__simt_callee__ inline uint32_t asc_ballot(int32_t predicate);

__simt_callee__ inline uint32_t asc_activemask();

__simt_callee__ inline int32_t asc_shfl(int32_t var, int32_t srcLane, int32_t width = warpSize);

__simt_callee__ inline uint32_t asc_shfl(uint32_t var, int32_t srcLane, int32_t width = warpSize);

__simt_callee__ inline float asc_shfl(float var, int32_t srcLane, int32_t width = warpSize);

__simt_callee__ inline int64_t asc_shfl(int64_t var, int32_t srcLane, int32_t width = warpSize);

__simt_callee__ inline uint64_t asc_shfl(uint64_t var, int32_t srcLane, int32_t width = warpSize);

__simt_callee__ inline int32_t asc_shfl_up(int32_t var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline uint32_t asc_shfl_up(uint32_t var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline float asc_shfl_up(float var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline int64_t asc_shfl_up(int64_t var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline uint64_t asc_shfl_up(uint64_t var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline int32_t asc_shfl_down(int32_t var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline uint32_t asc_shfl_down(uint32_t var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline float asc_shfl_down(float var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline int64_t asc_shfl_down(int64_t var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline uint64_t asc_shfl_down(uint64_t var, uint32_t delta, int32_t width = warpSize);

__simt_callee__ inline int32_t asc_shfl_xor(int32_t var, int32_t laneMask, int32_t width = warpSize);

__simt_callee__ inline uint32_t asc_shfl_xor(uint32_t var, int32_t laneMask, int32_t width = warpSize);

__simt_callee__ inline float asc_shfl_xor(float var, int32_t laneMask, int32_t width = warpSize);

__simt_callee__ inline int64_t asc_shfl_xor(int64_t var, int32_t laneMask, int32_t width = warpSize);

__simt_callee__ inline uint64_t asc_shfl_xor(uint64_t var, int32_t laneMask, int32_t width = warpSize);

__simt_callee__ inline int32_t asc_reduce_add(int32_t val);

__simt_callee__ inline uint32_t asc_reduce_add(uint32_t val);

__simt_callee__ inline float asc_reduce_add(float val);

__simt_callee__ inline int32_t asc_reduce_max(int32_t val);

__simt_callee__ inline uint32_t asc_reduce_max(uint32_t val);

__simt_callee__ inline float asc_reduce_max(float val);

__simt_callee__ inline int32_t asc_reduce_min(int32_t val);

__simt_callee__ inline uint32_t asc_reduce_min(uint32_t val);

__simt_callee__ inline float asc_reduce_min(float val);

__simt_callee__ inline void asc_syncthreads();

__simt_callee__ inline void asc_threadfence();

#include "impl/simt_api/simt_warp_level_intf_impl.h"
#endif  // ASCENDC_MODULE_SIMT_C_WARP_LEVEL_INTERFACE_H
