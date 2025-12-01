/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDC_MODULE_SIMT_C_WARP_LEVEL_INTERFACE_IMPL_H
#define ASCENDC_MODULE_SIMT_C_WARP_LEVEL_INTERFACE_IMPL_H

#include "simt_common_intf_impl.h"

__simt_callee__ inline int32_t asc_all(int32_t predicate)
{
    return AscendC::Simt::AllSyncImpl(predicate);
}

__simt_callee__ inline int32_t asc_any(int32_t predicate)
{
    return AscendC::Simt::AnySyncImpl(predicate);
}

__simt_callee__ inline uint32_t asc_ballot(int32_t predicate)
{
    return AscendC::Simt::BallotSyncImpl(predicate);
}

__simt_callee__ inline uint32_t asc_activemask()
{
    return AscendC::Simt::ActiveMaskImpl();
}

__simt_callee__ inline int32_t asc_shfl(int32_t var, int32_t srcLane, int32_t width)
{
    return AscendC::Simt::ShflSyncImpl(var, srcLane, width);
}

__simt_callee__ inline int32_t asc_shfl_up(int32_t var, uint32_t delta, int32_t width)
{
    return AscendC::Simt::ShflUpSyncImpl(var, delta, width);
}

__simt_callee__ inline int32_t asc_shfl_down(int32_t var, uint32_t delta, int32_t width)
{
    return AscendC::Simt::ShflDownSyncImpl(var, delta, width);
}

__simt_callee__ inline int32_t asc_shfl_xor(int32_t var, int32_t laneMask, int32_t width)
{
    return AscendC::Simt::ShflXorSyncImpl(var, laneMask, width);
}

__simt_callee__ inline int32_t asc_reduce_add(int32_t val)
{
    return AscendC::Simt::ReduceAddSyncImpl(val);
}

__simt_callee__ inline int32_t asc_reduce_max(int32_t val)
{
    return AscendC::Simt::ReduceMaxSyncImpl(val);
}

__simt_callee__ inline int32_t asc_reduce_min(int32_t val)
{
    return AscendC::Simt::ReduceMinSyncImpl(val);
}

__simt_callee__ inline void asc_syncthreads()
{
    AscendC::Simt::ThreadBarrierImpl();
}

__simt_callee__ inline void asc_threadfence()
{
    AscendC::Simt::ThreadFenceImpl();
}
#endif  // ASCENDC_MODULE_SIMT_WARP_LEVEL_INTERFACE_IMPL_H
