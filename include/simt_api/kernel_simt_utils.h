/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file kernel_simt_utils.h
 * \brief
 */
#ifndef ASCENDC_MODULE_SIMT_UTILS_H
#define ASCENDC_MODULE_SIMT_UTILS_H

#ifdef ASCENDC_CPU_DEBUG
constexpr int32_t warpSize = 32;
#endif

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
namespace cce {
struct Dim3 {
    uint32_t x = 1u, y = 1u, z = 1u;
    Dim3(uint32_t x_) { x = x_; }
    Dim3(uint32_t x_, uint32_t y_)
    {
        x = x_;
        y = y_;
    }
    Dim3(uint32_t x_, uint32_t y_, uint32_t z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }
};

using dim3 = Dim3;

template <auto funcPtr, typename... Args>
void async_invoke(const dim3 &dim, Args &&...args)
{
    g_threadDimX = dim.x;
    g_threadDimY = dim.y;
    g_threadDimZ = dim.z;
    AscendC::Simt::ThreadBlock &threadBlock = AscendC::Simt::ThreadBlock::GetBlockInstance();
    const uint32_t threadNum = g_threadDimX * g_threadDimY * g_threadDimZ;
    threadBlock.Init(threadNum);
    auto func = [&args...]() { funcPtr(args...); };
    for (uint32_t i = 0; i < threadNum; i++) {
        threadBlock.Schedule(func, i);
    }
    threadBlock.FinishJobs();
}
}  // namespace cce
#endif

namespace AscendC {
namespace Simt {

using Dim3 = cce::dim3;

template <auto funcPtr, typename... Args>
__aicore__ inline void VF_CALL(Dim3 threadNums, Args &&...args)
{
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 5102) || defined(SPLIT_CORE_VEC) || defined(ASCENDC_CPU_DEBUG)
    cce::async_invoke<funcPtr>(threadNums, args...);
#endif
}

}  // namespace Simt
}  // namespace AscendC
#endif  // ASCENDC_MODULE_SIMT_UTILS_H
