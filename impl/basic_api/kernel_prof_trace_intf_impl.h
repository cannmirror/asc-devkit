/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_prof_trace_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_PROF_TRACE_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_PROF_TRACE_INTERFACE_IMPL_H
#include "kernel_prof_trace.h"

namespace AscendC {
__aicore__ inline void MetricsProfStart()
{
    ProfStartImpl();
}

__aicore__ inline void MetricsProfStop()
{
    ProfStopImpl();
}

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
template<pipe_t pipe, uint16_t index>
__aicore__ inline void MarkStamp()
{
    MarkStampImpl<pipe, index>();
}
#endif
} // namespace AscendC
#endif // ASCENDC_KERNEL_PROF_TRACE_INTERFACE_IMPL_H