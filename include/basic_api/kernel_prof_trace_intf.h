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
 * \file kernel_prof_trace_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_KERNEL_PROF_TRACE_INTERFACE_H
#define ASCENDC_MODULE_KERNEL_PROF_TRACE_INTERFACE_H

namespace AscendC {
#ifdef ASCENDC_TRACE_ON
enum class TraceId : uint32_t {
    KFC_CLIENT_POST_MSG = 0x7001,
    KFC_CLIENT_REV_MSG_GM = 0x7002,
    KFC_CLIENT_REV_MSG_UB = 0x7003,
    KFC_SERVER_RUN = 0x7101,
    KFC_SERVER_REV_MSG = 0x7102,
    KFC_SERVER_PROCESS_MSG = 0x7103,
    MatMul_PROCESS_MSG = 0x8001,
    MatMul_CALC,
    Conv = 0x8101,
    DropOut = 0x8201,
    SoftMax = 0x8301,
    SoftmaxGrad,
    SoftmaxFlash,
    SoftmaxFlashV2,
    LogSoftMax,
    SoftmaxFlashV3,
    LayerNorm = 0x8401,
    LayerNormGrad,
    LayerNormGradBeta,
    Pad = 0x8501,
    UnPad,
    BroadCast = 0x8601,
};

#define TRACE_START(apid)                                          \
    do {                                                           \
        set_lpcnt(::AscendC::PROF_START_EVENT | static_cast<uint32_t>(apid)); \
        ::AscendC::ProfMarkEvent();                                           \
    } while (0)

#define TRACE_STOP(apid)                                          \
    do {                                                          \
        set_lpcnt(::AscendC::PROF_STOP_EVENT | static_cast<uint32_t>(apid)); \
        ::AscendC::ProfMarkEvent();                                          \
    } while (0)

#else

#define TRACE_START(apid)
#define TRACE_STOP(apid)
#endif
__aicore__ inline void MetricsProfStart();

__aicore__ inline void MetricsProfStop();
} // namespace AscendC

#include "../../impl/basic_api/kernel_prof_trace_intf_impl.h"
#endif // ASCENDC_MODULE_KERNEL_PROF_TRACE_INTERFACE_H