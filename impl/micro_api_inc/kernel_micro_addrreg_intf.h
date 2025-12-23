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
 * \file kernel_micro_addrreg_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_ADDRREG_INTERFACE_H
#define ASCENDC_MODULE_MICRO_ADDRREG_INTERFACE_H

#include "micro_api_inc/kernel_micro_common_intf.h"
namespace AscendC {
namespace MicroAPI {
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113))
template <typename T> __aicore__ inline AddrReg CreateAddrReg(uint16_t index0, uint16_t stride0);

template <typename T>
__aicore__ inline AddrReg CreateAddrReg(uint16_t index0, uint16_t stride0, uint16_t index1, uint16_t stride1);

template <typename T>
__aicore__ inline AddrReg CreateAddrReg(uint16_t index0, uint16_t stride0, uint16_t index1, uint16_t stride1,
                                        uint16_t index2, uint16_t stride2);

template <typename T>
__aicore__ inline AddrReg CreateAddrReg(uint16_t index0, uint16_t stride0, uint16_t index1, uint16_t stride1,
                                        uint16_t index2, uint16_t stride2, uint16_t index3, uint16_t stride3);
#else
template <typename T> __simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0);

template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0, uint16_t index1, uint32_t stride1);

template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0, uint16_t index1, uint32_t stride1,
                                             uint16_t index2, uint32_t stride2);

template <typename T>
__simd_callee__ inline AddrReg CreateAddrReg(uint16_t index0, uint32_t stride0, uint16_t index1, uint32_t stride1,
                                             uint16_t index2, uint32_t stride2, uint16_t index3, uint32_t stride3);
#endif
} // namespace MicroAPI
} // namespace AscendC

#include "impl/micro_api/kernel_micro_addrreg_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_ADDRREG_INTERFACE_H