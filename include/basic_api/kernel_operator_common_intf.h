/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_operator_common_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_mm.h"
#include "kernel_operator_cache_intf.h"
#include "kernel_operator_block_sync_intf.h"
#include "kernel_operator_sys_var_intf.h"
#include "kernel_operator_utils_intf.h"

/*
 * ingroup：SetAtomicAdd
 * brief：Set the next data from UB to the outside of AI Core whether the move write Tensor operation performs
 * atomic accumulation.
 */
namespace AscendC {
/*
 * @ingroup：SetNextTaskStart, WaitPreTaskEnd
 * @brief：In SuperKernel fusion mode, set wait flag between two operators
 */
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 1001) && (__NPU_ARCH__ != 2002)
template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX>
#else
template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_MTE3>
#endif
__aicore__ inline void SetNextTaskStart();

__aicore__ inline void WaitPreTaskEnd();

__aicore__ inline void InitSocState();

template <typename T>
__aicore__ inline __in_pipe__(V)
    __out_pipe__(MTE3) void InitOutput(GlobalTensor<T> gmWorkspaceAddr, uint32_t size, T value = 0);

enum class AtomicDtype { ATOMIC_NONE = 0, ATOMIC_F32, ATOMIC_F16, ATOMIC_S16, ATOMIC_S32, ATOMIC_S8, ATOMIC_BF16 };

enum class AtomicOp { ATOMIC_SUM = 0 };

template <AtomicDtype type, AtomicOp op>
__aicore__ inline void SetStoreAtomicConfig();

__aicore__ inline void GetStoreAtomicConfig(uint16_t& atomicType, uint16_t& atomicOp);

__aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams);

#if (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102)

template <int8_t startBit, int8_t endBit>
__aicore__ static inline void SetCtrlSpr(int64_t value);

template <int8_t startBit, int8_t endBit>
__aicore__ static inline int64_t GetCtrlSpr();

template <int8_t startBit, int8_t endBit>
__aicore__ static inline void ResetCtrlSpr();
#endif
}  // namespace AscendC

#include "../../impl/basic_api/kernel_operator_common_intf_impl.h"
#endif  // ASCENDC_MODULE_OPERATOR_COMMON_INTERFACE_H
