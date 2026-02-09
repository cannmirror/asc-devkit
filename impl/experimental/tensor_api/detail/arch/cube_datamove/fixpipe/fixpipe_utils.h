/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file fixpipe_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_FIXPIPE_UTILS_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_FIXPIPE_UTILS_H

namespace AscendC {
namespace Te{

template <typename T>
__aicore__ inline auto AllocTempBuf(const T& calNSize)
{
    uint64_t deqTensorTempBuf = 0;
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        deqTensorTempBuf = reinterpret_cast<uint64_t>(get_imm(0));
    }
    return deqTensorTempBuf;
}

template <typename T>
__aicore__ inline void SetFpc(const T& deqTensorTempBuf)
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        uint64_t deqTensorAddr = (deqTensorTempBuf >> static_cast<uint64_t>(7)) << 8;
        set_fpc(deqTensorAddr);
    }
}

 __aicore__ inline void InsertSync()
{
 	if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 || 
 	              CURRENT_ARCH_VERSION == ArchVersion::V2201) {
 	    pipe_barrier(PIPE_FIX);
 	}
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_NPU_ARCH_FIXPIPE_UTILS_H