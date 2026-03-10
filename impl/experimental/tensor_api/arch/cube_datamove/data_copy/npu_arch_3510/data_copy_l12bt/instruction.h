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
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12BT_INSTRUCTION_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ARCH_CUBE_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12BT_INSTRUCTION_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "impl/experimental/tensor_api/tensor/pointer_impl.h"
#include "impl/experimental/tensor_api/tensor/local_tensor_impl.h"

namespace AscendC {
namespace Te {

class CopyL12BTInstr {
public:
    template <typename T, typename U, typename... Params>
    __aicore__ inline void DataCopy(const T& dst, const U& src, const Params& ...params) {
        CopyL12BT(reinterpret_cast<uint64_t>(dst.Data().Get()), src.Data().Get(), params...);
    }

private:
    template <typename T>
    __aicore__ inline void CopyL12BT(uint64_t dst, __cbuf__ T* src, bool convControl, uint16_t blockCount, uint16_t blockLen,
        uint16_t srcStride, uint16_t dstStride)
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            copy_cbuf_to_bt(dst, src, convControl, blockCount, blockLen, srcStride, dstStride);
        }
    }
};

} // namespace Te
} // namespace AscendC

#endif