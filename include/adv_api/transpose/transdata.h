/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LIB_TRANSPOSE_TRANSDATA_H
#define LIB_TRANSPOSE_TRANSDATA_H
#if __CCE_AICORE__ == 220
#include "transdata_common.h"
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "../../../impl/adv_api/detail/transpose/transdata/transdata_impl.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_log.h"
#include <type_traits>
#endif

namespace AscendC {

template <const TransDataConfig& config, typename T, typename U, typename S>
__aicore__ inline void TransData(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const TransDataParams<U, S>& params)
{
    Internal::TransDataImpl<config, T, U, S>(dstTensor, srcTensor, sharedTmpBuffer, params);
}

template <const TransDataConfig& config, typename T, typename U, typename S>
__aicore__ inline void TransData(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const TransDataParams<U, S>& params)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> tmp;
    const bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(tmp);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    TransData<config, T, U, S>(dstTensor, srcTensor, tmp, params);
}
} // namespace AscendC
#endif
#endif // LIB_TRANSPOSE_TRANSDATA_H
