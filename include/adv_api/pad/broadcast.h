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
 * \file broadcast.h
 * \brief
 */
#ifndef LIB_PAD_BROADCAST_H
#define LIB_PAD_BROADCAST_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../../../impl/adv_api/detail/pad/broadcast/broadcast_common_impl.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002)

namespace AscendC {
#pragma begin_pipe(V)
/*
 * @ingroup Broadcast, now only support dim=1 or dim=2
 * @brief https://numpy.org.cn/user/basics/broadcasting.html
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] dstShape, the shape of dst tensor
 * @param [in] srcShape, the shape of src tensor
 * @param [in] sharedTmpBuffer input local temporary Tensor
 */
template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void Broadcast(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t> &sharedTmpBuffer)
{
    BroadCast<T, dim, axis, isReuseSource>(dstLocal, srcLocal, dstShape, srcShape, sharedTmpBuffer);
}

/*
 * @ingroup Broadcast, now only support dim=1 or dim=2
 * @brief https://numpy.org.cn/user/basics/broadcasting.html
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] dstShape, the shape of dst tensor
 * @param [in] srcShape, the shape of src tensor
 */
template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void Broadcast(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim])
{
    BroadCast<T, dim, axis, isReuseSource>(dstLocal, srcLocal, dstShape, srcShape);
}
#pragma end_pipe
}  // namespace AscendC

#endif

#endif  // LIB_PAD_BROADCAST_H
