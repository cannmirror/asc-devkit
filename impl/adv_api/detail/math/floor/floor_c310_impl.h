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
 * \file floor_C310_impl.h
 * \brief
 */
#ifndef DETAIL_MATH_FLOOR_FLOOR_C310_IMPL_H
#define DETAIL_MATH_FLOOR_FLOOR_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"
 
namespace AscendC {

template<typename T, bool isReuseSource = false>
__aicore__ inline void FloorImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, 
    const uint32_t calCount)
{
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Floor");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Floor");
    Truncate<T, RoundMode::CAST_FLOOR>(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void FloorImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, 
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    FloorImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}

}   // namespace AscendC
#endif  //DETAIL_MATH_FLOOR_FLOOR_C310_IMPL_H
