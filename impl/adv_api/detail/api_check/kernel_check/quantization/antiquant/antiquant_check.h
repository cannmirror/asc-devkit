/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file antiquant_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_QUANTIZATION_ANTIQUANT_ANTIQUANT_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_QUANTIZATION_ANTIQUANT_ANTIQUANT_CHECK_H_

#include "include/adv_api/quantization/ascend_antiquant_utils.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "antiquant_check_common.h"
#else
#include "antiquant_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename SrcType, typename DstType, bool isTranspose>
__aicore__ inline void CheckFuncAscendAntiQuant(__gm__ const char *apiName, const LocalTensor<DstType> &dst,
    const LocalTensor<SrcType> &src, const LocalTensor<DstType> &scale, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    CheckFuncClassAscendAntiQuantChannel<SrcType, DstType, isTranspose> checkFun(apiName);
    checkFun.VerifyingParameters(dst, src, scale, sharedTmpBuffer, K, shapeInfo);
}

template <typename SrcType, typename DstType, bool isTranspose>
__aicore__ inline void CheckFuncAscendAntiQuant(__gm__ const char *apiName, const LocalTensor<DstType> &dst,
    const LocalTensor<SrcType> &src, const DstType scale, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K,
    const AntiQuantShapeInfo& shapeInfo = {})
{
    CheckFuncClassAscendAntiQuantTensor<SrcType, DstType, isTranspose> checkFun(apiName);
    checkFun.VerifyingParameters(dst, src, scale, sharedTmpBuffer, K, shapeInfo);
}

template <typename SrcType, typename DstType, bool isTranspose>
__aicore__ inline void CheckFuncAscendAntiQuant(__gm__ const char *apiName, const LocalTensor<DstType> &dst,
    const LocalTensor<SrcType> &src, const LocalTensor<DstType> &offset, const LocalTensor<DstType> &scale,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{ 
    CheckFuncClassAscendAntiQuantChannelOffset<SrcType, DstType, isTranspose> checkFun(apiName);
    checkFun.VerifyingParameters(dst, src, offset, scale, sharedTmpBuffer, K, shapeInfo);
}

template <typename SrcType, typename DstType, bool isTranspose>
__aicore__ inline void CheckFuncAscendAntiQuant(__gm__ const char *apiName, const LocalTensor<DstType> &dst,
    const LocalTensor<SrcType> &src, const DstType offset, const DstType scale, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t K, const AntiQuantShapeInfo& shapeInfo = {})
{
    CheckFuncClassAscendAntiQuantTensorOffset<SrcType, DstType, isTranspose> checkFun(apiName);
    checkFun.VerifyingParameters(dst, src, offset, scale, sharedTmpBuffer, K, shapeInfo);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_QUANTIZATION_ANTIQUANT_ANTIQUANT_CHECK_H_
