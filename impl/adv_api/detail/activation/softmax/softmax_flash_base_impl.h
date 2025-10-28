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
 * \file softmax_flash_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASE_IMPL_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "regbase/c310/softmax_flash_impl.h"
#else
#include "softmax_common/softmax_common_flash.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &sumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inSumTensor, const LocalTensor<T> &inMaxTensor, const SoftMaxTiling &tiling,
    bool isUpdate, const SoftMaxShapeInfo &softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (T, isReuseSource, isBasicBlock), (dstTensor, sumTensor, maxTensor, srcTensor,
        expMaxTensor, inSumTensor, inMaxTensor, tiling, isUpdate, softmaxShapeInfo));
    SoftmaxFlashCommonImpl<T, T, isReuseSource, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor,
        expMaxTensor, inSumTensor, inMaxTensor, tiling, isUpdate, softmaxShapeInfo);
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashImpl(const LocalTensor<half> &dstTensor, const LocalTensor<float> &sumTensor,
    const LocalTensor<float> &maxTensor, const LocalTensor<half> &srcTensor, const LocalTensor<half> &expMaxTensor,
    const LocalTensor<float> &inSumTensor, const LocalTensor<float> &inMaxTensor, const SoftMaxTiling &tiling,
    bool isUpdate, const SoftMaxShapeInfo &softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (T, isReuseSource, isBasicBlock), (dstTensor, sumTensor, maxTensor,
        srcTensor, expMaxTensor, inSumTensor, inMaxTensor, tiling, isUpdate, softmaxShapeInfo));
    SoftmaxFlashCommonImpl<half, float, isReuseSource, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor,
        expMaxTensor, inSumTensor, inMaxTensor, tiling, isUpdate, softmaxShapeInfo);
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &sumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inSumTensor, const LocalTensor<T> &inMaxTensor, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const SoftMaxTiling &tiling, bool isUpdate, const SoftMaxShapeInfo &softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (T, isReuseSource, isBasicBlock), (dstTensor, sumTensor, maxTensor,
        srcTensor, expMaxTensor, inSumTensor, inMaxTensor, sharedTmpBuffer, tiling, isUpdate, softmaxShapeInfo));
    SoftmaxFlashTmpBufCommonImpl<T, T, isReuseSource, isBasicBlock>(dstTensor, sumTensor, maxTensor,
        srcTensor, expMaxTensor, inSumTensor, inMaxTensor, sharedTmpBuffer, tiling, isUpdate, softmaxShapeInfo);
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashImpl(const LocalTensor<half> &dstTensor, const LocalTensor<float> &sumTensor,
    const LocalTensor<float> &maxTensor, const LocalTensor<half> &srcTensor, const LocalTensor<half> &expMaxTensor,
    const LocalTensor<float> &inSumTensor, const LocalTensor<float> &inMaxTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const SoftMaxTiling &tiling, bool isUpdate,
    const SoftMaxShapeInfo &softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (T, isReuseSource, isBasicBlock), (dstTensor, sumTensor, maxTensor,
        srcTensor, expMaxTensor, inSumTensor, inMaxTensor, sharedTmpBuffer, tiling, isUpdate, softmaxShapeInfo));
    SoftmaxFlashTmpBufCommonImpl<half, float, isReuseSource, isBasicBlock>(dstTensor, sumTensor, maxTensor,
        srcTensor, expMaxTensor, inSumTensor, inMaxTensor, sharedTmpBuffer, tiling, isUpdate, softmaxShapeInfo);
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASE_IMPL_H