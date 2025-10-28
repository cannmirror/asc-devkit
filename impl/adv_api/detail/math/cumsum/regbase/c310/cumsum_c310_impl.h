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

/* !
 * \file cumsum_common_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_CUMSUM_CUMSUM_COMMON_IMPL_C310_H
#define IMPL_MATH_CUMSUM_CUMSUM_COMMON_IMPL_C310_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "include/adv_api/math/cumsum_utils.h"
#include "cumsum_c310_utils.h"

namespace AscendC {
    template <typename T>
__simd_vf__ inline void CumSumCopyLastRowVF(__local_mem__ T* dst, __local_mem__ T* src, uint32_t len)
{
    constexpr uint16_t sregLower = static_cast<uint16_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint32_t count = len;
    uint16_t repeatTimes = CeilDivision(count, sregLower);

    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::MaskReg preg;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        preg = MicroAPI::UpdateMask<T>(count);
        MicroAPI::DataCopy(srcReg, src + i * sregLower);
        MicroAPI::DataCopy(dst + i * sregLower, srcReg, preg);
    }
}

template <typename T>
__aicore__ inline void CumSumCopyLastRow(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t len)
{
    __local_mem__ T* src = (__local_mem__ T*)srcTensor.GetPhyAddr();
    __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
    CumSumCopyLastRowVF<T>(dst, src, len);
}

template <typename T, const CumSumConfig& config>
__aicore__ inline void CumSumLastDim(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    LocalTensor<T> tempBuffer, const CumSumInfo& cumSumInfo)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float, const CumSumConfig& config>
__aicore__ inline void CumSumLastDim(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    LocalTensor<float> tempBuffer, const CumSumInfo& cumSumInfo)
{
    uint16_t alignOutter =
        (cumSumInfo.outter + NCHW_CONV_ADDR_LIST_SIZE - 1) / NCHW_CONV_ADDR_LIST_SIZE * NCHW_CONV_ADDR_LIST_SIZE;
    const uint32_t minTmpBufferSize = alignOutter * cumSumInfo.inner * 2 * sizeof(float); // 2: tempBuffer + tempBuffer2
    const uint32_t tmpBufferSize = tempBuffer.GetSize() * sizeof(float);
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
        KERNEL_LOG(KERNEL_ERROR,
            "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
            tmpBufferSize, minTmpBufferSize);
    });
#endif
    LocalTensor<float> tempBuffer2 = tempBuffer[alignOutter * cumSumInfo.inner];
    Internal::TransposeAB(tempBuffer, srcTensor, alignOutter, cumSumInfo.inner);
    if constexpr (config.algorithm == CumSumAlgorithm::CUMSUM_ALGORITHM_SKLANSKY) {
        Internal::CumSumFirstDimSklansky(tempBuffer, cumSumInfo.inner, alignOutter);
    } else {
        // basic implementation
        Internal::CumSumFirstDimBasic(tempBuffer, cumSumInfo.inner, alignOutter);
    }
    Internal::TransposeAB(tempBuffer2, tempBuffer, cumSumInfo.inner, alignOutter);
    Internal::CumSumCopyOut(dstTensor, tempBuffer2, cumSumInfo.outter, cumSumInfo.inner);
}

template <typename T = half, const CumSumConfig& config>
__aicore__ inline void CumSumLastDim(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    LocalTensor<half> tempBufferHalf, const CumSumInfo& cumSumInfo)
{
    uint16_t alignOutter =
        (cumSumInfo.outter + NCHW_CONV_ADDR_LIST_SIZE - 1) / NCHW_CONV_ADDR_LIST_SIZE * NCHW_CONV_ADDR_LIST_SIZE;

    const uint32_t minTmpBufferSize = alignOutter * cumSumInfo.inner * 2 * sizeof(half); // 2: tempBuffer + tempBuffer2
    LocalTensor<float> tempBuffer = tempBufferHalf.ReinterpretCast<float>();
    const uint32_t tmpBufferSize = tempBuffer.GetSize() * sizeof(float);
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
        KERNEL_LOG(KERNEL_ERROR,
            "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
            tmpBufferSize, minTmpBufferSize);
    });
#endif
    LocalTensor<float> tempBuffer2 = tempBuffer[alignOutter * cumSumInfo.inner];
    LocalTensor<half> tempBuffer2Half = tempBuffer2.ReinterpretCast<half>();
    Internal::TransposeAB(tempBuffer, srcTensor, alignOutter, cumSumInfo.inner);
    if constexpr (config.algorithm == CumSumAlgorithm::CUMSUM_ALGORITHM_SKLANSKY) {
        Internal::CumSumFirstDimSklansky(tempBuffer, cumSumInfo.inner, alignOutter);
    } else {
        // basic implementation
        Internal::CumSumFirstDimBasic(tempBuffer, cumSumInfo.inner, alignOutter);
    }
    Internal::TransposeAB(tempBuffer2Half, tempBuffer, cumSumInfo.inner, alignOutter);
    Internal::CumSumCopyOut(dstTensor, tempBuffer2Half, cumSumInfo.outter, cumSumInfo.inner);
}

template <typename T, const CumSumConfig& config>
__aicore__ inline void CumSumFirstDim(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    LocalTensor<uint8_t>& sharedTmpBuffer, const CumSumInfo& cumSumInfo)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float, const CumSumConfig& config>
__aicore__ inline void CumSumFirstDim(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    LocalTensor<uint8_t>& sharedTmpBuffer, const CumSumInfo& cumSumInfo)
{
    Internal::CumSumCopyOut(dstTensor, srcTensor, cumSumInfo.outter, cumSumInfo.inner);
    if constexpr (config.algorithm == CumSumAlgorithm::CUMSUM_ALGORITHM_SKLANSKY) {
        Internal::CumSumFirstDimSklansky(dstTensor, cumSumInfo.outter, cumSumInfo.inner);
    } else {
        // basic implementation
        Internal::CumSumFirstDimBasic(dstTensor, cumSumInfo.outter, cumSumInfo.inner);
    }
}

template <typename T = half, const CumSumConfig& config>
__aicore__ inline void CumSumFirstDim(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    LocalTensor<uint8_t>& sharedTmpBuffer, const CumSumInfo& cumSumInfo)
{
    const uint32_t minTmpBufferSize = cumSumInfo.outter * cumSumInfo.inner * sizeof(float);
    const uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
#if ASCENDC_CPU_DEBUG
    ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
        KERNEL_LOG(KERNEL_ERROR,
            "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
            tmpBufferSize, minTmpBufferSize);
    });
#endif
    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    Internal::CumSumCopyWithCast(tmpBuffer, srcTensor, cumSumInfo.outter, cumSumInfo.inner);
    if constexpr (config.algorithm == CumSumAlgorithm::CUMSUM_ALGORITHM_SKLANSKY) {
        Internal::CumSumFirstDimSklansky(tmpBuffer, cumSumInfo.outter, cumSumInfo.inner);
    } else {
        // basic implementation
        Internal::CumSumFirstDimBasic(tmpBuffer, cumSumInfo.outter, cumSumInfo.inner);
    }
    Internal::CumSumCopyWithCast(dstTensor, tmpBuffer, cumSumInfo.outter, cumSumInfo.inner);
}

template <typename T, const CumSumConfig& config>
__aicore__ inline void CumSumImpl(LocalTensor<T>& dstTensor, LocalTensor<T>& lastRowTensor,
    const LocalTensor<T>& srcTensor, LocalTensor<uint8_t>& sharedTmpBuffer, const CumSumInfo& cumSumInfo)
{
    if ASCEND_IS_AIC {
        return;
    }

    CHECK_FUNC_HIGHLEVEL_API(CumSum, (T, config), (dstTensor, lastRowTensor, srcTensor, sharedTmpBuffer, cumSumInfo));

    if constexpr (config.isLastAxis) {
        uint32_t minCastTempBufferSize = 0;
        if constexpr (sizeof(T) == 2) { // 2 is for half
            minCastTempBufferSize = cumSumInfo.inner * NCHW_CONV_ADDR_LIST_SIZE * sizeof(half);
        }
        const uint32_t minTmpBufferSize =
            minCastTempBufferSize
            + NCHW_CONV_ADDR_LIST_SIZE * cumSumInfo.inner * sizeof(T) * 2; // both transpose require a tempBuffer
        const uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
#if ASCENDC_CPU_DEBUG
        ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
            KERNEL_LOG(KERNEL_ERROR,
                "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
                tmpBufferSize, minTmpBufferSize);
        });
#endif
        // loop over the outter and process at least 16 rows of data each time.
        const uint32_t oneRepeateSize = tmpBufferSize / minTmpBufferSize * NCHW_CONV_ADDR_LIST_SIZE;
        const uint32_t rangeM = cumSumInfo.outter / oneRepeateSize;
        const uint32_t tailM = cumSumInfo.outter - oneRepeateSize * rangeM;
        uint32_t dstLocalOffset = 0;
        uint32_t srcLocalOffset = 0;
        LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
        for (uint32_t i = 0; i < rangeM; i++) {
            CumSumLastDim<T, config>(
                dstTensor[dstLocalOffset], srcTensor[srcLocalOffset], tmpBuffer, {oneRepeateSize, cumSumInfo.inner});
            dstLocalOffset += cumSumInfo.inner * oneRepeateSize;
            srcLocalOffset += cumSumInfo.inner * oneRepeateSize;
        }

        if (tailM != 0) {
            CumSumLastDim<T, config>(
                dstTensor[dstLocalOffset], srcTensor[srcLocalOffset], tmpBuffer, {tailM, cumSumInfo.inner});
        }
    } else {
        CumSumFirstDim<T, config>(dstTensor, srcTensor, sharedTmpBuffer, cumSumInfo);
    }
    if constexpr (config.outputLastRow) {
        CumSumCopyLastRow(lastRowTensor, dstTensor[(cumSumInfo.outter - 1) * cumSumInfo.inner], cumSumInfo.inner);
    }
}
} // namespace AscendC
#endif // IMPL_MATH_CUMSUM_CUMSUM_COMMON_IMPL_C310_H