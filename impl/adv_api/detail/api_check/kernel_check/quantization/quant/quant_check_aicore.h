/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_QUANTIZATION_QUANT_QUANT_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_QUANTIZATION_QUANT_QUANT_CHECK_AICORE_H_

#include "include/adv_api/quantization/ascend_quant_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource, const AscendQuantConfig& config>
class CheckFuncClassAscendQuantTensor {
public:
    __aicore__ inline CheckFuncClassAscendQuantTensor() {};
    __aicore__ inline CheckFuncClassAscendQuantTensor(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const float scale, const float offset, const uint32_t calCount) {};
};

template <typename T, bool isReuseSource, const AscendQuantConfig& config>
class CheckFuncClassAscendQuantChannelOffset {
public:
    __aicore__ inline CheckFuncClassAscendQuantChannelOffset() {};
    __aicore__ inline CheckFuncClassAscendQuantChannelOffset(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
        const T offset, const uint32_t scaleCount, const uint32_t calCount) {};
};

template <typename T, bool isReuseSource, const AscendQuantConfig& config>
class CheckFuncClassAscendQuantChannelOffsetTensor {
public:
    __aicore__ inline CheckFuncClassAscendQuantChannelOffsetTensor() {};
    __aicore__ inline CheckFuncClassAscendQuantChannelOffsetTensor(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<int8_t>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<T>& scaleTensor,
        const LocalTensor<T>& offsetTensor, const uint32_t scaleCount, const uint32_t offsetCount,
        const uint32_t calCount) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_QUANTIZATION_QUANT_QUANT_CHECK_AICORE_H_
 