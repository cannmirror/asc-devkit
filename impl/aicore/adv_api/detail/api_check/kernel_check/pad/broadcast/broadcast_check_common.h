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
 * \file broadcast_check_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_COMMON_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_COMMON_H

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/reuse_source_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
class CheckBroadcastParamsClass {
public:
    template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
    __aicore__ inline void CheckBroadcastParams(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t>& sharedTmpBuffer)
    {
        VerifyingParameters<T, dim, axis, isReuseSource>(dstLocal, srcLocal, dstShape, srcShape, sharedTmpBuffer);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T, dim, axis, isReuseSource>(dstLocal, srcLocal, dstShape, srcShape, sharedTmpBuffer);
        }
    }

private:
    template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t>& sharedTmpBuffer)
    {
        ASCENDC_ASSERT(((dim == 1 || dim == 2) || HighLevelAPIParametersPrint),
            { KERNEL_LOG(KERNEL_ERROR, "[Broadcast] The dim parameter cannot be %u, should be 1 or 2.", dim); });
        ASCENDC_ASSERT(((axis == 1 || axis == 0) || HighLevelAPIParametersPrint),
            { KERNEL_LOG(KERNEL_ERROR, "[Broadcast] The axis parameter cannot be %u, should be 0 or 1.", axis); });
#if __CCE_AICORE__ == 200
        if (dim == 2 && axis == 1) {
            ASCENDC_ASSERT((srcShape[0] * sizeof(T) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_ERROR,
                    "[Broadcast] The result of srcShape[0] * sizeof(T) is %u, "
                    "should be an integer multiple of 32 when dim = 2 and axis = 1.",
                    srcShape[0] * sizeof(T));
            });
        }
#endif
        if (dim == 2 && axis == 0) {
            ASCENDC_ASSERT((srcShape[1] * sizeof(T) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
                KERNEL_LOG(KERNEL_ERROR,
                    "[Broadcast] The result of srcShape[1] * sizeof(T) is %u, "
                    "should be an integer multiple of 32 when dim = 2 and axis = 0.",
                    srcShape[1] * sizeof(T));
            });
        }
    }

    template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
    __aicore__ inline void PrintParameters(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t>& sharedTmpBuffer)
    {
        KERNEL_LOG(KERNEL_INFO, "[Broadcast] The dim is %u, axis is %u.", dim, axis);
    }
};

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
class CheckFuncClassBroadcast : public DataTypeCheckFuncBasicClass,
                                public ReuseSourceCheckFuncBasicClass,
                                public SingleTensorCheckFuncBasicClass,
                                public MultipleTensorCheckFuncBasicClass,
                                public CheckBroadcastParamsClass {
public:
    __aicore__ inline CheckFuncClassBroadcast(){};
    __aicore__ inline CheckFuncClassBroadcast(__gm__ const char* apiName) :
        DataTypeCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
        const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t>& sharedTmpBuffer)
    {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, uint8_t, int8_t, half, float>(
            "template parameter (T) is not uint8_t/int8_t/half/float");

        ReuseSourceCheckFuncBasicClass::IsReuseSourceVerifyingParameters<false>(ARG_AND_STRING(isReuseSource));

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstLocal, srcLocal, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(VA_ARGS_TO_MAKE_TUPLE(dstLocal, srcLocal));

        CheckBroadcastParamsClass::CheckBroadcastParams<T, dim, axis, isReuseSource>(
            dstLocal, srcLocal, dstShape, srcShape, sharedTmpBuffer);
    };
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_PAD_BROADCAST_BROADCAST_CHECK_COMMON_H
