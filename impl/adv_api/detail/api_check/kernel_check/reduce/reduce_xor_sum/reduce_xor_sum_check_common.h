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
 * \file reduce_xor_sum_check
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_CHECK_COMMON_H_

#include "../reduce_check_utils.h"

namespace AscendC {  
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false>
class CheckFuncClassReduceXorSum : public CheckFuncClassReduceBase {
public:
    __aicore__ inline CheckFuncClassReduceXorSum() {};
    __aicore__ inline CheckFuncClassReduceXorSum(__gm__ const char *apiName) : CheckFuncClassReduceBase(apiName) {
    };

 public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount) {
        CalCountCheckFuncBasicClass::CalCountVerifyingParameters(ARG_AND_STRING(calCount),
            VA_ARGS_TO_MAKE_TUPLE(src0Tensor, src1Tensor));

        SingleTensorCheckFuncBasicClass::TPositionVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        SingleTensorCheckFuncBasicClass::TensorSizeVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, src0Tensor, sharedTmpBuffer));

        MultipleTensorCheckFuncBasicClass::TensorReuseVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dstTensor, src1Tensor, sharedTmpBuffer));
    };
};
} // namespace HighLevelApiCheck
} // AscendC
#endif // IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_XOR_SUM_REDUCE_XOR_SUM_CHECK_COMMON_H_
 