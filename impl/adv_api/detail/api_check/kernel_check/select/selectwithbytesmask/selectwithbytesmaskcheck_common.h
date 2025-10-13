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
 * \file selectwithbytesmaskcheck_common.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_SELECT_SELECTWITHBYTESMAKS_SELECTWITHBYTESMAKS_CHECK_COMMON_H_
#define IMPL_API_CHECK_KERNEL_CHECK_SELECT_SELECTWITHBYTESMAKS_SELECTWITHBYTESMAKS_CHECK_COMMON_H_

#include "../../basic_check/datatype_check.h"
#include "../../basic_check/single_tensor_check.h"
#include "../../basic_check/multiple_tensor_check.h"

namespace AscendC {
namespace HighLevelApiCheck {
constexpr uint32_t CHECK_SELECT_WITH_BYTES_MASK = 16;

class CheckSelectWithBytesMaskParamsClass {
public:
    template <typename T, typename U, bool isReuseMask, bool reverse = false>
    __aicore__ inline void CheckSelectWithBytesMaskParams(const LocalTensor<T> &dst, const LocalTensor<T> &srcTensor,
        T srcScalar, const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer,
        const SelectWithBytesMaskShapeInfo &info) {
        VerifyingParameters<T, U, isReuseMask, true>(dst, srcTensor, srcScalar, mask, sharedTmpBuffer, info);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters<T, U, isReuseMask, true>(dst, srcTensor, srcScalar, mask, sharedTmpBuffer, info);
        }
    }

private:
    template <typename T, typename U, bool isReuseMask, bool reverse = false>
    __aicore__ inline void VerifyingParameters(const LocalTensor<T> &dst, const LocalTensor<T> &srcTensor,
        T srcScalar, const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer,
        const SelectWithBytesMaskShapeInfo &info) {
        ASCENDC_ASSERT((srcTensor.GetSize() == info.firstAxis * info.srcLastAxis || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
            "[SelectWithBytesMask] The result of info.firstAxis * info.srcLastAxis is %u, should be equal to srcTensor size %u.",
            info.firstAxis * info.srcLastAxis, srcTensor.GetSize()); });
        ASCENDC_ASSERT((mask.GetSize() == info.firstAxis * info.maskLastAxis || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR,
            "[SelectWithBytesMask] The result of info.firstAxis * info.maskLastAxis is %u, should be equal to mask size %u.",
            info.firstAxis * info.maskLastAxis, mask.GetSize()); });
        ASCENDC_ASSERT((info.maskLastAxis >= info.srcLastAxis || HighLevelAPIParametersPrint), { KERNEL_LOG(KERNEL_ERROR,
            "[SelectWithBytesMask] The info.maskLastAxis %u must be greater than or equal to info.srcLastAxis %u.",
            info.maskLastAxis, info.srcLastAxis); });
        ASCENDC_ASSERT((info.maskLastAxis * sizeof(U) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[SelectWithBytesMask] The info.maskLastAxis %u must be 32-byte aligned.",
            info.maskLastAxis); });
        ASCENDC_ASSERT((info.maskLastAxis % CHECK_SELECT_WITH_BYTES_MASK == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[SelectWithBytesMask] The info.maskLastAxis %u must be an integer multiple of 16.",
            info.maskLastAxis); });
        ASCENDC_ASSERT((info.srcLastAxis * sizeof(T) % ONE_BLK_SIZE == 0 || HighLevelAPIParametersPrint), {
            KERNEL_LOG(KERNEL_ERROR, "[SelectWithBytesMask] The info.srcLastAxis %u must be 32-byte aligned.",
            info.srcLastAxis); });
    }

    template <typename T, typename U, bool isReuseMask, bool reverse = false>
    __aicore__ inline void PrintParameters(const LocalTensor<T> &dst, const LocalTensor<T> &srcTensor,
        T srcScalar, const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer,
        const SelectWithBytesMaskShapeInfo &info) {
        KERNEL_LOG(KERNEL_INFO,
            "[SelectWithBytesMask] The info.firstAxis is %u, info.srcLastAxis is %u, info.maskLastAxis is %u.",
            info.firstAxis, info.srcLastAxis, info.maskLastAxis);
    }
};

template <typename T, typename U, bool isReuseMask, bool reverse = false>
class CheckFuncClassSelectWithBytesMask : public DataTypeCheckFuncBasicClass, public SingleTensorCheckFuncBasicClass,
    public MultipleTensorCheckFuncBasicClass, public CheckSelectWithBytesMaskParamsClass {
public:
    __aicore__ inline CheckFuncClassSelectWithBytesMask() {};
    __aicore__ inline CheckFuncClassSelectWithBytesMask(__gm__ const char *apiName) : DataTypeCheckFuncBasicClass(apiName),
        SingleTensorCheckFuncBasicClass(apiName), MultipleTensorCheckFuncBasicClass(apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T> &dst, const LocalTensor<T> &srcTensor,
        T srcScalar, const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer,
        const SelectWithBytesMaskShapeInfo &info) {
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<T, half, float>(
            "first template parameter (T) is not half or float");
        DataTypeCheckFuncBasicClass::DataTypeVerifyingParameters<U, bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(
            "second template parameter (U) is not bool/uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t");

        SingleTensorCheckFuncBasicClass::TensorVerifyingParameters(
            VA_ARGS_TO_MAKE_TUPLE(dst, srcTensor, mask, sharedTmpBuffer),
            VA_ARGS_TO_MAKE_TUPLE_STRING(TPosition::VECIN, TPosition::VECOUT, TPosition::VECCALC));

        CheckSelectWithBytesMaskParamsClass::CheckSelectWithBytesMaskParams<T, U, isReuseMask, true>(
            dst, srcTensor, srcScalar, mask, sharedTmpBuffer, info);
    };
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_SELECT_SELECTWITHBYTESMAKS_SELECTWITHBYTESMAKS_CHECK_COMMON_H_
 