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
 * \file log_check_aicore.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_MATH_LOG_LOG_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_MATH_LOG_LOG_CHECK_AICORE_H_

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T, bool isReuseSource = false>
class CheckFuncClassLog  {
public:
    __aicore__ inline CheckFuncClassLog() {};
    __aicore__ inline CheckFuncClassLog(__gm__ const char *apiName) {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        uint32_t calCount) {};
};

template <typename T, bool isReuseSource = false>
class CheckFuncClassLog2  {
public:
    __aicore__ inline CheckFuncClassLog2() {};
    __aicore__ inline CheckFuncClassLog2(__gm__ const char *apiName)  {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount) {};
};

template <typename T, bool isReuseSource = false>
class CheckFuncClassLog10 {
public:
    __aicore__ inline CheckFuncClassLog10() {};
    __aicore__ inline CheckFuncClassLog10(__gm__ const char *apiName)  {};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
        uint32_t calCount) {};
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_MATH_LOG_LOG_CHECK_AICORE_H_
