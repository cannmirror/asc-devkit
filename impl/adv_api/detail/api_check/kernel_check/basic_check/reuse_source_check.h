/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reuse_source_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_BASIC_CHECK_REUSE_SOURCE_CHECK_H
#define IMPL_API_CHECK_KERNEL_CHECK_BASIC_CHECK_REUSE_SOURCE_CHECK_H

#include "basic_check_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {

class ReuseSourceCheckFuncBasicClass {
public:
    __aicore__ inline ReuseSourceCheckFuncBasicClass() {};
    __aicore__ inline ReuseSourceCheckFuncBasicClass(__gm__ const char *apiName) {
        this->apiName = apiName;
    };

public:
    template <bool isConfigurable = true>
    __aicore__ inline void IsReuseSourceVerifyingParameters(const bool isReuseSource, __gm__ const char* paraName)
    {   
        ReuseSourceKernelCheckVerify<isConfigurable>(isReuseSource, paraName);
        if constexpr (HighLevelAPIParametersPrint) {
            PrintParameters(isReuseSource, paraName);
        }
    }

private:
    template <bool isConfigurable = true>
    __aicore__ inline void ReuseSourceKernelCheckVerify(const bool isReuseSource, __gm__ const char* paraName) {
        if (!isConfigurable && isReuseSource == true) {
            KERNEL_LOG(KERNEL_WARN, "[%s] The parameter of %s is true, may not be effective.", apiName, paraName);
        }
    }
    __aicore__ inline void PrintParameters(const bool isReuseSource, __gm__ const char* paraName)
    {
        if (isReuseSource) {
            KERNEL_LOG(KERNEL_INFO, "[%s] The parameter of %s is true!", apiName, paraName);
        } else {
            KERNEL_LOG(KERNEL_INFO, "[%s] The parameter of %s is false!", apiName, paraName);
        }
    }

private:
    __gm__ const char *apiName = nullptr;
};

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_BASIC_CHECK_REUSE_SOURCE_CHECK_H