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
 * \file reduce_check
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_REDUCE_CHECK_UTILS_H_
#define IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_REDUCE_CHECK_UTILS_H_

#include "../basic_check/datatype_check.h"
#include "../basic_check/calcount_check.h"
#include "../basic_check/reuse_source_check.h"
#include "../basic_check/single_tensor_check.h"
#include "../basic_check/multiple_tensor_check.h"

namespace AscendC {  
namespace HighLevelApiCheck {
class CheckFuncClassReduceBase : public CalCountCheckFuncBasicClass, public DataTypeCheckFuncBasicClass,
    public MultipleTensorCheckFuncBasicClass, public ReuseSourceCheckFuncBasicClass,
    public SingleTensorCheckFuncBasicClass {
public:
    __aicore__ inline CheckFuncClassReduceBase() {};
    __aicore__ inline CheckFuncClassReduceBase(__gm__ const char *apiName) :
    CalCountCheckFuncBasicClass(apiName), DataTypeCheckFuncBasicClass(apiName),
    MultipleTensorCheckFuncBasicClass(apiName), ReuseSourceCheckFuncBasicClass(apiName), SingleTensorCheckFuncBasicClass(apiName)  {};
};
}  // namespace HighLevelApiCheck
}  // namespace AscendC


#endif // IMPL_API_CHECK_KERNEL_CHECK_REDUCE_REDUCE_REDUCE_CHECK_UTILS_H_
 