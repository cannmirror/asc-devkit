/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_EXTRACT_EXTRACT_CHECK_310_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_EXTRACT_EXTRACT_CHECK_310_H

namespace AscendC {
namespace HighLevelApiCheck {
template <typename T>
class CheckFuncClassExtract {
public:
    __aicore__ inline CheckFuncClassExtract() {}
    __aicore__ inline CheckFuncClassExtract(__gm__ const char* apiName) {}

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstValue, const LocalTensor<uint32_t>& dstIndex,
        const LocalTensor<T>& sorted, const int32_t repeatTime)
    {}
};
} // namespace HighLevelApiCheck
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_SORT_EXTRACT_EXTRACT_CHECK_310_H