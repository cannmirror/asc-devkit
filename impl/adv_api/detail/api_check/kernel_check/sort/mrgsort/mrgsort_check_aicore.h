/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef IMPL_API_CHECK_KERNEL_CHECK_SORT_MRGSORT_MRGSORT_CHECK_AICORE_H_
#define IMPL_API_CHECK_KERNEL_CHECK_SORT_MRGSORT_MRGSORT_CHECK_AICORE_H_

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isExhaustedSuspension = false>
class CheckFuncClassMrgSort {
public:
    __aicore__ inline CheckFuncClassMrgSort()
    {}
    __aicore__ inline CheckFuncClassMrgSort(__gm__ const char* apiName)
    {}

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dst, const MrgSortSrcList<T>& sortList,
                                               const uint16_t elementCountList[4], uint32_t sortedNum[4],
                                               uint16_t validBit, const int32_t repeatTime)
    {}
};

}  // namespace HighLevelApiCheck
}  // namespace AscendC

#endif  // IMPL_API_CHECK_KERNEL_CHECK_SORT_MRGSORT_MRGSORT_CHECK_AICORE_H_