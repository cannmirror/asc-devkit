/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file sort.h
 * \brief
 */

#ifndef LIB_SORT_SORT_H
#define LIB_SORT_SORT_H

#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#include "kernel_tensor.h"
#include "../../../impl/adv_api/detail/sort/sort/sort_impl.h"
#endif

/*
 * @ingroup Sort
 * @brief Sort them according to the value
 * @param [out] dstLocal output LocalTensor
 * @param [in] concatLocal input LocalTensor
 * @param [in] indexLocal input LocalTensor
 * @param [in] tmpLocal tmp buffer
 * @param [in] repeatTimes repeat times
 * 
 * template <typename T, bool isFullSort>
 * __aicore__ inline void Sort(const LocalTensor<T>& dstLocal, const LocalTensor<T>& concatLocal,
 *     const LocalTensor<uint32_t>& indexLocal, LocalTensor<T>& tmpLocal, const int32_t repeatTimes);
*/
#endif // LIB_SORT_SORT_H
