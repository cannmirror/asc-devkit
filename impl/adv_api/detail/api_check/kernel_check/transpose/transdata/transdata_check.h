
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
 * \file transdata_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_TRANSPOSE_TRANSDATA_TRANSDATA_CHECK_H
#define IMPL_API_CHECK_KERNEL_CHECK_TRANSPOSE_TRANSDATA_TRANSDATA_CHECK_H
#include "include/adv_api/transpose/transdata_common.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#include "transdata_check_common.h"
#else
#include "transdata_check_aicore.h"
#endif
namespace AscendC {  
namespace HighLevelApiCheck {
template <const TransDataConfig& config, typename T, typename U, typename S>
__aicore__ inline void CheckFuncTransData(__gm__ const char *apiName, const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const TransDataParams<U, S>& params)
{
    CheckFuncClassTransData<config, T, U, S> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, srcTensor, sharedTmpBuffer, params);
}
} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // IMPL_API_CHECK_KERNEL_CHECK_TRANSPOSE_TRANSDATA_TRANSDATA_CHECK_H
 
