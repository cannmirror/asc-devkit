/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_cache_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_CACHE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_CACHE_IMPL_H

#include <cstdint>
#include "kernel_log.h"
#include "kernel_macros.h"

namespace AscendC {

template <typename T>
__aicore__ inline void DataCachePreloadImpl(const GlobalTensor<uint64_t> &src, const T cacheOffset)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "DataCachePreload");
}

__aicore__ inline int64_t GetICachePreloadStatusImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetICachePreloadStatus");
    return 0;
}

__aicore__ inline void PreLoad(const int64_t preFetchLen)
{
    ASCENDC_ASSERT((false), "ICachePreLoad is not supported on current device");
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_CACHE_IMPL_H
