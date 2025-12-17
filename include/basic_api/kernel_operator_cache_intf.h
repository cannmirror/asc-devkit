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
 * \file kernel_operator_cache_intf.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_CACHE_INTF_H
#define ASCENDC_MODULE_OPERATOR_CACHE_INTF_H

#include "kernel_reg.h"

namespace AscendC {

template <typename T>
__aicore__ inline void DataCachePreload(const GlobalTensor<uint64_t>& src, const T cacheOffset);

#if defined(__NPU_ARCH__) &&            \
    ((__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3002))
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)
{
    DcciGMImpl<T, entireType, dcciDst>(const_cast<__gm__ T*>(dst.GetPhyAddr()));
}

template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst)
{
    DcciUBImpl<T, entireType, dcciDst>(const_cast<__ubuf__ T*>(dst.GetPhyAddr()));
}
#endif

#if defined(__NPU_ARCH__) &&                                                            \
    ((__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 3002))
template <typename T, CacheLine entireType>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)
{
    DcciGMImpl<T, entireType>(const_cast<__gm__ T*>(dst.GetPhyAddr()));
}
#endif

__aicore__ inline void ICachePreLoad(const int64_t preFetchLen);

__aicore__ inline int64_t GetICachePreloadStatus();

} // namespace AscendC

#endif // KERNEL_CACHE_INTF_H

