/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_micro_vec_createvecindex_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_CREATEVECINDEX_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_CREATEVECINDEX_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, IndexOrder order = IndexOrder::INCREASE_ORDER, typename T1, typename RegT>
__aicore__ inline void ArangeImpl(RegT &dstReg, T1 scalar)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, int8_t, int16_t, int32_t, float, half>()),
        "current Arange data type is not supported on current device!");
    constexpr auto orderMode = std::integral_constant<::Order, static_cast<::Order>(order)>();
    vci(dstReg, (ActualT)scalar, orderMode);
}

template <typename T = DefaultType, typename T1, typename RegT>
__aicore__ inline void ArangeWithPatternImpl(RegT &dstReg, T1 scalar)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ArangeWithPattern is not supported on current device!"); });
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_CREATEVECINDEX_IMPL_H