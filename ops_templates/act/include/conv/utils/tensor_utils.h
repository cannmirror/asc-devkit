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
 * \file tensor_utils.h
 * \brief
 */

#ifndef UTILS_TENSOR_UTILS_H
#define UTILS_TENSOR_UTILS_H
#include "conv_integral_constant.h"

namespace AscendC {
template <typename Tp>
struct is_global_tensor : public Std::false_type {};

template <typename Tp>
struct is_global_tensor<GlobalTensor<Tp>> : public Std::true_type {};

template <typename Tp>
constexpr bool is_global_tensor_v = is_global_tensor<Tp>::value;

template <typename Tp>
struct is_local_tensor : public Std::false_type {};

template <typename Tp>
struct is_local_tensor<LocalTensor<Tp>> : public Std::true_type {};

template <typename Tp>
constexpr bool is_local_tensor_v = is_local_tensor<Tp>::value;

template <typename Tp>
struct tensor_trait {
    static_assert(Std::always_false_v<Tp>, "Unsupported tensor type");
};

template <typename Tp>
struct tensor_trait<GlobalTensor<Tp>> {
    using trait_type = Tp;
};

template <typename Tp>
struct tensor_trait<LocalTensor<Tp>> {
    using trait_type = Tp;
};
} // namespace AscendC

namespace Act {
namespace Conv {

template <class AGlobalTensor_, class TensorType_>
__aicore__ inline void InitGlobalTensor(AGlobalTensor_& aGlobal, GM_ADDR gmAddr, uint64_t offset)
{
#if defined(__DAV_C310__)
    AscendC::GlobalTensor<TensorType_> aTmp;
    aTmp.SetGlobalBuffer(reinterpret_cast<__gm__ TensorType_*>(gmAddr + offset * sizeof(TensorType_)));
    aGlobal.address_ = aTmp.address_;
#else
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ TensorType_*>(gmAddr + offset * sizeof(TensorType_)));
#endif
}

} // namespace Conv
} // namespace Act
#endif