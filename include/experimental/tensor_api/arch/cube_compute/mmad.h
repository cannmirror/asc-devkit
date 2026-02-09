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
 * \file mmad.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_H
#define INCLUDE_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_H

#include "impl/experimental/tensor_api/detail/arch/cube_compute/mmad_impl.h"

namespace AscendC {

template <const MmadTrait& trait, typename T, typename U, typename S>
__aicore__ inline typename Std::enable_if<VerifyingMmadTemplate<T, U, S>, void>::type 
Mmad(const T& dst, const U& fm, const S& filter);

template <const MmadTrait& trait, typename T, typename U, typename S, typename V>
__aicore__ inline typename Std::enable_if<VerifyingMmadWithBiasTemplate<T, U, S, V>, void>::type 
Mmad(const T& dst, const U& fm, const S& filter, const V& bias);

} // namespace AscendC
#endif // INCLUDE_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_H