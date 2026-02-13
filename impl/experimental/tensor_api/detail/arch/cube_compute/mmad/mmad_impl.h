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
 * \file mmad_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_IMPL_H
#define IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_IMPL_H

#include "impl/experimental/tensor_api/detail/arch/cube_compute/mmad/mmad_routing.h"

namespace AscendC {
namespace Te {

template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename Params>
__aicore__ inline typename Std::enable_if<VerifyingMmadTemplate<T, U, S>, void>::type 
Mmad(const T& dst, const U& fm, const S& filter, const Params& params)
{
   static_assert(Std::is_one_of_v<Params, MmadParams>, "The type of params is not supported;");
   constexpr Hardware dstPos = GetHardPos<T>();
   constexpr Hardware fmPos = GetHardPos<U>();
   constexpr Hardware filterPos = GetHardPos<S>();
   using Tensor2Tensor = typename MmadTensor2Tensor<dstPos, fmPos, filterPos, Hardware::MAX, 
      CURRENT_ARCH_VERSION, FOUR_DIM_DATA>::type;
   Tensor2Tensor{}.template Run<trait>(dst, fm, filter, params);
}

template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename V, typename Params>
__aicore__ inline typename Std::enable_if<VerifyingMmadWithBiasTemplate<T, U, S, V>, void>::type 
Mmad(const T& dst, const U& fm, const S& filter, const V& bias, const Params& params)
{
   static_assert(Std::is_one_of_v<Params, MmadParams>, "The type of params is not supported;");
   constexpr Hardware dstPos = GetHardPos<T>();
   constexpr Hardware fmPos = GetHardPos<U>();
   constexpr Hardware filterPos = GetHardPos<S>();
   constexpr Hardware biasPos = GetHardPos<V>();
   using Tensor2Tensor = typename MmadTensor2Tensor<dstPos, fmPos, filterPos, biasPos, 
      CURRENT_ARCH_VERSION, FOUR_DIM_DATA>::type;
   Tensor2Tensor{}.template Run<trait>(dst, fm, filter, bias, params);
}
} // namespace Te
} // namespace AscendC
#endif // IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_MMAD_IMPL_H