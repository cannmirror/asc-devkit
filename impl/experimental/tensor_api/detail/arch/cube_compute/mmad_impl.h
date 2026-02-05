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
#ifndef IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_IMPL_H
#define IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_IMPL_H

namespace AscendC {
struct MmadTrait {
    int32_t fmOffset = 0;
    bool enSsparse = false;
    bool enWinogradA = false;
    bool enWinogradB = false;
    int8_t unitFlag = 0;
    bool kDirectionAlign = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;

   __aicore__ constexpr MmadTrait () {};

   __aicore__ constexpr MmadTrait (int32_t fmOffsetIn, bool enSsparseIn, bool enWinogradAIn, bool enWinogradBIn, 
      int8_t unitFlagIn, bool kDirectionAlignIn, bool cmatrixSourceIn, bool cmatrixInitValIn) 
   {
      fmOffset = fmOffsetIn;
      enSsparse = enSsparseIn;
      enWinogradA = enWinogradAIn;
      enWinogradB = enWinogradBIn;
      unitFlag = unitFlagIn;
      kDirectionAlign = kDirectionAlignIn;
      cmatrixSource = cmatrixSourceIn;
      cmatrixInitVal = cmatrixInitValIn;
   };
};
constexpr MmadTrait DEFAULT_MMAD_TRAIT; 
}

#include "impl/experimental/tensor_api/detail/arch/cube_compute/mmad/npu_arch_2201/mmad_routing.h"

namespace AscendC {

template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S>
__aicore__ inline typename Std::enable_if<VerifyingMmadTemplate<T, U, S>, void>::type 
Mmad(const T& dst, const U& fm, const S& filter)
{
   constexpr Hardware dstPos = TensorInternal::GetHardPos<T>();
   constexpr Hardware fmPos = TensorInternal::GetHardPos<U>();
   constexpr Hardware filterPos = TensorInternal::GetHardPos<S>();
   using Tensor2Tensor = typename TensorInternal::MmadTensor2Tensor<dstPos, fmPos, filterPos, 
      TensorInternal::CURRENT_ARCH_VERSION, TensorInternal::FOUR_DIM_DATA>::type;
   Tensor2Tensor{}.template Run<T, U, S, trait>(dst, fm, filter);
}

template <const MmadTrait& trait = DEFAULT_MMAD_TRAIT, typename T, typename U, typename S, typename V>
__aicore__ inline typename Std::enable_if<VerifyingMmadWithBiasTemplate<T, U, S, V>, void>::type 
Mmad(const T& dst, const U& fm, const S& filter, const V& bias)
{
   constexpr Hardware dstPos = TensorInternal::GetHardPos<T>();
   constexpr Hardware fmPos = TensorInternal::GetHardPos<U>();
   constexpr Hardware filterPos = TensorInternal::GetHardPos<S>();
   constexpr Hardware biasPos = TensorInternal::GetHardPos<V>();
   using Tensor2Tensor = typename TensorInternal::MmadWithBiasTensor2Tensor<dstPos, fmPos, filterPos, biasPos, 
      TensorInternal::CURRENT_ARCH_VERSION, TensorInternal::FOUR_DIM_DATA>::type;
   Tensor2Tensor{}.template Run<T, U, S, V, trait>(dst, fm, filter, bias);
}
} // namespace AscendC
#endif // IMPL_TENSOR_API_ARCH_CUBE_COMPUTE_MMAD_IMPL_H