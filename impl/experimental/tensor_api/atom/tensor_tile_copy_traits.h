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
* \file tensor_tile_copy_traits.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ATOM_TENSOR_TILE_COPY_TRAIT_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ATOM_TENSOR_TILE_COPY_TRAIT_H

#include "impl/experimental/tensor_api/atom/copy_impl/tensor_tile_copy_gm2l1_traits.h"
#include "impl/experimental/tensor_api/atom/copy_impl/tensor_tile_copy_l0c2gm_traits.h"
#include "impl/experimental/tensor_api/atom/copy_impl/tensor_tile_copy_l12l0_traits.h"

namespace AscendC {

template <typename CopyOperation, typename TraitType, typename... CopyOpArgs>
struct CopyTraits{};

template <typename TraitStruct>
struct CopyTraits<CopyGM2L1, TraitStruct>
{
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline  static void CopyUnpack(const Args& ...args) {
      CopyGM2L1::Copy<TraitType, trait, Args...>(args...);
    }
};

template <typename TraitStruct>
struct CopyTraits<CopyL0C2GM, TraitStruct>
{
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline  static void CopyUnpack(const Args& ...args) {
      CopyL0C2GM::Copy<TraitType, trait, Args...>(args...);
    }
};

template <typename TraitStruct>
struct CopyTraits<CopyL12L0, TraitStruct>
{
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline  static void CopyUnpack(const Args& ...args) {
      CopyL12L0::Copy<TraitType, trait, Args...>(args...);
    }
};

}

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ATOM_TENSOR_TILE_COPY_TRAIT_H