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
* \file copy_traits_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_TRAIT_IMPL_H
#define IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_TRAIT_IMPL_H

#include "impl/experimental/tensor_api/detail/atom/cube_datamove/copy/copy_gm2l1_traits_impl.h"
#include "impl/experimental/tensor_api/detail/atom/cube_datamove/copy/copy_l0c2gm_traits_impl.h"
#include "impl/experimental/tensor_api/detail/atom/cube_datamove/copy/copy_l12l0_traits_impl.h"

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

#endif // IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_TRAIT_IMPL_H