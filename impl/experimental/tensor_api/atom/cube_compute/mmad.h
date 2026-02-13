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
* \file cube_mad.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ATOM_CUBE_COMPUTE_CUBE_MAD_H
#define IMPL_TENSOR_API_ATOM_CUBE_COMPUTE_CUBE_MAD_H

#include "include/experimental/tensor_api/utils/utils.h"

#include "impl/experimental/tensor_api/atom/mad_traits_impl.h"
#include "impl/experimental/tensor_api/arch/cube_compute/mmad/mmad_impl.h"

namespace AscendC {
namespace Te {

struct MmadTraitDefault {
    using TraitType = MmadTrait;
    static constexpr const TraitType value = DEFAULT_MMAD_TRAIT;
};

struct MmadOperation {
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Mad(const Args& ...args)
    {
        Mmad<traits, Args...>(args...);
    }
};

template <typename TraitStruct>
struct MmadTraits<MmadOperation, TraitStruct>
{
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void MmadUnpack(const Args& ...args) const {
    MmadOperation::Mad<TraitType, trait, Args...>(args...);
    }

    template <typename... Args>
    __aicore__ inline constexpr MmadTraits<MmadOperation, MmadTraitDefault>
    with(const Args& ...args) const
    {
        return {args...};
    }
};

template <>
struct MmadTraits<MmadOperation> : public MmadTraits<MmadOperation, MmadTraitDefault> {};

}
}

#endif // IMPL_TENSOR_API_ATOM_CUBE_COMPUTE_CUBE_MAD_H