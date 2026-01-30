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
* \file tensor_tile_copy_atom.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ATOM_TENSOR_TILE_COPY_ATOM_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ATOM_TENSOR_TILE_COPY_ATOM_H

#include "impl/experimental/tensor_api/atom/tensor_tile_copy_traits.h"

namespace AscendC {

template <typename... Args>
struct CopyAtom;

template <typename... Args>
struct CopyAtom<CopyTraits<Args...>>: CopyTraits<Args...>
{
    using CopyTraitType = CopyTraits<Args...>;
    using TraitType = typename CopyTraitType::TraitType;
    static constexpr const TraitType defaultTrait = CopyTraitType::defaultTrait;

    template <const TraitType& traits = defaultTrait, typename... Params>
    __aicore__ inline static void Call(const Params& ...params) {
        CopyTraitType::template CopyUnpack<traits>(params...);
    }
};

template <typename Tp, const Tp& traits, typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)
{
    atomCopy.template Call<traits>(params...);
}

template <typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)
{
    atomCopy.Call(params...);
}

template <typename... Args>
__aicore__ inline auto MakeCopy(const Args& ...traits) {
    return CopyAtom<CopyTraits<Args...>>{};
}

}

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ATOM_TENSOR_TILE_COPY_ATOM_H