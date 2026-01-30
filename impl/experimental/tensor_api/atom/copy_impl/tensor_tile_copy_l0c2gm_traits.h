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
* \file tensor_tile_copy_l0c2gm_traits.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ATOM_COPY_IMPL_TENSOR_TILE_COPY_L0C2GM_TRAITS_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ATOM_COPY_IMPL_TENSOR_TILE_COPY_L0C2GM_TRAITS_H

#include "impl/experimental/tensor_api/utils/tensor_tile_utils.h"
#include "impl/experimental/tensor_api/arch/tensor_tile_arch.h"

namespace AscendC {

struct FixpipeTraitDefault {
    using TraitType = FixpipeTrait;
    static constexpr const TraitType value = DEFAULT_FIXPIPE_TRAIT;
};

struct CopyL0C2GM {
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Copy(const Args& ...args)
    {
        Fixpipe<traits, Args...>(args...);
    }
};

}

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ATOM_COPY_IMPL_TENSOR_TILE_COPY_L0C2GM_TRAITS_H