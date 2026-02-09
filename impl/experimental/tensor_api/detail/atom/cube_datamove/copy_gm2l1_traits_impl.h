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
* \file copy_gm2l1_traits_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_GM2L1_TRAITS_IMPL_H
#define IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_GM2L1_TRAITS_IMPL_H

#include "include/experimental/tensor_api/utils/utils.h"

#include "impl/experimental/tensor_api/detail/arch/cube_datamove/data_copy_impl.h"
#include "impl/experimental/tensor_api/detail/atom/copy_traits_impl.h"

namespace AscendC {
namespace Te {

struct DataCopyTraitDefault {
    using TraitType = DataCopyTrait;
    static constexpr const TraitType value = DEFAULT_DATA_COPY_TRAIT;
};

struct CopyGM2L1 {
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Copy(const Args& ...args)
    {
        DataCopy<traits, Args...>(args...);
    }
};

template <typename TraitStruct>
struct CopyTraits<CopyGM2L1, TraitStruct>
{
    using TraitType = typename TraitStruct::TraitType;
    static constexpr const TraitType defaultTrait = TraitStruct::value;

    template <const TraitType& trait = defaultTrait, typename... Args>
    __aicore__ inline void CopyUnpack(const Args& ...args) const {
      CopyGM2L1::Copy<TraitType, trait, Args...>(args...);
    }

    template <typename... Args>
    __aicore__ inline constexpr CopyTraits<CopyGM2L1, DataCopyTraitDefault>
    with(const Args& ...args) const
    {
        return {args...};
    }
};

template <>
struct CopyTraits<CopyGM2L1> : public CopyTraits<CopyGM2L1, DataCopyTraitDefault> {};

}
}

#endif // IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_GM2L1_TRAITS_IMPL_H