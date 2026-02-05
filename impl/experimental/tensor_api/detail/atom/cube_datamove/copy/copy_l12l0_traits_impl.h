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
* \file copy_l12l0_traits_impl.h
* \brief
*/
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_ATOM_CUBE_DATAMOVE_COPY_COPY_L12L0_TRAITS_IMPL_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_ATOM_CUBE_DATAMOVE_COPY_COPY_L12L0_TRAITS_IMPL_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "include/experimental/tensor_api/arch/arch.h"

namespace AscendC {

struct LoadDataTraitDefault {
    using TraitType = LoadDataTrait;
    static constexpr const TraitType value = DEFAULT_LOAD_DATA_TRAIT;
};

struct CopyL12L0 {
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Copy(const Args& ...args)
    {
        LoadData<traits, Args...>(args...);
    }
};

}

#endif // EXPERIMENTAL_TENSOR_API_DETAIL_ATOM_CUBE_DATAMOVE_COPY_COPY_L12L0_TRAITS_IMPL_H