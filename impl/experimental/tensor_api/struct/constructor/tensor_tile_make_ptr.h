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
* \file tensor_tile_make_ptr.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_PTR_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_PTR_H

#include "impl/experimental/tensor_api/struct/definition/tensor_tile_tensor.h"

namespace AscendC {
namespace TileInternal
{
template <Hardware hardPos, typename Iterator>
__aicore__ inline constexpr auto MakeMemPtr(Iterator iter) 
{
    if constexpr (IsHardwareMem<hardPos, Iterator>::value) {
        return iter;
    } else {
        return HardwareMemPtr<hardPos, Iterator>{iter};
    }
}
}

} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_PTR_H