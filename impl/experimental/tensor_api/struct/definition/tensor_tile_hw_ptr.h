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
* \file tensor_tile_hw_ptr.h
* \brief
*/
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_HW_PTR_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_HW_PTR_H

#include "impl/experimental/tensor_api/struct/definition/tensor_tile_pointer.h"

namespace AscendC {
namespace TileInternal {
template <Hardware hPos, typename Pointer>
struct HardwareMemPtr : IterAdaptor<Pointer, HardwareMemPtr<hPos, Pointer>> {
    using IterAdaptor<Pointer, HardwareMemPtr<hPos, Pointer>>::IterAdaptor;
    static constexpr const Hardware hardPos = hPos;
};

// is hardware mem
template <Hardware hardPos, typename Pointer, typename = void>
struct IsHardwareMem : Std::false_type {};

template <Hardware hardPos, typename Pointer>
struct IsHardwareMem<hardPos, HardwareMemPtr<hardPos, Pointer>> : Std::true_type {};

template <Hardware hardPos, typename Pointer>
struct IsHardwareMem<hardPos, Pointer, TileInternal::void_t<typename Pointer::iterator>> : IsHardwareMem<hardPos, typename Pointer::iterator> {};

template <Hardware hardPos, typename Pointer>
constexpr bool IsHardwareMemV = IsHardwareMem<hardPos, Pointer>::value;

}
} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_HW_PTR_H