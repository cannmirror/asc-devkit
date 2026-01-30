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
 * \file tensor_tile_ptr_impl.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_TILE_PTR_IMPL_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_TILE_PTR_IMPL_H

#include "impl/experimental/tensor_api/utils/tensor_tile_utils.h"
#include "impl/experimental/tensor_api/struct/tensor_tile_struct.h"

namespace AscendC {

template <typename Iterator>
__aicore__ inline constexpr auto MakeGMmemPtr(Iterator iter) {
    return TileInternal::MakeMemPtr<Hardware::GM, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeUBmemPtr(Iterator iter) {
    return TileInternal::MakeMemPtr<Hardware::UB, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL1memPtr(Iterator iter) {
    return TileInternal::MakeMemPtr<Hardware::L1, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0AmemPtr(Iterator iter) {
    return TileInternal::MakeMemPtr<Hardware::L0A, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0BmemPtr(Iterator iter) {
    return TileInternal::MakeMemPtr<Hardware::L0B, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0CmemPtr(Iterator iter) {
    return TileInternal::MakeMemPtr<Hardware::L0C, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeBiasmemPtr(Iterator iter) {
    return TileInternal::MakeMemPtr<Hardware::BIAS, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeFixbufmemPtr(Iterator iter) {
    return TileInternal::MakeMemPtr<Hardware::FIXBUF, Iterator>(iter);
}

} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_PTR_IMPL_H
