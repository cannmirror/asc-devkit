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
* \file make_pointer_impl.h
* \brief
*/
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_TENSOR_MAKE_POINTER_IMPL_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_TENSOR_MAKE_POINTER_IMPL_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "impl/experimental/tensor_api/detail/tensor/hardware_pointer.h"

namespace AscendC {
namespace TensorInternal {
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

// make_pointer.h
namespace AscendC {
template <typename Iterator>
__aicore__ inline constexpr auto MakeGMmemPtr(Iterator iter) {
    return TensorInternal::MakeMemPtr<Hardware::GM, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeUBmemPtr(Iterator iter) {
    return TensorInternal::MakeMemPtr<Hardware::UB, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL1memPtr(Iterator iter) {
    return TensorInternal::MakeMemPtr<Hardware::L1, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0AmemPtr(Iterator iter) {
    return TensorInternal::MakeMemPtr<Hardware::L0A, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0BmemPtr(Iterator iter) {
    return TensorInternal::MakeMemPtr<Hardware::L0B, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeL0CmemPtr(Iterator iter) {
    return TensorInternal::MakeMemPtr<Hardware::L0C, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeBiasmemPtr(Iterator iter) {
    return TensorInternal::MakeMemPtr<Hardware::BIAS, Iterator>(iter);
}

template <typename Iterator>
__aicore__ inline constexpr auto MakeFixbufmemPtr(Iterator iter) {
    return TensorInternal::MakeMemPtr<Hardware::FIXBUF, Iterator>(iter);
}
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_DETAIL_TENSOR_MAKE_POINTER_IMPL_H