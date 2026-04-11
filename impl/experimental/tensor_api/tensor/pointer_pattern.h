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
 * \file pointer_pattern.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_H
#define IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_H

#include "impl/experimental/tensor_api/utils/map_impl.h"
#include "impl/experimental/tensor_api/tensor/pointer_pattern_impl.h"

namespace AscendC {
namespace Te {
    
struct EmptyTrait {};
struct GMMemPtr {};
struct UBMemPtr {};
struct L1MemPtr {};
struct L0AMemPtr {};
struct L0BMemPtr {};
struct L0CMemPtr {};
struct BiasMemPtr {};
struct FixbufMemPtr {};

using PtrMemSet = TupleMap<
    Std::tuple<GMMemPtr, MakeGMMemPtr>,
    Std::tuple<UBMemPtr, MakeUBMemPtr>,
    Std::tuple<L1MemPtr, MakeL1MemPtr>,
    Std::tuple<L0AMemPtr, MakeL0AMemPtr>,
    Std::tuple<L0BMemPtr, MakeL0BMemPtr>,
    Std::tuple<L0CMemPtr, MakeL0CMemPtr>,
    Std::tuple<BiasMemPtr, MakeBiasMemPtr>,
    Std::tuple<FixbufMemPtr, MakeFixbufMemPtr>>;

template <typename PtrPattern, typename TraitType, typename... Args>
__aicore__ inline constexpr auto MakeMemPtr(Args... args)
{
    using GetPtrMakeFun = typename PtrMemSet::template Get<PtrPattern>;
    static_assert(!Std::is_same_v<GetPtrMakeFun, EmptyValue>, "Unsupported pointer pattern.");
    return GetPtrMakeFun::template Make<TraitType>(args...);
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_POINTER_PATTERN_H
