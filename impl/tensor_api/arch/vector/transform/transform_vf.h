/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/tensor_api/arch/vector/transform/transform_vf.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file transform_vf.h
* \brief
*/
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_TRANSFORM_TRANSFORM_VF_H
#define IMPL_TENSOR_API_ARCH_VECTOR_TRANSFORM_TRANSFORM_VF_H

#include "impl/tensor_api/arch/vector/binary/binary_vf.h"
#include "impl/tensor_api/arch/vector/cast/cast_vf.h"

namespace AscendC {
namespace Te {

template<typename CalcFunc, typename TraitType>
class Transform2VF {
private:
    using binarySet = Std::tuple<Inst::Add, Inst::Sub, Inst::Madd, Inst::And, Inst::Select, Inst::AbsSub, Inst::Min, Inst::Max,
        Inst::Or, Inst::Mul, Inst::ExpSubEven, Inst::ExpSubOdd, Inst::ShiftLeft, Inst::ShiftRight, Inst::Div, Inst::Xor, Inst::Prelu>;
    using castSet = Std::tuple<Inst::Ceil, Inst::U82U16>;

    using transformSet = TupleMap<
        Std::tuple<binarySet, Transform2BinaryVF<CalcFunc, TraitType>>,
        Std::tuple<castSet, Transform2CastVF<CalcFunc, TraitType>>>;

    template <typename Input>
    using Find = typename transformSet::template Find<Input>;

public:
    template<typename... Args>
    __aicore__ inline static void Run(const Args&... args) {
        using func = Find<CalcFunc>;
        func::Run(args...);
    }
};

}
}

#endif // IMPL_TENSOR_API_ARCH_VECTOR_TRANSFORM_TRANSFORM_VF_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
