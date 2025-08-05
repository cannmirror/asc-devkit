/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_info.h
 * \brief
 */
#ifndef ATVC_TENSOR_INFO_H
#define ATVC_TENSOR_INFO_H

#include <type_traits>
#include "kernel_operator.h"
#include "type_list.h"
#include "forward.h"
#include "tuple.h"

namespace ATVC {
// 记录一个LocalTensor在In/Out/Local中的偏移位置，以及自己的类型
template<typename T>
struct TensorInfo {
    __aicore__ inline TensorInfo() {}
    using Dtype = T;
    AscendC::GlobalTensor<T> gmTensor;
    int32_t local_offset;
};

template <typename T>
struct TypeToTensor {
    using Type = TensorInfo<T>;
};

template <typename List>
struct TensorTuple {
private:
    using Tensors = typename ATVC::TypeListMap<List, TypeToTensor>::Type;
public:
    using Type = typename ATVC::TupleFromTypeList<Tensors>::Type;
};

/////////////////////////////////////////////////////////////////////////////////////
// 获取特定索引的 Tuple 元素的对应的LocalTensor
template <std::size_t N, typename TupleType>
__aicore__ inline auto TupleElemGetLocalTensor(AscendC::LocalTensor<uint8_t> local, TupleType& tuple, uint32_t size) {
    using Dtype = typename ATVC::TupleElemType<N, TupleType>::Type::Dtype;
    if constexpr (N == 0) {
        auto tensor = local[tuple.head.local_offset].template ReinterpretCast<Dtype>();
        tensor.SetSize(size);
        return tensor;
    } else {
        return TupleElemGetLocalTensor<N - 1>(local, tuple.tail, size);
    }
}

/////////////////////////////////////////////////////////////////////////////////////

}

#endif // ATVC_TENSOR_INFO_H

