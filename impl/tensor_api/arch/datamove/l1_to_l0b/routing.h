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
    "impl/tensor_api/arch/datamove/l1_to_l0b/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file routing.h
 * \brief
 */

#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0B_ROUTING_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0B_ROUTING_H

#include "impl/tensor_api/arch/datamove/l1_to_l0b/npu_arch_3510/l12l0b.h"

namespace AscendC {
namespace Te {

class CopyL12L0BIgnore {
public:
    template <const LoadDataTrait& trait, typename ...Args>
    __aicore__ inline void static Run(const Args&... args) {}
};

template <Hardware dstPos, Hardware srcPos, uint32_t Version>
struct CopyL12L0BTensor2Tensor {
    using type = CopyL12L0BIgnore;
};

template <Hardware dstPos, Hardware srcPos, uint32_t Version>
struct CopyL12L0BTensor2TensorNoCoord {
    using type = CopyL12L0BIgnore;
};

template <>
struct CopyL12L0BTensor2TensorNoCoord<Hardware::L0B, Hardware::L1, ArchVersion::V3510>
{
    using type = LoadDataL12L0B3510;
};

template <>
struct CopyL12L0BTensor2Tensor<Hardware::L0B, Hardware::L1, ArchVersion::V3510>
{
    using type = LoadDataL12L0BWithCoord3510;
};
} // namespace Te
} // namespace AscendC
#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0B_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
