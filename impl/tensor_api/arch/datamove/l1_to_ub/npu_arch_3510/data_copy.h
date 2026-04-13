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
    "impl/tensor_api/arch/datamove/l1_to_ub/npu_arch_3510/data_copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12UB_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12UB_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/arch/datamove/l1_to_ub/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

using CopyL12UBTrait = DataCopyTrait;

class DataCopyL12UB3510 {
public:
    template <const CopyL12UBTrait& trait, typename T, typename U>
    __aicore__ inline static void Run(const T& dst, const U& src)
    { Execute<trait>(dst, src); }

private:
    template <const CopyL12UBTrait& trait, typename T, typename U>
    __aicore__ inline static void Execute(const T& dst, const U& src)
    {
        using SRC_TYPE = typename U::elementType;
        using DST_TYPE = typename T::elementType;
        static_assert(sizeof(SRC_TYPE) == sizeof(DST_TYPE),
                      "Source and destination element types must have the same size.");
        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();
        if constexpr (IsNZFormat<U>::value && IsNZFormat<T>::value) {
            uint16_t blockCount =
                GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

            // Next three parameters are in unit of 32B
            uint32_t blockLen = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
                                * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);

            int64_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout)
                                    / C0_ELEMENT<SRC_TYPE>
                                - blockLen;
            int64_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout)
                                    / C0_ELEMENT<DST_TYPE>
                                - blockLen;

            CopyCbufToUbufInstr::DataCopy(dst, src, blockCount, blockLen, srcStride, dstStride);
        } else if constexpr (IsNDFormat<U>::value && IsNDFormat<T>::value) {
            uint16_t blockCount = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
            // Next three parameters are in unit of 32B
            uint32_t blockLen = Std::ceil_division(
                GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout),
                C0_ELEMENT<SRC_TYPE>);

            int64_t srcStride = Std::ceil_division(
                (GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout)
                 - GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout)),
                C0_ELEMENT<SRC_TYPE>);
            int64_t dstStride = Std::ceil_division(
                (GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout)
                 - GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout)),
                C0_ELEMENT<DST_TYPE>);

            CopyCbufToUbufInstr::DataCopy(dst, src, blockCount, blockLen, srcStride, dstStride);
        } else if constexpr (IsDNFormat<U>::value && IsDNFormat<T>::value) {
            uint16_t blockCount =
                GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
            // Next three parameters are in unit of 32B
            uint32_t blockLen =
                Std::ceil_division(GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout),
                                   C0_ELEMENT<SRC_TYPE>);

            int64_t srcStride = Std::ceil_division(
                (GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout)
                 - GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout)),
                C0_ELEMENT<SRC_TYPE>);
            int64_t dstStride = Std::ceil_division(
                (GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout)
                 - GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout)),
                C0_ELEMENT<DST_TYPE>);

            CopyCbufToUbufInstr::DataCopy(dst, src, blockCount, blockLen, srcStride, dstStride);
        }
        // ND和DN场景，需要保证UB和L1上申请的空间和tensor的stride满足32字节对齐，否则CopyCbufToUbuf会有问题，无法正确加载数据，导致数据错误
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_L12UB_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif