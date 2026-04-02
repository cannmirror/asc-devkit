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
    "impl/tensor_api/arch/vector_datamove/data_copy/npu_arch_3510/data_copy_gm2ub.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file data_copy_gm2ub.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2UB_H
#define IMPL_TENSOR_API_ARCH_VECTOR_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2UB_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "impl/experimental/tensor_api/arch/vector_datamove/data_copy/npu_arch_3510/instruction.h"

namespace AscendC {
namespace Te {

class DataCopyGM2UB3510 {
public:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline void Run(const T& dst, const U& src)
    { Execute<trait>(dst, src); }

private:
    template <const DataCopyTrait& trait, typename T, typename U>
    __aicore__ inline static void Execute(const T& dst, const U& src)
    {
        using SRC_TYPE = typename U::elementType;
        using DST_TYPE = typename T::elementType;
        static_assert(sizeof(SRC_TYPE) == sizeof(DST_TYPE),
                      "Source and destination element types must have the same size.");
        constexpr uint32_t ALIGN_BYTES = 32;

        const auto& dstLayout = dst.Layout();
        const auto& srcLayout = src.Layout();

        uint8_t cacheMode = GetCacheModeFromTensor(src);
        constexpr uint8_t leftPaddingCnt = 0;
        constexpr uint8_t rightPaddingCnt = 0;

        uint32_t C0_ELEMENT_SRC =
            GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
        uint32_t C0_ELEMENT_DST =
            GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);

        if constexpr (IsNDFormat<U>::value && IsNDFormat<T>::value) {
            uint16_t blockCount = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);

            uint32_t blockLen = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout)
                                * sizeof(SRC_TYPE);

            int64_t srcStride =
                GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout) * sizeof(SRC_TYPE);
            int64_t dstStride =
                GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout) * sizeof(DST_TYPE);

            if constexpr (is_b4_type<SRC_TYPE>) {
                // move fp4 as b8, need to be divided by 2
                blockLen = blockLen >> 1;
                srcStride = srcStride >> 1;
            }

            if constexpr (is_b4_type<DST_TYPE>) {
                dstStride = dstStride >> 1;
            }

            CopyGmToUbufAlignV2Instr::DataCopy(dst, src, blockCount, blockLen, leftPaddingCnt, rightPaddingCnt,
                                               srcStride, dstStride, cacheMode);
        } else if constexpr (IsDNFormat<U>::value && IsDNFormat<T>::value) {
            uint16_t blockCount =
                GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

            uint32_t blockLen =
                GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout) * sizeof(SRC_TYPE);

            int64_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout)
                                * sizeof(SRC_TYPE);
            int64_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout)
                                * sizeof(DST_TYPE);

            if constexpr (is_b4_type<SRC_TYPE>) {
                // move fp4 as b8, need to be divided by 2
                blockLen = blockLen >> 1;
                srcStride = srcStride >> 1;
            }

            if constexpr (is_b4_type<DST_TYPE>) {
                dstStride = dstStride >> 1;
            }

            CopyGmToUbufAlignV2Instr::DataCopy(dst, src, blockCount, blockLen, leftPaddingCnt, rightPaddingCnt,
                                               srcStride, dstStride, cacheMode);
        } else { // NZ format
            uint16_t blockCount =
                GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

            uint32_t row = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout)
                           * GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);

            uint32_t blockLen = row * C0_ELEMENT_SRC * sizeof(SRC_TYPE);
            int64_t srcStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout)
                                * sizeof(SRC_TYPE);
            int64_t dstStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(dstLayout)
                                * sizeof(DST_TYPE);

            if constexpr (is_b4_type<SRC_TYPE>) {
                // move fp4 as b8, need to be divided by 2
                blockLen = blockLen >> 1;
                srcStride = srcStride >> 1;
            }

            if constexpr (is_b4_type<DST_TYPE>) {
                dstStride = dstStride >> 1;
            }

            CopyGmToUbufAlignV2Instr::DataCopy(dst, src, blockCount, blockLen, leftPaddingCnt, rightPaddingCnt,
                                               srcStride, dstStride, cacheMode);
        }
        // ND和DN场景，需要保证UB上申请的空间和tensor的stride满足32字节对齐，否则CopyGmToUbufAlignV2会有问题，无法正确加载数据，导致数据错误
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_VECTOR_DATAMOVE_DATA_COPY_NPU_ARCH_3510_DATA_COPY_GM2UB_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif