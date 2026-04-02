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
    "impl/tensor_api/arch/vector_datamove/data_copy/npu_arch_3510/instruction.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_VECTOR_DATAMOVE_DATA_COPY_NPU_ARCH_3510_INSTRUCTION_H
#define IMPL_TENSOR_API_ARCH_VECTOR_DATAMOVE_DATA_COPY_NPU_ARCH_3510_INSTRUCTION_H

#include "impl/experimental/tensor_api/tensor/pointer_impl.h"
#include "impl/experimental/tensor_api/tensor/local_tensor_impl.h"
#include "impl/experimental/tensor_api/arch/utils/arch_utils.h"

namespace AscendC {
namespace Te {

class CopyGmToUbufAlignV2Instr {
public:
    template <typename T, typename U, typename... Params>
    __aicore__ inline static void DataCopy(const T& dst, const U& src, const Params&... params)
    {
        using srcType = typename U::elementType;
        if constexpr (sizeof(srcType) == 1) {
            CopyGmToUbufAlignV2((__ubuf__ uint8_t*)(dst.Data().Get()), (__gm__ uint8_t*)(src.Data().Get()), params...);
        } else if constexpr (sizeof(srcType) == 2) {
            CopyGmToUbufAlignV2((__ubuf__ uint16_t*)(dst.Data().Get()), (__gm__ uint16_t*)(src.Data().Get()),
                                params...);
        } else if constexpr (sizeof(srcType) == 4) {
            CopyGmToUbufAlignV2((__ubuf__ uint32_t*)(dst.Data().Get()), (__gm__ uint32_t*)(src.Data().Get()),
                                params...);
        } else if constexpr (sizeof(srcType) == 8) {
            CopyGmToUbufAlignV2((__ubuf__ uint32_t*)(dst.Data().Get()), (__gm__ uint32_t*)(src.Data().Get()),
                                params...);
        }
    }

private:
    template <typename T>
    __aicore__ inline static void CopyGmToUbufAlignV2(__ubuf__ T* dst, __gm__ T* src, const uint16_t blockCount,
                                                      const uint32_t blockLen, const uint8_t leftPaddingCount,
                                                      const uint8_t rightPaddingCount, const int64_t srcStride,
                                                      const int64_t dstStride, const uint8_t cacheMode = 0, const bool isPad = true)
    {
        if ASCEND_IS_AIC {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            copy_gm_to_ubuf_align_v2(dst, src, 0, blockCount, blockLen, leftPaddingCount, rightPaddingCount, isPad,
                                     cacheMode, srcStride, dstStride);
        }
    }
};

class CopyUbufToGmAlignV2Instr {
public:
    template <typename T, typename U, typename... Params>
    __aicore__ inline static void DataCopy(const T& dst, const U& src, const Params&... params)
    { CopyUbufToGmAlignV2(dst.Data().Get(), src.Data().Get(), params...); }

    template <typename T>
    __aicore__ inline static void CopyUbufToGmAlignV2(__gm__ T* dst, __ubuf__ T* src, const uint16_t blockCount,
                                                      const uint32_t blockLen, const int64_t srcStride,
                                                      const int64_t dstStride, const uint8_t cacheMode = 0)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            copy_ubuf_to_gm_align_v2((__gm__ void*)dst, (__ubuf__ void*)src, 0, blockCount, blockLen, cacheMode,
                                     dstStride, srcStride);
        }
    }
};

class CopyUbufToCbufInstr {
public:
    template <typename T, typename U, typename... Params>
    __aicore__ inline static void DataCopy(const T& dst, const U& src, const Params&... params)
    { CopyUbufToCbuf(dst.Data().Get(), src.Data().Get(), params...); }

    template <typename T>
    __aicore__ inline static void CopyUbufToCbuf(__cbuf__ T* dst, __ubuf__ T* src, const uint16_t blockCount,
                                                 const uint16_t blockLen, const uint16_t srcStride,
                                                 const uint16_t dstStride)
    {
        if ASCEND_IS_AIC {
            return;
        }

        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            copy_ubuf_to_cbuf((__cbuf__ void*)dst, (__ubuf__ void*)src, 0, blockCount, blockLen, srcStride, dstStride);
        }
    }
};


} // namespace Te
} // namespace AscendC

#endif

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
