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
    "impl/tensor_api/arch/datamove/l0c_to_gm/copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_GM_COPY_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_GM_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/atom/copy_traits_impl.h"
#include "impl/tensor_api/arch/datamove/common/l0c2out_utils.h"
#include "impl/tensor_api/arch/datamove/l0c_to_gm/routing.h"

namespace AscendC {
namespace Te {

struct CopyL0C2GM {
public:
    template <typename Tp, const Tp& trait, typename... Args>
    __aicore__ inline static void Copy(const Args&... args)
    {
        if ASCEND_IS_AIV {
            return;
        }
        DataCopyImpl<trait>(args...);
    }

private:
    template <const CopyL0C2GMTrait& trait = DEFAULT_COPY_L0C2GM_TRAIT, typename T, typename U>
    __aicore__ inline static void DataCopyImpl(const T& dst, const U& src,
                                               const FixpipeParams& params = DEFAULT_FIXPIPE_PARAMS)
    {
        using dstPos = GetMemLocation<T>;
        using srcPos = GetMemLocation<U>;
        using Tensor2Tensor = typename CopyL0C2GMTensor2Tensor<dstPos, srcPos, CURRENT_ARCH_VERSION>::type;
        Tensor2Tensor{}.template Run<trait>(dst, src, params);
    }

    template <const CopyL0C2GMTrait& trait = DEFAULT_COPY_L0C2GM_TRAIT, typename T, typename U, typename S>
    __aicore__ inline static typename Std::enable_if<Std::is_same_v<S, uint64_t>, void>::type
    DataCopyImpl(const T& dst, const U& src, const S& quant, const FixpipeParams& params = DEFAULT_FIXPIPE_PARAMS)
    {
        using dstPos = GetMemLocation<T>;
        using srcPos = GetMemLocation<U>;
        using Tensor2Tensor = typename CopyL0C2GMTensor2Tensor<dstPos, srcPos, CURRENT_ARCH_VERSION>::type;
        Tensor2Tensor{}.template Run<trait>(dst, src, quant, params);
    }

    template <const CopyL0C2GMTrait& trait = DEFAULT_COPY_L0C2GM_TRAIT, typename T, typename U, typename S>
    __aicore__ inline static typename Std::enable_if<IsAttrTensorV<S>, void>::type
    DataCopyImpl(const T& dst, const U& src, const S& quant, const FixpipeParams& params = DEFAULT_FIXPIPE_PARAMS)
    {
        using dstPos = GetMemLocation<T>;
        using srcPos = GetMemLocation<U>;
        using Tensor2Tensor = typename CopyL0C2GMVectorQuantTensor2Tensor<dstPos, srcPos, CURRENT_ARCH_VERSION>::type;
        Tensor2Tensor{}.template Run<trait>(dst, src, quant, params);
    }

    template <const CopyL0C2GMTrait& trait = DEFAULT_COPY_L0C2GM_TRAIT, typename T, typename U, typename Coord>
    __aicore__ inline static typename Std::enable_if<Std::is_tuple_v<Coord>, void>::type
    DataCopyImpl(const T& dst, const U& src, const Coord& coord, const FixpipeParams& params = DEFAULT_FIXPIPE_PARAMS)
    {
        auto sliceTensor = dst(coord, src);
        DataCopyImpl<trait>(sliceTensor, src, params);
    }

    template <const CopyL0C2GMTrait& trait = DEFAULT_COPY_L0C2GM_TRAIT, typename T, typename U, typename S,
              typename Coord>
    __aicore__ inline static typename Std::enable_if<(Std::is_same_v<S, uint64_t> || IsAttrTensorV<S>) && Std::is_tuple_v<Coord>,
                                            void>::type
    DataCopyImpl(const T& dst, const U& src, const S& quant, const Coord& coord,
                 const FixpipeParams& params = DEFAULT_FIXPIPE_PARAMS)
    {
        auto sliceTensor = dst(coord, src);
        DataCopyImpl<trait>(sliceTensor, src, quant, params);
    }
};

struct CopyL0C2GMWith {
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Copy(const Args&... args)
    {
        // custom function Fixpipe<traits, Args...>(args...);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_GM_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

