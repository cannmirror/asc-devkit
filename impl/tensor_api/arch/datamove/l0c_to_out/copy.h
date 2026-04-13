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
    "impl/tensor_api/arch/datamove/l0c_to_out/copy.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file copy.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_OUT_COPY_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_OUT_COPY_H

#include "impl/tensor_api/utils/utils_impl.h"
#include "impl/tensor_api/atom/copy_traits_impl.h"
#include "impl/tensor_api/arch/datamove/l0c_to_out/routing.h"

namespace AscendC {
namespace Te {

template <typename T, typename U>
static constexpr bool VerifyingFixpipeTemplate = (IsTileTensorV<T> && IsTileTensorV<U>);

template <typename T, typename U, typename S>
static constexpr bool VerifyingFixpipeQuantTemplate =
    (IsTileTensorV<T> && IsTileTensorV<U> && (IsTileTensorV<S> || Std::is_same_v<S, uint64_t>));

template <typename T, typename U, typename Coord>
constexpr bool VerifyingFixpipeTemplateWithCoord = Std::is_tuple_v<Coord> && VerifyingFixpipeTemplate<T, U>;

template <typename T, typename U, typename S, typename Coord>
constexpr bool VerifyingFixpipeQuantTemplateWithCoord =
    Std::is_tuple_v<Coord> && VerifyingFixpipeQuantTemplate<T, U, S>;

constexpr FixpipeTrait DEFAULT_FIXPIPE_TRAIT = FixpipeTrait{};

struct FixpipeTraitDefault {
    using TraitType = FixpipeTrait;
    static constexpr const TraitType value = DEFAULT_FIXPIPE_TRAIT;
};

struct CopyL0C2Out {
public:
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Copy(const Args&... args)
    {
        if ASCEND_IS_AIV {
            return;
        }
        FixpipeImpl<traits>(args...);
    }

private:
    template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U>
    __aicore__ inline static typename Std::enable_if<VerifyingFixpipeTemplate<T, U>, void>::type
    FixpipeImpl(const T& dst, const U& src, const FixpipeParams& params = FixpipeParams{})
    {
        constexpr Hardware dstPos = GetHardPos<T>();
        constexpr Hardware srcPos = GetHardPos<U>();
        constexpr Hardware quantPos = Hardware::MAX;
        using Tensor2Tensor = typename FixpipeTensor2Tensor<dstPos, srcPos, quantPos, CURRENT_ARCH_VERSION>::type;
        Tensor2Tensor{}.template Run<trait>(dst, src, params);
    }

    template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename S>
    __aicore__ inline static typename Std::enable_if<VerifyingFixpipeQuantTemplate<T, U, S>, void>::type
    FixpipeImpl(const T& dst, const U& src, const S& quant, const FixpipeParams& params = FixpipeParams{})
    {
        constexpr Hardware dstPos = GetHardPos<T>();
        constexpr Hardware srcPos = GetHardPos<U>();
        constexpr Hardware quantPos = Hardware::L1;
        using Tensor2Tensor = typename FixpipeTensor2Tensor<dstPos, srcPos, quantPos, CURRENT_ARCH_VERSION>::type;
        Tensor2Tensor{}.template Run<trait>(dst, src, quant, params);
    }

    template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename Coord>
    __aicore__ inline static typename Std::enable_if<VerifyingFixpipeTemplateWithCoord<T, U, Coord>, void>::type
    FixpipeImpl(const T& dst, const U& src, const Coord& coord, const FixpipeParams& params = FixpipeParams{})
    {
        auto sliceTensor = dst(coord, src);
        FixpipeImpl<trait>(sliceTensor, src, params);
    }

    template <const FixpipeTrait& trait = DEFAULT_FIXPIPE_TRAIT, typename T, typename U, typename S, typename Coord>
    __aicore__ inline static typename Std::enable_if<VerifyingFixpipeQuantTemplateWithCoord<T, U, S, Coord>, void>::type
    FixpipeImpl(const T& dst, const U& src, const S& quant, const Coord& coord,
                const FixpipeParams& params = FixpipeParams{})
    {
        auto sliceTensor = dst(coord, src);
        FixpipeImpl<trait>(sliceTensor, src, quant, params);
    }
};

struct CopyL0C2OutWith {
    template <typename Tp, const Tp& traits, typename... Args>
    __aicore__ inline static void Copy(const Args&... args)
    {
        // custom function Fixpipe<traits, Args...>(args...);
    }
};


using CopyL0C2GM = CopyL0C2Out;
using CopyL0C2UB = CopyL0C2Out;
using CopyL0C2GMWith = CopyL0C2OutWith;
using CopyL0C2UBWith = CopyL0C2OutWith;

using CopyL0C2GMTrait = FixpipeTrait;
using CopyL0C2UBTrait = FixpipeTrait;

using CopyL0C2GMTraitDefault = FixpipeTraitDefault;
using CopyL0C2UBTraitDefault = FixpipeTraitDefault;

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_OUT_COPY_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
