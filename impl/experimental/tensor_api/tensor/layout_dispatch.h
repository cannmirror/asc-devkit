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
* \file layout_dispatch.h
* \brief
*/
#ifndef IMPL_TENSOR_API_TENSOR_LAYOUT_DISPATCH_H
#define IMPL_TENSOR_API_TENSOR_LAYOUT_DISPATCH_H

#include "impl/experimental/tensor_api/utils/utils_impl.h"

namespace AscendC {
namespace Te {

__aicore__ inline int64_t CeilAlign(int64_t a, int64_t b) {
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

struct MakeTupleCons {
    template <typename... Ts>
    __aicore__ inline decltype(auto) operator()(Ts&&... ts) {
        return Std::make_tuple(Std::forward<Ts>(ts)...);
    }
};

template <typename F, typename T>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T&& t) {
    return t;
}

template <typename F, typename T0, typename T1>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1) {
    return f(t0, t1);
}

template <typename F, typename T0, typename T1, typename... Ts>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1, Ts&&... ts) {
    auto tuple1 = Make2Params2Tuple(f, t0, t1);
    auto tuple2 = Make2Params2Tuple(f, ts...);
    return Make2Params2Tuple(f, tuple1, tuple2);
}

template <typename T0, typename T1, typename T2, typename T3, typename... Ts>
__aicore__ inline decltype(auto) LayoutConstructor(T0&& t0, T1&& t1, T2&& t2, T3&& t3, Ts&&... ts) {
    auto shape = Make2Params2Tuple(MakeTupleCons{}, t0, t1, t2, t3);
    auto stride = Make2Params2Tuple(MakeTupleCons{}, ts...);
    return Layout(shape, stride);
}

// layout_dispatch.h
template <LayoutFormat format, typename T>
struct LayoutDispatcher;

template <typename T>
struct LayoutDispatcher<LayoutFormat::NZ, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        constexpr auto c0Size = is_b4_type<T> ? C0_SIZE * 2 : C0_SIZE / sizeof(T);
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED), Std::Int<c0Size>{},
                                 CeilDivision(column, (c0Size)), Std::Int<c0Size>{}, Std::Int<c0Size * FRACTAL_FIXED>{},
                                 Std::Int<1>{}, c0Size * CeilAlign(row, FRACTAL_FIXED));
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::NZ, Std::ignore_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{},  CeilDivision(row, FRACTAL_FIXED), 
                                Std::Int<C0_SIZE / sizeof(uint16_t)>{},  CeilDivision(column, (C0_SIZE / sizeof(uint16_t))), 
                                Std::Int<C0_SIZE / sizeof(uint16_t)>{},  Std::Int<C0_SIZE / sizeof(uint16_t) * FRACTAL_FIXED>{},
                                Std::Int<1>{},  C0_SIZE / sizeof(uint16_t) * CeilAlign(row, FRACTAL_FIXED)); 
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ZN, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        constexpr auto c0Size = is_b4_type<T> ? C0_SIZE * 2 : C0_SIZE / sizeof(T);
        return LayoutConstructor(Std::Int<c0Size>{},  CeilDivision(row, (c0Size)),
                                Std::Int<FRACTAL_FIXED>{},  CeilDivision(column, FRACTAL_FIXED),
                                Std::Int<1>{},  c0Size * CeilAlign(column, FRACTAL_FIXED),
                                Std::Int<c0Size>{},  Std::Int<c0Size * FRACTAL_FIXED>{});
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::DN, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<1>{}, column,
                                    Std::Int<0>{}, Std::Int<1>{}, Std::Int<0>{}, row);
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::DN, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<2>{}, column / MX_SCALE_K0,
                                    Std::Int<0>{}, Std::Int<MX_SCALE_K0>{}, Std::Int<1>{}, MX_SCALE_K0 * row);
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ND, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<1>{}, column,
                                    Std::Int<0>{}, column, Std::Int<0>{}, Std::Int<1>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::ND, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<2>{}, row / MX_SCALE_K0, Std::Int<1>{}, column,
                                    Std::Int<1>{}, MX_SCALE_K0 * column, Std::Int<0>{}, Std::Int<MX_SCALE_K0>{});
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ZZ, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<C0_SIZE / sizeof(T)>{}, CeilDivision(column, (C0_SIZE / sizeof(T))),
                                    Std::Int<C0_SIZE / sizeof(T)>{}, FRACTAL_FIXED * CeilAlign(column, (C0_SIZE / sizeof(T))),
                                    Std::Int<1>{}, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::ZZ, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<MX_SCALE_K0>{}, column / MX_SCALE_K0,
                                    Std::Int<MX_SCALE_K0>{}, column * FRACTAL_FIXED,
                                    Std::Int<1>{}, Std::Int<C0_SIZE>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::NN, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) { // (scaleK, n)
        return LayoutConstructor(Std::Int<MX_SCALE_K0>{}, row / MX_SCALE_K0,
                                    Std::Int<FRACTAL_FIXED>{}, CeilDivision(column, FRACTAL_FIXED),
                                    Std::Int<1>{}, Std::Int<C0_SIZE>{},
                                    Std::Int<MX_SCALE_K0>{}, row * FRACTAL_FIXED);
    }
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_LAYOUT_DISPTACH_H