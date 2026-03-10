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
 * \file fixpipe_utils.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_UTILS_H
#define IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_UTILS_H

namespace AscendC {
namespace Te{

constexpr uint32_t MAIN_LOOP_N_SIZE_3510 = 512;
constexpr uint32_t CBURST_NUM_3510 = MAIN_LOOP_N_SIZE_3510 / BLOCK_CUBE;

template <typename T>
__aicore__ inline auto AllocTempBuf(const T& calNSize)
{
    uint64_t deqTensorTempBuf = 0;
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        deqTensorTempBuf = reinterpret_cast<uint64_t>(get_imm(0));
    }
    return deqTensorTempBuf;
}

template <typename T>
__aicore__ inline void SetFpc(const T& deqTensorTempBuf)
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 ||
                  CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        uint64_t deqTensorAddr = (reinterpret_cast<uint64_t>(deqTensorTempBuf) >> static_cast<uint64_t>(7)) << 8;
        set_fpc(deqTensorAddr);
    }
}

__aicore__ inline void InsertSync()
{
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510 || 
                    CURRENT_ARCH_VERSION == ArchVersion::V2201) {
        pipe_barrier(PIPE_FIX);
    }
}

class CopyDeqTensorToFbuf3510 {
public:
    template <typename T>
    __aicore__ inline void CopyDeqTensorToFbufImpl(const T& src, uint16_t calNSize, uint16_t nIterIndex)
    {
        auto dstAddr = reinterpret_cast<__fbuf__ uint64_t*>(AllocTempBuf(calNSize));
        auto dst = MakeTensor(MakeFixbufmemPtr(dstAddr), src.Layout());
        auto tileSrc = TileSrcTensor(src, calNSize, nIterIndex);
        DataCopy(dst, tileSrc);
        SetFpc(dstAddr);
    }
private:
    template <typename T>
    __aicore__ inline decltype(auto) TileSrcTensor(const T& src, uint16_t calNSize, uint16_t nIterIndex) {
        auto coord = MakeCoord(MakeCoord(0, 0), MakeCoord(0, nIterIndex * MAIN_LOOP_N_SIZE_3510));
        auto shape = MakeShape(1, calNSize);
        return src(coord, shape);
    }
};


class FormatCheckUtils3510 {
public:
    template <typename T>
    __aicore__ inline constexpr void CheckL0CNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Shape->Column->ZeroDim, is not Std::Int<16> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Stride->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "Fixpipe Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T>
    __aicore__ inline constexpr void CheckNDTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>,
            "Fixpipe Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
            "Fixpipe Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        using StrideColumn1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 1>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>,
            "Fixpipe Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
            "Fixpipe Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideColumn1, Std::Int<1>>,
            "Fixpipe Layout->Stride->Column->OneDim, is not Std::Int<1> type!");
    }

    template <typename T>
    __aicore__ inline constexpr void CheckNZTemplate()
    {
        using type = typename T::elementType;
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<FRACTAL_FIXED>>,
            "Fixpipe Layout->Shape->Row->ZeroDim, is not Std::Int<16> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<C0_SIZE / sizeof(type)>>,
            "Fixpipe Layout->Shape->Column->ZeroDim, is not Std::Int<C0Size/Type> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<C0_SIZE / sizeof(type)>>,
            "Fixpipe Layout->Stride->Row->ZeroDim, is not Std::Int<C0Size/Type> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<1>>,
            "Fixpipe Layout->Stride->Column->ZeroDim, is not Std::Int<1> type!");
    }

    template <typename T>
    __aicore__ inline constexpr void CheckDNTemplate()
    {
        using ShapeRow0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::ROW, 0>::type;
        using ShapeColumn0 = typename GetFourDimType<T, AttrInfo::SHAPE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<ShapeRow0, Std::Int<1>>,
            "Fixpipe Src->Layout->Shape->Row->ZeroDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<ShapeColumn0, Std::Int<1>>,
            "Fixpipe Src->Layout->Shape->Column->ZeroDim, is not Std::Int<1> type!");

        using StrideRow0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 0>::type;
        using StrideRow1 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::ROW, 1>::type;
        using StrideColumn0 = typename GetFourDimType<T, AttrInfo::STRIDE, AttrInfo::COLUMN, 0>::type;
        static_assert(Std::is_same_v<StrideRow0, Std::Int<0>>,
            "Fixpipe Src->Layout->Stride->Row->ZeroDim, is not Std::Int<0> type!");
        static_assert(Std::is_same_v<StrideRow1, Std::Int<1>>,
            "Fixpipe Src->Layout->Stride->Row->OneDim, is not Std::Int<1> type!");
        static_assert(Std::is_same_v<StrideColumn0, Std::Int<0>>,
            "Fixpipe Src->Layout->Stride->Column->ZeroDim, is not Std::Int<0> type!");
    }
};


template <const FixpipeTrait& trait, typename T, typename U>
__aicore__ inline void CheckFixpipeQuantParams()
{
    using srcType = typename U::elementType;
    using dstType = typename T::elementType;
    using currentType = Std::tuple<srcType, dstType>;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    if constexpr (trait.quantPre == QuantMode_t::NoQuant) {
        using quantDataType1 = Std::tuple<__cc__ float, __gm__ float>;
        using quantDataType2 = Std::tuple<__cc__ int32_t, __gm__ int32_t>;
        static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::F322F16 || trait.quantPre == QuantMode_t::QF322F16_PRE
                         || trait.quantPre == QuantMode_t::VQF322F16_PRE) {
        using quantDataType = Std::tuple<__cc__ float, __gm__ half>;
        static_assert((Std::is_one_of_v<currentType, quantDataType>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::F322BF16 || trait.quantPre == QuantMode_t::QF322BF16_PRE
                         || trait.quantPre == QuantMode_t::VQF322BF16_PRE) {
        using quantDataType = Std::tuple<__cc__ float, __gm__ bfloat16_t>;
        static_assert((Std::is_one_of_v<currentType, quantDataType>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::DEQF16 || trait.quantPre == QuantMode_t::VDEQF16) {
        using quantDataType = Std::tuple<__cc__ int32_t, __gm__ half>;
        static_assert((Std::is_one_of_v<currentType, quantDataType>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::QF322B8_PRE || trait.quantPre == QuantMode_t::VQF322B8_PRE) {
        using quantDataType1 = Std::tuple<__cc__ float, __gm__ int8_t>;
        using quantDataType2 = Std::tuple<__cc__ float, __gm__ uint8_t>;
        static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::REQ8 || trait.quantPre == QuantMode_t::VREQ8) {
        using quantDataType1 = Std::tuple<__cc__ int32_t, __gm__ int8_t>;
        using quantDataType2 = Std::tuple<__cc__ int32_t, __gm__ uint8_t>;
        static_assert((Std::is_one_of_v<currentType, quantDataType1, quantDataType2>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::QF322FP8_PRE || trait.quantPre == QuantMode_t::VQF322FP8_PRE) {
        using quantDataType = Std::tuple<__cc__ float, __gm__ fp8_e4m3fn_t>;
        static_assert((Std::is_one_of_v<currentType, quantDataType>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::QF322HIF8_PRE || trait.quantPre == QuantMode_t::VQF322HIF8_PRE
                         || trait.quantPre == QuantMode_t::QF322HIF8_PRE_HYBRID
                         || trait.quantPre == QuantMode_t::VQF322HIF8_PRE_HYBRID) {
        using quantDataType = Std::tuple<__cc__ float, __gm__ hifloat8_t>;
        static_assert((Std::is_one_of_v<currentType, quantDataType>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::QS322BF16_PRE
                         || trait.quantPre == QuantMode_t::VQS322BF16_PRE) {
        using quantDataType = Std::tuple<__cc__ int32_t, __gm__ bfloat16_t>;
        static_assert((Std::is_one_of_v<currentType, quantDataType>), "Failed to check quantPre value in Fixpipe");
    } else if constexpr (trait.quantPre == QuantMode_t::QF322F32_PRE || trait.quantPre == QuantMode_t::VQF322F32_PRE) {
        using quantDataType = Std::tuple<__cc__ float, __gm__ float>;
        static_assert((Std::is_one_of_v<currentType, quantDataType>), "Failed to check quantPre value in Fixpipe");
    }

#endif
}
    
enum class Format3510 : uint8_t { None, NZ, ND, DN };
enum class QuantMode3510 : uint8_t { None, Scalar, Vector, Direct };

template <typename T>
__aicore__ inline constexpr Format3510 GetDataFormat()
{
    if constexpr (IsL0cNZFormat<T>::value) {
        return Format3510::NZ;
    } else if constexpr (IsNDFormat<T>::value) {
        return Format3510::ND;
    } else if constexpr (IsDNFormat<T>::value) {
        return Format3510::DN;
    }
    return Format3510::None;
}

template <const FixpipeTrait& trait>
__aicore__ inline constexpr QuantMode3510 GetQuantMode()
{
    if constexpr (IsVectorQuantMode<trait.quantPre>()) {
        return QuantMode3510::Vector;
    } else if constexpr (IsScalarQuantMode<trait.quantPre>()) {
        return QuantMode3510::Scalar;
    } else if constexpr (IsDirectQuantMode<trait.quantPre>()) {
        return QuantMode3510::Direct;
    }
    return QuantMode3510::None;
}

class FormatRegistorIgnore3510 {
public:
    template <const FixpipeTrait& trait, typename T, typename U, typename V, typename Coord>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant, const Coord& coord) {}
    template <const FixpipeTrait& trait, typename T, typename U, typename V>
    __aicore__ inline void Run(const T& dst, const U& src, const V& quant) {}
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_CUBE_DATAMOVE_FIXPIPE_FIXPIPE_UTILS_H