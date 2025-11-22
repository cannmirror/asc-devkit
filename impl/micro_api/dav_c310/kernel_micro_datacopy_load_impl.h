/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file kernel_micro_datacopy_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_DATACOPY_LOAD_IMPL_H
#define ASCENDC_MODULE_MICRO_DATACOPY_LOAD_IMPL_H

namespace AscendC {
namespace MicroAPI {
template <int OutputNum, LoadDist dist> __simd_callee__ inline void CheckLoadDist()
{
    if constexpr (OutputNum == 1) {
        static_assert(SupportEnum<dist, LoadDist::DIST_NORM, LoadDist::DIST_BRC_B8, LoadDist::DIST_BRC_B16,
            LoadDist::DIST_BRC_B32, LoadDist::DIST_US_B8, LoadDist::DIST_US_B16, LoadDist::DIST_DS_B8,
            LoadDist::DIST_DS_B16, LoadDist::DIST_UNPACK_B8, LoadDist::DIST_UNPACK_B16, LoadDist::DIST_BLK,
            LoadDist::DIST_E2B_B16, LoadDist::DIST_E2B_B32, LoadDist::DIST_UNPACK_B32, LoadDist::DIST_UNPACK4_B8>(),
            "DataCopy not support this dist on current device");
    } else {
        static_assert(SupportEnum<dist, LoadDist::DIST_DINTLV_B8, LoadDist::DIST_DINTLV_B16,
            LoadDist::DIST_DINTLV_B32>(),
            "DataCopy not support this dist on current device");
    }
}

// vlds norm
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__simd_callee__ inline void DataCopyImpl(RegT &dstReg, __ubuf__ T *srcUbAddr)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    CheckLoadDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();

    if constexpr (SupportType<ActualT, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
        int4x2_t>()) {
        vlds((RegTensor<uint8_t> &)dstReg, (__ubuf__ uint8_t *)srcUbAddr, 0, distValue);
    } else if constexpr (SupportBytes<ActualT, 8>()) {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            vlds((RegTensor<uint32_t> &)dstReg, (__ubuf__ uint32_t *)srcUbAddr, 0, distValue);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            constexpr auto dintlvDist =
                std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_DINTLV_B32)>();
            vlds((RegTensor<uint32_t> &)dstReg.reg[0], (RegTensor<uint32_t> &)dstReg.reg[1],
                (__ubuf__ uint32_t *)srcUbAddr, 0, dintlvDist);
        }
    } else {
        if constexpr(SupportType<ActualT, complex32>() && (CheckRegTrait<RegT, RegTraitNumTwo>())) {
            constexpr auto dintlvDist =
                std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_DINTLV_B16)>();
            vlds((RegTensor<uint16_t> &)dstReg.reg[0], (RegTensor<uint16_t> &)dstReg.reg[1],
                (__ubuf__ uint16_t *)srcUbAddr, 0, dintlvDist);
        } else {
            static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
                "DataCopy only support type b8/b16/b32/b64 on current device");
            if constexpr (std::is_same_v<T, bool>) {
                vlds((RegTensor<int8_t> &)dstReg, (__ubuf__ int8_t *)srcUbAddr, 0, distValue);
            } else if constexpr (std::is_same_v<T, complex32>) {
                vlds((RegTensor<int32_t> &)dstReg, (__ubuf__ int32_t *)srcUbAddr, 0, distValue);
            } else {
                vlds(dstReg, srcUbAddr, 0, distValue);
            }
        }
    }
}

// vlds postupdate
template <typename T = DefaultType, PostLiteral postMode, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__simd_callee__ inline void DataCopyImpl(RegT &dstReg, __ubuf__ T *&srcUbAddr, int32_t postUpdateStride)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    CheckLoadDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();

    if constexpr (SupportType<ActualT, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
        int4x2_t>()) {
        vlds((RegTensor<uint8_t> &)dstReg, (__ubuf__ uint8_t *&)srcUbAddr, postUpdateStride, distValue, postValue);
    } else if constexpr (SupportBytes<ActualT, 8>()) {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            vlds((RegTensor<uint32_t> &)dstReg, (__ubuf__ uint32_t *&)srcUbAddr, postUpdateStride * 2, distValue,
                postValue);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            constexpr auto dintlvDist =
                std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_DINTLV_B32)>();
            vlds((RegTensor<uint32_t> &)dstReg.reg[0], (RegTensor<uint32_t> &)dstReg.reg[1],
                (__ubuf__ uint32_t *&)srcUbAddr, postUpdateStride * 2, dintlvDist, postValue);
        }
    } else {
        if constexpr(SupportType<ActualT, complex32>() && (CheckRegTrait<RegT, RegTraitNumTwo>())) {
            constexpr auto dintlvDist =
                std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_DINTLV_B16)>();
            vlds((RegTensor<uint16_t> &)dstReg.reg[0], (RegTensor<uint16_t> &)dstReg.reg[1],
                (__ubuf__ uint16_t *&)srcUbAddr, postUpdateStride * 2, dintlvDist, postValue);
        } else {
            static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
                "DataCopy only support type b8/b16/b32/b64 on current device");
            if constexpr (std::is_same_v<T, bool>) {
                vlds((RegTensor<int8_t> &)dstReg, (__ubuf__ int8_t *&)srcUbAddr, postUpdateStride, distValue, postValue);
            } else if constexpr (SupportBytes<ActualT, 4>()) {
                vlds((RegTensor<int32_t> &)dstReg, (__ubuf__ int32_t *&)srcUbAddr, postUpdateStride, distValue, postValue);
            } else {
                vlds(dstReg, srcUbAddr, postUpdateStride, distValue, postValue);
            }
        }
    }
}

// vld areg
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__simd_callee__ inline void DataCopyImpl(RegT &dstReg, __ubuf__ T *srcUbAddr, AddrReg offset)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckLoadDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    if constexpr (SupportType<ActualT, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
        int4x2_t>()) {
        vld((RegTensor<uint8_t> &)dstReg, (__ubuf__ uint8_t *)srcUbAddr, offset, distValue);
    } else {
        static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
            "DataCopy only support type b8/b16/b32/b64 on current device");
        if constexpr (std::is_same_v<T, bool>) {
            vld((RegTensor<int8_t> &)dstReg, (__ubuf__ int8_t *)srcUbAddr, offset, distValue);
        } else if constexpr (SupportBytes<ActualT, 4>()) {
            vld((RegTensor<int32_t> &)dstReg, (__ubuf__ int32_t *)srcUbAddr, offset, distValue);
        } else if constexpr (SupportBytes<ActualT, 8>()) {
            vld((RegTensor<int32_t> &)dstReg, (__ubuf__ int32_t *)srcUbAddr, offset, distValue);
        } else {
            vld(dstReg, srcUbAddr, offset, distValue);
        }
    }
}

// vlds dual norm
template <typename T = DefaultType, LoadDist dist, typename RegT>
__simd_callee__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __ubuf__ T *srcUbAddr)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckLoadDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    if constexpr (SupportType<ActualT, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
        int4x2_t>()) {
        vlds((RegTensor<uint8_t> &)dstReg0, (RegTensor<uint8_t> &)dstReg1, (__ubuf__ uint8_t *)srcUbAddr, 0, distValue);
    } else {
        static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
            "DataCopy only support type b8/b16/b32/b64 on current device");
        if constexpr (std::is_same_v<T, bool>) {
            vlds((RegTensor<int8_t> &)dstReg0, (RegTensor<int8_t> &)dstReg1, (__ubuf__ int8_t *)srcUbAddr, 0, distValue);
        } else if constexpr (SupportBytes<ActualT, 4>()) {
            vlds((RegTensor<int32_t> &)dstReg0, (RegTensor<int32_t> &)dstReg1, (__ubuf__ int32_t *)srcUbAddr, 0, distValue);
        } else if constexpr (SupportBytes<ActualT, 8>()) {
            vlds((RegTensor<int64_t> &)dstReg0, (RegTensor<int64_t> &)dstReg1, (__ubuf__ int64_t *)srcUbAddr, 0, distValue);
        } else {
            vlds(dstReg0, dstReg1, srcUbAddr, 0, distValue);
        }
    }
}

// vlds dual postupdate
template <typename T = DefaultType, PostLiteral postMode, LoadDist dist, typename RegT>
__simd_callee__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __ubuf__ T *&srcUbAddr, int32_t postUpdateStride)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckLoadDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    if constexpr (SupportType<ActualT, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
        int4x2_t>()) {
        vlds((RegTensor<uint8_t> &)dstReg0, (__ubuf__ uint8_t *&)dstReg1, srcUbAddr, postUpdateStride, distValue,
            postValue);
    } else {
        static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
            "DataCopy only support type b8/b16/b32/b64 on current device");
        if constexpr (std::is_same_v<T, bool>) {
            vlds((RegTensor<int8_t> &)dstReg0, (RegTensor<int8_t> &)dstReg1,
                (__ubuf__ int8_t *&)srcUbAddr, postUpdateStride, distValue, postValue);
        } else if constexpr (SupportBytes<ActualT, 4>()) {
            vlds((RegTensor<int32_t> &)dstReg0, (RegTensor<int32_t> &)dstReg1, (__ubuf__ int32_t *&)srcUbAddr,
                postUpdateStride, distValue, postValue);
        } else if constexpr (SupportBytes<ActualT, 8>()) {
            vlds((RegTensor<int64_t> &)dstReg0, (RegTensor<int64_t> &)dstReg1, (__ubuf__ int64_t *&)srcUbAddr,
                postUpdateStride, distValue, postValue);
        } else {
            vlds(dstReg0, dstReg1, srcUbAddr, postUpdateStride, distValue, postValue);
        }
    }
}

// vlds dual areg
template <typename T = DefaultType, LoadDist dist, typename RegT>
__simd_callee__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __ubuf__ T *srcUbAddr, AddrReg offset)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckLoadDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    if constexpr (SupportType<ActualT, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
        int4x2_t>()) {
        vld((RegTensor<uint8_t> &)dstReg0, (RegTensor<uint8_t> &)dstReg1, (__ubuf__ uint8_t *)srcUbAddr, offset,
            distValue);
    } else {
        static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
            "DataCopy only support type b8/b16/b32/b64 on current device");
        if constexpr (std::is_same_v<T, bool>) {
            vld((RegTensor<int8_t> &)dstReg0, (RegTensor<int8_t> &)dstReg1, (__ubuf__ int8_t *)srcUbAddr, offset, distValue);
        } else if constexpr (SupportBytes<ActualT, 4>()) {
            vld((RegTensor<int32_t> &)dstReg0, (RegTensor<int32_t> &)dstReg1, (__ubuf__ int32_t *)srcUbAddr, offset, distValue);
        } else if constexpr (SupportBytes<ActualT, 8>()) {
            vld((RegTensor<int32_t> &)dstReg0, (RegTensor<int32_t> &)dstReg1, (__ubuf__ int32_t *)srcUbAddr, offset, distValue);
        } else {
            vld(dstReg0, dstReg1, srcUbAddr, offset, distValue);
        }
    }
}

// vldas/vldus
template <typename T> __simd_callee__ inline void DataCopyUnAlignPreImpl(UnalignReg &ureg, __ubuf__ T *srcUbAddr)
{
    static_assert(SupportBytes<T, 1, 2, 4, 8>(),
        "DataCopyUnAlignPre only support type b8/b16/b32/b64 on current device");
    if constexpr (sizeof(T) == 8) {
        vldas(ureg, (__ubuf__ uint32_t *&)srcUbAddr);
    } else {
        if constexpr (std::is_same_v<T, bool>) {
            vldas(ureg, (__ubuf__ int8_t *)srcUbAddr);
        } else if constexpr (SupportBytes<T, 4>()) {
            vldas(ureg, (__ubuf__ int32_t *)srcUbAddr);
        } else {
            vldas(ureg, srcUbAddr);
        }
    }
}

template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename RegT>
__simd_callee__ inline void DataCopyUnAlignImpl(RegT &dstReg, UnalignReg &ureg, __ubuf__ T *&srcUbAddr, uint32_t stride)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportEnum<postMode, PostLiteral::POST_MODE_UPDATE>(),
        "DataCopyUnAlign only support update mode when load from local memory!");
    static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
        "DataCopyUnAlign only support type b8/b16/b32/b64 on current device");
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    if constexpr (SupportBytes<ActualT, 8>()) {
        if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            vldus((RegTensor<uint32_t> &)dstReg, ureg, (__ubuf__ uint32_t *&)srcUbAddr, stride * 2, postValue);
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            RegTensor<uint32_t> tmp1;
            RegTensor<uint32_t> tmp2;
            constexpr uint32_t one_repeat_num = VECTOR_REG_WIDTH / sizeof(ActualT);
            uint32_t tmpStride1 = (stride > one_repeat_num) ? one_repeat_num : stride;
            vldus(tmp1, ureg, (__ubuf__ uint32_t *&)srcUbAddr, tmpStride1 * 2, postValue);
            uint32_t tmpStride2 = (stride > one_repeat_num) ? stride - one_repeat_num : 0;
            vldus(tmp2, ureg, (__ubuf__ uint32_t *&)srcUbAddr, tmpStride2 * 2, postValue);
            DeInterleave((RegTensor<uint32_t> &)dstReg.reg[0], (RegTensor<uint32_t> &)dstReg.reg[1], tmp1, tmp2);
        }
    } else {
        if constexpr(SupportType<ActualT, complex32>() && (CheckRegTrait<RegT, RegTraitNumTwo>())) {
            RegTensor<uint16_t> tmp1;
            RegTensor<uint16_t> tmp2;
            constexpr uint32_t one_repeat_num = VECTOR_REG_WIDTH / sizeof(ActualT);
            uint32_t tmpStride1 = (stride > one_repeat_num) ? one_repeat_num : stride;
            vldus(tmp1, ureg, (__ubuf__ uint16_t *&)srcUbAddr, tmpStride1 * 2, postValue);
            uint32_t tmpStride2 = (stride > one_repeat_num) ? stride - one_repeat_num : 0;
            vldus(tmp2, ureg, (__ubuf__ uint16_t *&)srcUbAddr, tmpStride2 * 2, postValue);
            DeInterleave((RegTensor<uint16_t> &)dstReg.reg[0], (RegTensor<uint16_t> &)dstReg.reg[1], tmp1, tmp2);
        } else {
            if constexpr (std::is_same_v<T, bool>) {
                vldus((RegTensor<int8_t> &)dstReg, ureg, (__ubuf__ int8_t *&)srcUbAddr, stride, postValue);
            } else if constexpr (SupportBytes<ActualT, 4>()) {
                vldus((RegTensor<int32_t> &)dstReg, ureg, (__ubuf__ int32_t *&)srcUbAddr, stride, postValue);
            } else {
                vldus(dstReg, ureg, srcUbAddr, stride, postValue);
            }
        }
    }
}

template <typename T = DefaultType, typename RegT>
__simd_callee__ inline void DataCopyUnAlignImpl(RegT &dstReg, UnalignReg &ureg, __ubuf__ T *srcUbAddr)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
        "DataCopyUnAlign only support type b8/b16/b32/b64 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vldus((RegTensor<int8_t> &)dstReg, ureg, (__ubuf__ int8_t *)srcUbAddr);
    } else if constexpr (SupportBytes<T, 8>()) {
        vldus((RegTensor<int64_t> &)dstReg, ureg, (__ubuf__ int64_t *)srcUbAddr);
    } else if constexpr (SupportBytes<T, 4>()) {
        vldus((RegTensor<int32_t> &)dstReg, ureg, (__ubuf__ int32_t *)srcUbAddr);
    } else {
        vldus(dstReg, ureg, srcUbAddr);
    }
}

// vlda/vldu
template <typename T>
__simd_callee__ inline void DataCopyUnAlignPreImpl(UnalignReg &ureg, __ubuf__ T *srcUbAddr, AddrReg &areg)
{
    static_assert(SupportBytes<T, 1, 2, 4, 8>(),
        "DataCopyUnAlignPre only support type b8/b16/b32/b64 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vlda(ureg, (__ubuf__ int8_t *)srcUbAddr, areg);
    } else if constexpr (SupportBytes<T, 8>()) {
        vlda(ureg, (__ubuf__ int32_t *)srcUbAddr, areg);
    } else if constexpr (SupportBytes<T, 4>()) {
        vlda(ureg, (__ubuf__ int32_t *)srcUbAddr, areg);
    } else {
        vlda(ureg, srcUbAddr, areg);
    }
}

template <typename T = DefaultType, typename RegT>
__simd_callee__ inline void DataCopyUnAlignImpl(RegT &dstReg, UnalignReg &ureg, __ubuf__ T *&srcUbAddr, AddrReg &areg,
    uint32_t inc)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(SupportBytes<ActualT, 1, 2, 4, 8>(),
        "DataCopyUnAlign only support type b8/b16/b32/b64 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vldu((RegTensor<int8_t> &)dstReg, ureg, areg, (__ubuf__ int8_t *&)srcUbAddr, inc);
    } else if constexpr (SupportBytes<T, 8>()) {
        vldu((RegTensor<int32_t> &)dstReg, ureg, areg, (__ubuf__ int32_t *&)srcUbAddr, inc);
    } else if constexpr (SupportBytes<T, 4>()) {
        vldu((RegTensor<int32_t> &)dstReg, ureg, areg, (__ubuf__ int32_t *&)srcUbAddr, inc);
    } else {
        vldu(dstReg, ureg, areg, srcUbAddr, inc);
    }
}

// vsldb
template <typename T = DefaultType, DataCopyMode dataMode, typename RegT>
__simd_callee__ inline void DataCopyImpl(RegT &dstReg, __ubuf__ T *srcUbAddr, uint32_t dataBlockStride, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    if constexpr (SupportType<ActualT, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
        int4x2_t>()) {
        vsldb((RegTensor<uint8_t> &)dstReg, (__ubuf__ uint8_t *)srcUbAddr, (dataBlockStride << 16u), mask);
    } else {
        static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
        if constexpr (std::is_same_v<T, bool>) {
            vsldb((RegTensor<int8_t> &)dstReg, (__ubuf__ int8_t *)srcUbAddr, (dataBlockStride << 16u), mask);
        } else if constexpr (std::is_same_v<T, complex32>) {
            vsldb((RegTensor<int32_t> &)dstReg, (__ubuf__ int32_t *)srcUbAddr, (dataBlockStride << 16u), mask);
        } else {
            vsldb(dstReg, srcUbAddr, (dataBlockStride << 16u), mask);
        }
    }
}

template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename RegT>
__simd_callee__ inline void DataCopyImpl(RegT &dstReg, __ubuf__ T *&srcUbAddr, uint32_t dataBlockStride,
    uint32_t repeatStride, MaskReg &mask)
{
    if constexpr (postMode == PostLiteral::POST_MODE_NORMAL) {
        DataCopyImpl<T, dataMode, RegT>(dstReg, srcUbAddr, dataBlockStride, mask);
    }  else {
        using ActualT = typename RegT::ActualT;
        static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
        static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
        if constexpr (SupportType<ActualT, fp4x2_e2m1_t, fp4x2_e1m2_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t,
            int4x2_t>()) {
            constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
            vsldb((RegTensor<uint8_t> &)dstReg, (__ubuf__ uint8_t *&)srcUbAddr,
                (dataBlockStride << 16u) | (repeatStride & 0xFFFFU), mask, postValue);
        } else {
            static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32");
            constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
            if constexpr (std::is_same_v<T, bool>) {
                vsldb((RegTensor<int8_t> &)dstReg, (__ubuf__ int8_t *&)srcUbAddr,
                    (dataBlockStride << 16u) | (repeatStride & 0xFFFFU), mask, postValue);
            } else if constexpr (std::is_same_v<T, complex32>) {
                vsldb((RegTensor<int32_t> &)dstReg, (__ubuf__ int32_t *&)srcUbAddr,
                    (dataBlockStride << 16u) | (repeatStride & 0xFFFFU), mask, postValue);
            } else {
                vsldb(dstReg, srcUbAddr, (dataBlockStride << 16u) | (repeatStride & 0xFFFFU), mask, postValue);
            }
        }
    }
}

} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_DATACOPY_LOAD_IMPL_H
