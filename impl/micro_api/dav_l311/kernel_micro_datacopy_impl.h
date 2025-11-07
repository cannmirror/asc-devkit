/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_micro_datacopy_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_DATACOPY_IMPL_H
#define ASCENDC_MODULE_MICRO_DATACOPY_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {
template <int OutputNum, LoadDist dist>
__aicore__ inline void CheckLoadDist()
{
    if constexpr (OutputNum == 1) {
        static_assert(SupportEnum<dist,
                          LoadDist::DIST_NORM,
                          LoadDist::DIST_BRC_B8,
                          LoadDist::DIST_BRC_B16,
                          LoadDist::DIST_BRC_B32,
                          LoadDist::DIST_US_B8,
                          LoadDist::DIST_US_B16,
                          LoadDist::DIST_DS_B8,
                          LoadDist::DIST_DS_B16,
                          LoadDist::DIST_UNPACK_B8,
                          LoadDist::DIST_UNPACK_B16,
                          LoadDist::DIST_BLK,
                          LoadDist::DIST_E2B_B16,
                          LoadDist::DIST_E2B_B32,
                          LoadDist::DIST_UNPACK_B32,
                          LoadDist::DIST_UNPACK4_B8,
                          LoadDist::DIST_SPLT4CHN_B8,
                          LoadDist::DIST_SPLT2CHN_B8,
                          LoadDist::DIST_SPLT2CHN_B16>(),
            "DataCopy not support this dist on current device");
    } else {
        static_assert(SupportEnum<dist,
                          LoadDist::DIST_BDINTLV,
                          LoadDist::DIST_DINTLV_B8,
                          LoadDist::DIST_DINTLV_B16,
                          LoadDist::DIST_DINTLV_B32>(),
            "DataCopy not support this dist on current device");
    }
}

template <int InputNum, StoreDist dist>
__aicore__ inline void CheckStoreDist()
{
    if constexpr (InputNum == 1) {
        static_assert(SupportEnum<dist,
                          StoreDist::DIST_NORM_B8,
                          StoreDist::DIST_NORM_B16,
                          StoreDist::DIST_NORM_B32,
                          StoreDist::DIST_FIRST_ELEMENT_B8,
                          StoreDist::DIST_FIRST_ELEMENT_B16,
                          StoreDist::DIST_FIRST_ELEMENT_B32,
                          StoreDist::DIST_PACK_B16,
                          StoreDist::DIST_PACK_B32,
                          StoreDist::DIST_PACK_B64,
                          StoreDist::DIST_PACK4_B32,
                          StoreDist::DIST_MRG4CHN_B8,
                          StoreDist::DIST_MRG2CHN_B8,
                          StoreDist::DIST_MRG2CHN_B16,
                          StoreDist::DIST_NORM>(),
            "DataCopy not support this dist on current device");
    } else {
        static_assert(
            SupportEnum<dist, StoreDist::DIST_INTLV_B8, StoreDist::DIST_INTLV_B16, StoreDist::DIST_INTLV_B32>(),
            "DataCopy not support this dist on current device");
    }
}

// vlds norm
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg, __local_mem__ T *srcUbAddr)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    CheckLoadDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();

    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (sizeof(T) == 8) {
        vlds(dstReg, srcUbAddr, 0);  // use default Dist::DIST_DINTLV_B32
    } else {
        vlds(dstReg, srcUbAddr, 0, distValue);
    }
}

// vlds postupdate
template <typename T = DefaultType, PostLiteral postMode, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg, __local_mem__ T *&srcUbAddr, int32_t postUpdateStride)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    CheckLoadDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();

    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vlds((RegTensor<int8_t> &)dstReg, (__ubuf__ int8_t *&)srcUbAddr, postUpdateStride, distValue, postValue);
    } else {
        vlds(dstReg, srcUbAddr, postUpdateStride, distValue, postValue);
    }
}

// vld areg
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg, __local_mem__ T *srcUbAddr, AddrReg offset)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckLoadDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vld((RegTensor<int8_t> &)dstReg, (__ubuf__ int8_t *)srcUbAddr, offset, distValue);
    } else {
        vld(dstReg, srcUbAddr, offset, distValue);
    }
}

// vlds dual norm
template <typename T = DefaultType, LoadDist dist, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __local_mem__ T *srcUbAddr)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckLoadDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vlds((RegTensor<int8_t> &)dstReg0, (RegTensor<int8_t> &)dstReg1, (__ubuf__ int8_t *)srcUbAddr, 0, distValue);
    } else {
        vlds(dstReg0, dstReg1, srcUbAddr, 0, distValue);
    }
}

// vlds dual postupdate
template <typename T = DefaultType, PostLiteral postMode, LoadDist dist, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __local_mem__ T *&srcUbAddr, int32_t postUpdateStride)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckLoadDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vlds((RegTensor<int8_t> &)dstReg0,
            (RegTensor<int8_t> &)dstReg1,
            (__ubuf__ int8_t *&)srcUbAddr,
            postUpdateStride,
            distValue,
            postValue);
    } else {
        vlds(dstReg0, dstReg1, srcUbAddr, postUpdateStride, distValue, postValue);
    }
}

// vlds dual areg
template <typename T = DefaultType, LoadDist dist, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __local_mem__ T *srcUbAddr, AddrReg offset)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckLoadDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vld((RegTensor<int8_t> &)dstReg0,
            (RegTensor<int8_t> &)dstReg1,
            (__ubuf__ int8_t *)srcUbAddr,
            offset,
            distValue);
    } else {
        vld(dstReg0, dstReg1, srcUbAddr, offset, distValue);
    }
}

// vsts
template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>() || CheckRegTrait<RegT, RegTraitNumTwo>(),
        "RegTensor only suppoort RegTraitNumOne or RegTraitNumTwo on current device!");
    CheckStoreDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vsts((RegTensor<int8_t> &)srcReg, (__ubuf__ int8_t *)dstUbAddr, 0, distValue, mask);
    } else {
        vsts(srcReg, dstUbAddr, 0, distValue, mask);
    }
}

// vsts postupdate
template <typename T = DefaultType, PostLiteral postMode, StoreDist dist = StoreDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *&dstUbAddr, RegT &srcReg, int32_t postUpdateStride, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    CheckStoreDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vsts((RegTensor<int8_t> &)srcReg, (__ubuf__ int8_t *&)dstUbAddr, postUpdateStride, distValue, mask, postValue);
    } else {
        vsts(srcReg, dstUbAddr, postUpdateStride, distValue, mask, postValue);
    }
}

// vst areg
template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, RegT &srcReg, AddrReg offset, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckStoreDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vst((RegTensor<int8_t> &)srcReg, (__ubuf__ int8_t *)dstUbAddr, offset, distValue, mask);
    } else {
        vst(srcReg, dstUbAddr, offset, distValue, mask);
    }
}

// vsts dual
template <typename T = DefaultType, StoreDist dist, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckStoreDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vsts((RegTensor<int8_t> &)srcReg0,
            (RegTensor<int8_t> &)srcReg1,
            (__ubuf__ int8_t *)dstUbAddr,
            0,
            distValue,
            mask);
    } else {
        vsts(srcReg0, srcReg1, dstUbAddr, 0, distValue, mask);
    }
}

// vsts dual areg
template <typename T = DefaultType, StoreDist dist, typename RegT>
__aicore__ inline void DataCopyImpl(
    __local_mem__ T *dstUbAddr, RegT &srcReg0, RegT &srcReg1, AddrReg offset, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    CheckStoreDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vst((RegTensor<int8_t> &)srcReg0,
            (RegTensor<int8_t> &)srcReg1,
            (__ubuf__ int8_t *)dstUbAddr,
            offset,
            distValue,
            mask);
    } else {
        vst(srcReg0, srcReg1, dstUbAddr, offset, distValue, mask);
    }
}

// vsldb
template <typename T = DefaultType, DataCopyMode dataMode, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg, __local_mem__ T *srcUbAddr, uint32_t dataBlockStride, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vsldb((RegTensor<int8_t> &)dstReg, (__ubuf__ int8_t *)srcUbAddr, (dataBlockStride << 16u), mask);
    } else {
        vsldb(dstReg, srcUbAddr, (dataBlockStride << 16u), mask);
    }
}

template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename RegT>
__aicore__ inline void DataCopyImpl(
    RegT &dstReg, __local_mem__ T *&srcUbAddr, uint32_t dataBlockStride, uint32_t repeatStride, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32");
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    if constexpr (std::is_same_v<T, bool>) {
        vsldb((RegTensor<int8_t> &)dstReg,
            (__ubuf__ int8_t *&)srcUbAddr,
            (dataBlockStride << 16u) | (repeatStride & 0xFFFFU),
            mask,
            postValue);
    } else {
        vsldb(dstReg, srcUbAddr, (dataBlockStride << 16u) | (repeatStride & 0xFFFFU), mask, postValue);
    }
}

// vsstb
template <typename T = DefaultType, DataCopyMode dataMode, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, RegT &srcReg, uint32_t dataBlockStride, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vsstb((RegTensor<int8_t> &)srcReg, (__ubuf__ int8_t *)dstUbAddr, (dataBlockStride << 16u), mask);
    } else {
        vsstb(srcReg, dstUbAddr, (dataBlockStride << 16u), mask);
    }
}

template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename RegT>
__aicore__ inline void DataCopyImpl(
    __local_mem__ T *&dstUbAddr, RegT &srcReg, uint32_t dataBlockStride, uint32_t repeatStride, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    if constexpr (std::is_same_v<T, bool>) {
        vsstb((RegTensor<int8_t> &)srcReg,
            (__ubuf__ int8_t *&)dstUbAddr,
            (dataBlockStride << 16u) | (repeatStride & 0xFFFFU),
            mask,
            postValue);
    } else {
        vsstb(srcReg, dstUbAddr, (dataBlockStride << 16u) | (repeatStride & 0xFFFFU), mask, postValue);
    }
}

// vldas/vldus
template <typename T>
__aicore__ inline void DataCopyUnAlignPreImpl(UnalignReg &ureg, __local_mem__ T *srcUbAddr)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopyUnAlignPre only support type b8/b16/b32 on current device");
    if constexpr (sizeof(T) == 8) {
        vldas(ureg, (__local_mem__ uint32_t *&)srcUbAddr);
    } else {
        if constexpr (std::is_same_v<T, bool>) {
            vldas(ureg, (__ubuf__ int8_t *)srcUbAddr);
        } else {
            vldas(ureg, srcUbAddr);
        }
    }
}

template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(RegT &dstReg, UnalignReg &ureg, __local_mem__ T *&srcUbAddr, uint32_t stride)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopyUnAlign only support type b8/b16/b32 on current device");
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    if constexpr (std::is_same_v<T, bool>) {
        vldus((RegTensor<int8_t> &)dstReg, ureg, (__ubuf__ int8_t *&)srcUbAddr, stride, postValue);
    } else {
        vldus(dstReg, ureg, srcUbAddr, stride, postValue);
    }
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(RegT &dstReg, UnalignReg &ureg, __local_mem__ T *srcUbAddr)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopyUnAlign only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vldus((RegTensor<int8_t> &)dstReg, ureg, (__ubuf__ int8_t *)srcUbAddr);
    } else {
        vldus(dstReg, ureg, srcUbAddr);
    }
}

// vlda/vldu
template <typename T>
__aicore__ inline void DataCopyUnAlignPreImpl(UnalignReg &ureg, __local_mem__ T *srcUbAddr, AddrReg &areg)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopyUnAlignPre only support type b8/b16/b32 on current device");
    vlda(ureg, srcUbAddr, areg);
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(
    RegT &dstReg, UnalignReg &ureg, __local_mem__ T *&srcUbAddr, AddrReg &areg, uint32_t inc)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopyUnAlign only support type b8/b16/b32 on current device");
    if constexpr (std::is_same_v<T, bool>) {
        vldu((RegTensor<int8_t> &)dstReg, ureg, areg, (__ubuf__ int8_t *&)srcUbAddr, inc);
    } else {
        vldu(dstReg, ureg, areg, srcUbAddr, inc);
    }
}

// vstus/vstas
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(
    __local_mem__ T *&dstUbAddr, RegT &srcReg, UnalignReg &ureg, uint32_t postUpdateStride)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopyUnAlign only support type b8/b16/b32 on current device");
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    if constexpr (std::is_same_v<T, bool>) {
        vstus(ureg, postUpdateStride, (RegTensor<int8_t> &)srcReg, (__ubuf__ int8_t *&)dstUbAddr, postValue);
    } else {
        vstus(ureg, postUpdateStride, srcReg, dstUbAddr, postValue);
    }
}

template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE>
__aicore__ inline void DataCopyUnAlignPostImpl(__local_mem__ T *&dstUbAddr, UnalignReg &ureg, int32_t postUpdateStride)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopyUnAlignPost only support type b8/b16/b32 on current device");
    if constexpr (postMode == PostLiteral::POST_MODE_UPDATE) {
        if constexpr (std::is_same_v<T, bool>) {
            vstas(ureg, (__ubuf__ int8_t *&)dstUbAddr, postUpdateStride, POST_UPDATE);
        } else {
            vstas(ureg, dstUbAddr, postUpdateStride, POST_UPDATE);
        }
    } else {
        if constexpr (std::is_same_v<T, bool>) {
            vstas(ureg, (__ubuf__ int8_t *&)dstUbAddr, postUpdateStride);
        } else {
            vstas(ureg, dstUbAddr, postUpdateStride);
        }
    }
}

// vstu/vsta
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(__local_mem__ T *&dstUbAddr, RegT &srcReg, UnalignReg &ureg, AddrReg &areg)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopyUnAlign only support type b8/b16/b32 on current device");
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    if constexpr (std::is_same_v<T, bool>) {
        vstu(ureg, areg, (RegTensor<int8_t> &)srcReg, (__ubuf__ int8_t *&)dstUbAddr, postValue);
    } else {
        vstu(ureg, areg, srcReg, dstUbAddr, postValue);
    }
}

template <typename T>
__aicore__ inline void DataCopyUnAlignPostImpl(__local_mem__ T *&dstUbAddr, UnalignReg &ureg, AddrReg &areg)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopyUnAlignPost only support type b8/b16/b32 on current device");
    vsta(ureg, dstUbAddr, areg);
}

// vstur/vstar
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(__local_mem__ T *dstUbAddr, RegT &srcReg, UnalignReg &ureg)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(
        SupportType<ActualT, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, half, float, bfloat16_t>(),
        "current data type is not supported on current device!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only support RegTraitNumOne on current device!");
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    if constexpr (std::is_same_v<T, bool>) {
        vstur(ureg, (RegTensor<int8_t> &)srcReg, (__ubuf__ int8_t *)dstUbAddr, postValue);
    } else {
        vstur(ureg, srcReg, dstUbAddr, postValue);
    }
}

template <typename T>
__aicore__ inline void DataCopyUnAlignPostImpl(__local_mem__ T *dstUbAddr, UnalignReg &ureg)
{
    static_assert(
        SupportType<T, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, half, float, bfloat16_t>(),
        "current data type is not supported on current device!");
    vstar(ureg, dstUbAddr);
}

// vgather2
template <typename DstT = DefaultType, typename SrcT, typename IndexT = DefaultType, typename RegDstT,
    typename RegIndexT>
__aicore__ inline void DataCopyGatherImpl(
    RegDstT &dstReg, __local_mem__ SrcT *baseAddr, RegIndexT &index, MaskReg &mask)
{
    using ActualDstT = typename RegDstT::ActualT;
    using ActualIndexT = typename RegIndexT::ActualT;
    static_assert(std::is_same_v<DstT, DefaultType> || std::is_same_v<DstT, ActualDstT>, "DstT type is not correct!");
    static_assert(
        std::is_same_v<IndexT, DefaultType> || std::is_same_v<IndexT, ActualIndexT>, "IndexT type is not correct!");
    static_assert((sizeof(SrcT) == 1 && sizeof(ActualDstT) == 2 && std::is_same_v<ActualIndexT, uint16_t>) ||
                      (sizeof(SrcT) == 2 && sizeof(ActualDstT) == 2 && std::is_same_v<ActualIndexT, uint16_t>) ||
                      (sizeof(SrcT) == 4 && sizeof(ActualDstT) == 4 && std::is_same_v<ActualIndexT, uint32_t>),
        "DataCopyGather only support src data type b8/b16/b32 with dst type is b16/b16/b32 respectively and each index "
        "type is u16/u16/u32 respectively");
    if constexpr (sizeof(SrcT) == 1 && sizeof(ActualDstT) == 2) {
        vgather2((vector_s16 &)dstReg, (__ubuf__ int8_t *)baseAddr, index, mask);
    } else if constexpr (sizeof(SrcT) == 2 && sizeof(ActualDstT) == 2) {
        vgather2((vector_s16 &)dstReg, (__ubuf__ int16_t *)baseAddr, index, mask);
    } else if constexpr (sizeof(SrcT) == 4 && sizeof(ActualDstT) == 4) {
        vgather2((vector_s32 &)dstReg, (__ubuf__ int32_t *)baseAddr, index, mask);
    }
}

template <typename DstT, typename SrcT, typename IndexT, typename RegDstT>
__aicore__ inline void DataCopyGatherImpl(
    RegDstT &dstReg, __local_mem__ SrcT *baseAddr, AddrReg &areg, __local_mem__ IndexT *index)
{
    using ActualDstT = typename RegDstT::ActualT;
    static_assert(std::is_same_v<DstT, DefaultType> || std::is_same_v<DstT, ActualDstT>, "DstT type is not correct!");
    static_assert(std::is_same_v<IndexT, DefaultType>, "IndexT type is not correct!");
    if constexpr (sizeof(SrcT) == 1 && sizeof(ActualDstT) == 2) {
        vgather2((vector_s16 &)dstReg, index, areg, (uint32_t)baseAddr);
    } else if constexpr (sizeof(SrcT) == 2 && sizeof(ActualDstT) == 2) {
        vgather2((vector_s16 &)dstReg, index, areg, (uint32_t)baseAddr);
    } else {
        vgather2((vector_s32 &)dstReg, index, areg, (uint32_t)baseAddr);
    }
}

// vgatherb
template <typename T = DefaultType, typename RegT, typename RegIndexT>
__aicore__ inline void DataCopyGatherBImpl(RegT &dstReg, __local_mem__ T *baseAddr, RegIndexT &index, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualIndexT = typename RegIndexT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<ActualIndexT, uint32_t>, "IndexT type is not correct!");
    static_assert(CheckRegTrait<RegT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(
        CheckRegTrait<RegIndexT, RegTraitNumOne>(), "RegTensor only suppoort RegTraitNumOne on current device!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(),
        "DataCopyGatherB only support src & dst datatype b8/b16/b32 on current device");
    if constexpr (sizeof(ActualT) == 1) {
        vgatherb((vector_s8 &)dstReg, (__ubuf__ int8_t *)baseAddr, index);
    } else if constexpr (sizeof(ActualT) == 2) {
        vgatherb((vector_s16 &)dstReg, (__ubuf__ int16_t *)baseAddr, index);
    } else if constexpr (sizeof(ActualT) == 4) {
        vgatherb((vector_s32 &)dstReg, (__ubuf__ int32_t *)baseAddr, index);
    }
}

// vgatherb
template <typename T, typename RegT>
__aicore__ inline void DataCopyGatherBImpl(
    RegT &dstReg, __local_mem__ T *baseAddr, __local_mem__ uint32_t *index, AddrReg areg)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopyGatherB only support type b8/b16/b32 on current device");
    vgatherb(dstReg, index, areg, (uint32_t)baseAddr);
}

// vscatter
template <typename T = DefaultType, typename IndexT = DefaultType, typename RegT, typename RegIndexT>
__aicore__ inline void DataCopyScatterImpl(__local_mem__ T *baseAddr, RegT &srcReg, RegIndexT &index, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualIndexT = typename RegIndexT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(
        std::is_same_v<IndexT, DefaultType> || std::is_same_v<IndexT, ActualIndexT>, "IndexT type is not correct!");
    static_assert((sizeof(ActualT) == 1 && std::is_same_v<ActualIndexT, uint16_t>) ||
                      (sizeof(ActualT) == 2 && std::is_same_v<ActualIndexT, uint16_t>) ||
                      (sizeof(ActualT) == 4 && std::is_same_v<ActualIndexT, uint32_t>),
        "DataCopyScatter only support data type b8/b16/b32"
        "with each index type is u16/u16/u32/(u32/) respectively on current device");
    vscatter(srcReg, baseAddr, index, mask);
}

// pld
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(MaskReg &mask, __local_mem__ T *srcUbAddr, AddrReg offset)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    static_assert(SupportEnum<dist, MaskDist::DIST_NORM, MaskDist::DIST_US, MaskDist::DIST_DS>(),
        "DataCopy not support this dist on current device");
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    pld(mask, (__ubuf__ uint32_t *)srcUbAddr, offset, distValue);
}

// plds
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(MaskReg &mask, __local_mem__ T *srcUbAddr)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    static_assert(SupportEnum<dist, MaskDist::DIST_NORM, MaskDist::DIST_US, MaskDist::DIST_DS>(),
        "DataCopy not support this dist on current device");
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    plds(mask, (__ubuf__ uint32_t *)srcUbAddr, 0, distValue);
}

template <typename T, PostLiteral postMode, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(MaskReg &mask, __local_mem__ T *&srcUbAddr, int32_t offset)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    static_assert(SupportEnum<dist, MaskDist::DIST_NORM, MaskDist::DIST_US, MaskDist::DIST_DS>(),
        "DataCopy not support this dist on current device");
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    plds(mask, (__ubuf__ uint32_t *&)srcUbAddr, offset, distValue, postValue);
}

// pst
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, MaskReg &mask, AddrReg offset)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    static_assert(SupportEnum<dist, MaskDist::DIST_NORM, MaskDist::DIST_US, MaskDist::DIST_DS>(),
        "DataCopy not support this dist on current device");
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    pst(mask, (__ubuf__ uint32_t *)dstUbAddr, offset, distValue);
}

// psts
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, MaskReg &mask)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    static_assert(SupportEnum<dist, MaskDist::DIST_NORM, MaskDist::DIST_PACK>(),
        "DataCopy not support this dist on current device");
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    psts(mask, (__ubuf__ uint32_t *)dstUbAddr, 0, distValue);
}

template <typename T, PostLiteral postMode, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(__local_mem__ T *&dstUbAddr, MaskReg &mask, int32_t offset)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    static_assert(SupportEnum<dist, MaskDist::DIST_NORM, MaskDist::DIST_PACK>(),
        "DataCopy not support this dist on current device");
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    psts(mask, (__ubuf__ uint32_t *&)dstUbAddr, offset, distValue, postValue);
}

template <typename T>
__aicore__ inline void DataCopyUnAlignImpl(__local_mem__ T *&dstUbAddr, MaskReg &mask, UnalignReg &ureg)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "DataCopyUnAlign is not supported on current device!"); });
}
}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_MODULE_MICRO_DATACOPY_IMPL_H
