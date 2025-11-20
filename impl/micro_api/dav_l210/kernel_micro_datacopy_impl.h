/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
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
                          LoadDist::DIST_E2B_B16>(),
            "DataCopy not support this dist on this device");
    } else {
        static_assert(SupportEnum<dist, LoadDist::DIST_BDINTLV, LoadDist::DIST_DINTLV_B8, LoadDist::DIST_DINTLV_B16>(),
            "DataCopy not support this dist on this device");
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
                          StoreDist::DIST_PACK_B32>(),
            "DataCopy not support this dist on this device");
    } else {
        static_assert(SupportEnum<dist, StoreDist::DIST_INTLV_B8, StoreDist::DIST_INTLV_B16>(),
            "DataCopy not support this dist on this device");
    }
}

// vld
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg, __local_mem__ T *srcUbAddr)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    CheckLoadDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    vld(dstReg, srcUbAddr, 0, distValue);
}

template <typename T = DefaultType, PostLiteral postMode, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg, __local_mem__ T *&srcUbAddr, int32_t offset)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __local_mem__ T *srcUbAddr)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    CheckLoadDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    vld(dstReg0, dstReg1, srcUbAddr, 0, distValue);
}

template <typename T = DefaultType, PostLiteral postMode, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __local_mem__ T *&srcUbAddr, int32_t offset)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

// pstu
template <typename T>
__aicore__ inline void DataCopyUnAlignImpl(__local_mem__ T *&dstUbAddr, MaskReg &mask, UnalignReg &ureg)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "DataCopyUnAlign is not supported on current device!"); });
}

// vld
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg, __local_mem__ T *srcUbAddr, AddrReg offset)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    CheckLoadDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    vld(dstReg, srcUbAddr, offset, distValue);
}

template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg0, RegT &dstReg1, __local_mem__ T *srcUbAddr, AddrReg offset)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    CheckLoadDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    vld(dstReg0, dstReg1, srcUbAddr, offset, distValue);
}

// vsts
template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    CheckStoreDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    vst(srcReg, dstUbAddr, 0, distValue, mask);
}

template <typename T = DefaultType, PostLiteral postMode, StoreDist dist = StoreDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *&dstUbAddr, RegT &srcReg, int32_t offset, MaskReg &mask)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    CheckStoreDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    vst(srcReg0, srcReg1, dstUbAddr, 0, distValue, mask);
}

// vst
template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, RegT &srcReg, AddrReg offset, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    CheckStoreDist<1, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    vst(srcReg, dstUbAddr, offset, distValue, mask);
}

template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename RegT>
__aicore__ inline void DataCopyImpl(
    __local_mem__ T *dstUbAddr, RegT &srcReg0, RegT &srcReg1, AddrReg offset, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    CheckStoreDist<2, dist>();
    constexpr auto distValue = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<T, dist>())>();
    vst(srcReg0, srcReg1, dstUbAddr, offset, distValue, mask);
}

// vsldb
template <typename T = DefaultType, DataCopyMode dataMode, typename RegT>
__aicore__ inline void DataCopyImpl(RegT &dstReg, __local_mem__ T *srcUbAddr, uint32_t blockStride, MaskReg &mask)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename RegT>
__aicore__ inline void DataCopyImpl(
    RegT &dstReg, __local_mem__ T *&srcUbAddr, uint32_t blockStride, uint32_t repeatStride, MaskReg &mask)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

// vsstb
template <typename T = DefaultType, DataCopyMode dataMode, typename RegT>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, RegT &srcReg, uint32_t blockStride, MaskReg &mask)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename RegT>
__aicore__ inline void DataCopyImpl(
    __local_mem__ T *&dstUbAddr, RegT &srcReg, uint32_t blockStride, uint32_t repeatStride, MaskReg &mask)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

// vldas/vldus
template <typename T>
__aicore__ inline void DataCopyUnAlignPreImpl(UnalignReg &ureg, __local_mem__ T *srcUbAddr)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyUnAlign is not supported on current device!"); });
}

template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(RegT &dstReg, UnalignReg &ureg, __local_mem__ T *&srcUbAddr, uint32_t stride)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyUnAlign is not supported on current device!"); });
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(RegT &dstReg, UnalignReg &ureg, __local_mem__ T *srcUbAddr)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyUnAlign is not supported on current device!"); });
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
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopyUnAlign only support type b8/b16/b32 on current device");
    vldu(dstReg, ureg, areg, srcUbAddr, inc);
}

// vstus/vstas
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(__local_mem__ T *&dstUbAddr, RegT &srcReg, UnalignReg &ureg, uint32_t offset)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyUnAlign is not supported on current device!"); });
}

template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE>
__aicore__ inline void DataCopyUnAlignPostImpl(__local_mem__ T *&dstUbAddr, UnalignReg &ureg, int32_t offset)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyUnAlign is not supported on current device!"); });
}

// vstu/vsta
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename RegT>
__aicore__ inline void DataCopyUnAlignImpl(__local_mem__ T *&dstUbAddr, RegT &srcReg, UnalignReg &ureg, AddrReg &areg)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DataCopyUnAlign only support type b8/b16/b32 on current device");
    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(postMode)>();
    vstu(ureg, areg, srcReg, dstUbAddr, postValue);
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
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyUnAlign is not supported on current device!"); });
}

template <typename T>
__aicore__ inline void DataCopyUnAlignPostImpl(__local_mem__ T *dstUbAddr, UnalignReg &ureg)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyUnAlign is not supported on current device!"); });
}

// vgather2
template <typename DstT = DefaultType, typename SrcT, typename IndexT = DefaultType, typename RegDstT,
    typename RegIndexT>
__aicore__ inline void DataCopyGatherImpl(
    RegDstT &dstReg, __local_mem__ SrcT *baseAddr, RegIndexT &index, MaskReg &mask)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyGather is not supported on current device!"); });
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
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopyGatherB is not supported on current device!"); });
}

// vgatherb
template <typename T, typename RegT>
__aicore__ inline void DataCopyGatherBImpl(RegT &dstReg, __local_mem__ T *baseAddr, __local_mem__ uint32_t *index,
    AddrReg areg)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(),
        "DataCopyGatherB only support type b8/b16/b32 on current device");
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
        "DataCopyScatter only support data type b8/b16/b32 with each index type is u16/u16/u32 respectively in "
        "Ascend610Lite");
    RegIndexT realOffset;
    vshls(realOffset, index, (int16_t)sizeof(ActualT), mask, MaskMergeMode::MERGING);
    vscatter(srcReg, baseAddr, index, mask);
}

// plds
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(MaskReg &mask, __local_mem__ T *srcUbAddr)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

template <typename T, PostLiteral postMode, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(MaskReg &mask, __local_mem__ T *&srcUbAddr, int32_t offset)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

// psts
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, MaskReg &mask)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
}

template <typename T, PostLiteral postMode, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(__local_mem__ T *&dstUbAddr, MaskReg &mask, int32_t offset)
{
    ASCENDC_ASSERT(
        false, { KERNEL_LOG(KERNEL_ERROR, "this version of DataCopy is not supported on current device!"); });
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

// pst
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__aicore__ inline void DataCopyImpl(__local_mem__ T *dstUbAddr, MaskReg &mask, AddrReg offset)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "DataCopy only support type b8/b16/b32 on current device");
    static_assert(SupportEnum<dist, MaskDist::DIST_NORM, MaskDist::DIST_PACK>(),
        "DataCopy not support this dist on current device");
    constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(dist)>();
    pst(mask, (__ubuf__ uint32_t *)dstUbAddr, offset, distValue);
}

}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_MODULE_MICRO_DATACOPY_IMPL_H