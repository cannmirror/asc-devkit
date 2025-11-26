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
 * \file kernel_micro_datacopy_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_DATACOPY_INTERFACE_H
#define ASCENDC_MODULE_MICRO_DATACOPY_INTERFACE_H

#include "kernel_micro_common_intf.h"

namespace AscendC {
namespace MicroAPI {
// vld
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void DataCopy(U& dstReg, __ubuf__ T* srcUbAddr);

template <typename T = DefaultType, PostLiteral postMode, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void DataCopy(U& dstReg, __ubuf__ T*& srcUbAddr, int32_t postUpdateStride);

template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void DataCopy(U& dstReg, __ubuf__ T* srcUbAddr, AddrReg offset);

// vld dual
template <typename T = DefaultType, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U& dstReg0, U& dstReg1, __ubuf__ T* srcUbAddr);

template <typename T = DefaultType, PostLiteral postMode, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U& dstReg0, U& dstReg1, __ubuf__ T*& srcUbAddr, int32_t postUpdateStride);

template <typename T = DefaultType, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U& dstReg0, U& dstReg1, __ubuf__ T* srcUbAddr, AddrReg offset);

// vst
template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void DataCopy(__ubuf__ T* dstUbAddr, U& srcReg, MaskReg& mask);

template <typename T = DefaultType, PostLiteral postMode, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void DataCopy(__ubuf__ T*& dstUbAddr, U& srcReg, int32_t postUpdateStride, MaskReg& mask);

template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void DataCopy(__ubuf__ T* dstUbAddr, U& srcReg, AddrReg offset, MaskReg& mask);

// vst dual
template <typename T = DefaultType, StoreDist dist, typename U>
__simd_callee__ inline void DataCopy(__ubuf__ T* dstUbAddr, U& srcReg0, U& srcReg1, MaskReg& mask);

template <typename T = DefaultType, StoreDist dist, typename U>
__simd_callee__ inline void DataCopy(__ubuf__ T* dstUbAddr, U& srcReg0, U& srcReg1, AddrReg offset,
                                     MaskReg& mask);

// vsldb
template <typename T = DefaultType, DataCopyMode dataMode, typename U>
__simd_callee__ inline void DataCopy(U& dstReg, __ubuf__ T* srcUbAddr, uint32_t dataBlockStride, MaskReg& mask);

template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename U>
__simd_callee__ inline void DataCopy(U &dstReg, __ubuf__ T*& srcUbAddr, uint32_t dataBlockStride,
                                     uint32_t repeatStride, MaskReg& mask);

// vsstb
template <typename T = DefaultType, DataCopyMode dataMode, typename U>
__simd_callee__ inline void DataCopy(__ubuf__ T* dstUbAddr, U& srcReg, uint32_t dataBlockStride, MaskReg& mask);

template <typename T = DefaultType, DataCopyMode dataMode, PostLiteral postMode, typename U>
__simd_callee__ inline void DataCopy(__ubuf__ T*& dstUbAddr, U& srcReg, uint32_t dataBlockStride,
                                     uint32_t repeatStride, MaskReg& mask);

// vldas/vldus
template <typename T>
__simd_callee__ inline void DataCopyUnAlignPre(UnalignReg& ureg, __ubuf__ T* srcUbAddr);

template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void DataCopyUnAlign(U& dstReg, UnalignReg& ureg, __ubuf__ T*& srcUbAddr,
                                            uint32_t postUpdateStride);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void DataCopyUnAlign(U& dstReg, UnalignReg& ureg, __ubuf__ T* srcUbAddr);

// vlda/vldu
template <typename T>
__simd_callee__ inline void DataCopyUnAlignPre(UnalignReg& ureg, __ubuf__ T* srcUbAddr, AddrReg& areg);

template <typename T = DefaultType, typename U>
__simd_callee__ inline void DataCopyUnAlign(U& dstReg, UnalignReg& ureg, __ubuf__ T*& srcUbAddr, AddrReg& areg,
                                            uint32_t inc);

// vstus/vstas
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void DataCopyUnAlign(__ubuf__ T*& dstUbAddr, U& srcReg, UnalignReg& ureg,
                                            uint32_t postUpdateStride);

template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE>
__simd_callee__ inline void DataCopyUnAlignPost(__ubuf__ T*& dstUbAddr, UnalignReg& ureg, int32_t postUpdateStride);

// vstu/vsta
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void DataCopyUnAlign(__ubuf__ T*& dstUbAddr, U& srcReg, UnalignReg& ureg, AddrReg& areg);

template <typename T>
__simd_callee__ inline void DataCopyUnAlignPost(__ubuf__ T*& dstUbAddr, UnalignReg& ureg, AddrReg& areg);

// vstur/vstar
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void DataCopyUnAlign(__ubuf__ T* dstUbAddr, U& srcReg, UnalignReg& ureg);

template <typename T>
__simd_callee__ inline void DataCopyUnAlignPost(__ubuf__ T* dstUbAddr, UnalignReg& ureg);

// vgather2
template <typename T0 = DefaultType, typename T1, typename T2 = DefaultType, typename T3, typename T4>
__simd_callee__ inline void DataCopyGather(T3& dstReg, __ubuf__ T1* baseAddr, T4& index, MaskReg& mask);

// vgatherb
template <typename T = DefaultType, typename U, typename S>
__simd_callee__ inline void DataCopyGatherB(U& dstReg, __ubuf__ T* baseAddr, S& index, MaskReg& mask);

// vscatter
template <typename T = DefaultType, typename U = DefaultType, typename S, typename V>
__simd_callee__ inline void DataCopyScatter(__ubuf__ T* baseAddr, S& srcReg, V& index, MaskReg& mask);

// pld
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void DataCopy(MaskReg& mask, __ubuf__ T* srcUbAddr, AddrReg offset);

// plds
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void DataCopy(MaskReg& mask, __ubuf__ T* srcUbAddr);

template <typename T, PostLiteral postMode, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void DataCopy(MaskReg& mask, __ubuf__ T*& srcUbAddr, int32_t offset);

// pst
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void DataCopy(__ubuf__ T* dstUbAddr, MaskReg& mask, AddrReg offset);

// psts
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void DataCopy(__ubuf__ T* dstUbAddr, MaskReg& mask);

template <typename T, PostLiteral postMode, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void DataCopy(__ubuf__ T*& dstUbAddr, MaskReg& mask, int32_t offset);

// pstu
template <typename T>
__simd_callee__ inline void DataCopyUnAlign(__ubuf__ T*& dstUbAddr, MaskReg& mask, UnalignReg& ureg);
} // namespace MicroAPI
} // namespace AscendC

#include "impl/micro_api/kernel_micro_datacopy_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_DATACOPY_INTERFACE_H
