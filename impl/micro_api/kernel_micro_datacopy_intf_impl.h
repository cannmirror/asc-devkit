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

/* !
 * \file kernel_micro_datacopy_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_DATACOPY_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_DATACOPY_INTERFACE_IMPL_H

#include "kernel_micro_maskreg_intf_impl.h"

#if __NPU_ARCH__ == 2103
#include "micro_api/dav_l210/kernel_micro_datacopy_impl.h"
#elif __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_datacopy_impl.h"
#elif __NPU_ARCH__ == 3103
#include "micro_api/dav_l310/kernel_micro_datacopy_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_datacopy_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_datacopy_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
// vld
template <typename T, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U &dstReg, __local_mem__ T *srcUbAddr)
{
    DataCopyImpl<T, dist, U>(dstReg, srcUbAddr);
}

template <typename T, PostLiteral postMode, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U &dstReg, __local_mem__ T *&srcUbAddr, int32_t postUpdateStride)

{
    DataCopyImpl<T, postMode, dist, U>(dstReg, srcUbAddr, postUpdateStride);
}

template <typename T, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U &dstReg, __local_mem__ T *srcUbAddr, AddrReg offset)

{
    DataCopyImpl<T, dist, U>(dstReg, srcUbAddr, offset);
}

// vld dual
template <typename T, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U &dstReg0, U &dstReg1, __local_mem__ T *srcUbAddr)
{
    DataCopyImpl<T, dist, U>(dstReg0, dstReg1, srcUbAddr);
}

template <typename T, PostLiteral postMode, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U &dstReg0, U &dstReg1, __local_mem__ T *&srcUbAddr, int32_t postUpdateStride)

{
    DataCopyImpl<T, postMode, dist, U>(dstReg0, dstReg1, srcUbAddr, postUpdateStride);
}

template <typename T, LoadDist dist, typename U>
__simd_callee__ inline void DataCopy(U &dstReg0, U &dstReg1, __local_mem__ T *srcUbAddr, AddrReg offset)
{
    DataCopyImpl<T, dist, U>(dstReg0, dstReg1, srcUbAddr, offset);
}

// vst
template <typename T, StoreDist dist, typename U>
__simd_callee__ inline void DataCopy(__local_mem__ T *dstUbAddr, U &srcReg, MaskReg &mask)
{
    DataCopyImpl<T, dist, U>(dstUbAddr, srcReg, mask);
}

template <typename T, PostLiteral postMode, StoreDist dist, typename U>
__simd_callee__ inline void DataCopy(__local_mem__ T *&dstUbAddr, U &srcReg, int32_t postUpdateStride, MaskReg &mask)
{
    DataCopyImpl<T, postMode, dist, U>(dstUbAddr, srcReg, postUpdateStride, mask);
}

template <typename T, StoreDist dist, typename U>
__simd_callee__ inline void DataCopy(__local_mem__ T *dstUbAddr, U &srcReg, AddrReg offset, MaskReg &mask)
{
    DataCopyImpl<T, dist, U>(dstUbAddr, srcReg, offset, mask);
}

// vst dual
template <typename T, StoreDist dist, typename U>
__simd_callee__ inline void DataCopy(__local_mem__ T *dstUbAddr, U &srcReg0, U &srcReg1, MaskReg &mask)
{
    DataCopyImpl<T, dist, U>(dstUbAddr, srcReg0, srcReg1, mask);
}

template <typename T, StoreDist dist, typename U>
__simd_callee__ inline void DataCopy(__local_mem__ T *dstUbAddr, U &srcReg0, U &srcReg1, AddrReg offset,
    MaskReg &mask)
{
    DataCopyImpl<T, dist, U>(dstUbAddr, srcReg0, srcReg1, offset, mask);
}

// vsldb
template <typename T, DataCopyMode dataMode, typename U>
__simd_callee__ inline void DataCopy(U &dstReg, __local_mem__ T *srcUbAddr, uint32_t dataBlockStride, MaskReg &mask)
{
    DataCopyImpl<T, dataMode, U>(dstReg, srcUbAddr, dataBlockStride, mask);
}

template <typename T, DataCopyMode dataMode, PostLiteral postMode, typename U>
__simd_callee__ inline void DataCopy(U &dstReg, __local_mem__ T *&srcUbAddr, uint32_t dataBlockStride,
    uint32_t repeatStride, MaskReg &mask)
{
    DataCopyImpl<T, dataMode, postMode, U>(dstReg, srcUbAddr, dataBlockStride, repeatStride, mask);
}

// vsstb
template <typename T, DataCopyMode dataMode, typename U>
__simd_callee__ inline void DataCopy(__local_mem__ T *dstUbAddr, U &srcReg, uint32_t dataBlockStride, MaskReg &mask)
{
    DataCopyImpl<T, dataMode, U>(dstUbAddr, srcReg, dataBlockStride, mask);
}

template <typename T, DataCopyMode dataMode, PostLiteral postMode, typename U>
__simd_callee__ inline void DataCopy(__local_mem__ T *&dstUbAddr, U &srcReg, uint32_t dataBlockStride,
    uint32_t repeatStride, MaskReg &mask)
{
    DataCopyImpl<T, dataMode, postMode, U>(dstUbAddr, srcReg, dataBlockStride, repeatStride, mask);
}

// vldas/vldus
template <typename T>
__simd_callee__ inline void DataCopyUnAlignPre(UnalignReg &ureg, __local_mem__ T *srcUbAddr)
{
    DataCopyUnAlignPreImpl<T>(ureg, srcUbAddr);
}

template <typename T, PostLiteral postMode, typename U>
__simd_callee__ inline void DataCopyUnAlign(U &dstReg, UnalignReg &ureg, __local_mem__ T *&srcUbAddr, uint32_t postUpdateStride)
{
    DataCopyUnAlignImpl<T, postMode, U>(dstReg, ureg, srcUbAddr, postUpdateStride);
}

template <typename T, typename U>
__simd_callee__ inline void DataCopyUnAlign(U &dstReg, UnalignReg &ureg, __local_mem__ T *srcUbAddr)
{
    DataCopyUnAlignImpl<T, U>(dstReg, ureg, srcUbAddr);
}

// vlda/vldu
template <typename T>
__simd_callee__ inline void DataCopyUnAlignPre(UnalignReg &ureg, __local_mem__ T *srcUbAddr, AddrReg &areg)
{
    DataCopyUnAlignPreImpl<T>(ureg, srcUbAddr, areg);
}

template <typename T, typename U>
__simd_callee__ inline void DataCopyUnAlign(U &dstReg, UnalignReg &ureg, __local_mem__ T *&srcUbAddr, AddrReg &areg,
    uint32_t inc)
{
    DataCopyUnAlignImpl<T, U>(dstReg, ureg, srcUbAddr, areg, inc);
}

// vstus/vstas
template <typename T, PostLiteral postMode , typename U>
__simd_callee__ inline void DataCopyUnAlign(__local_mem__ T *&dstUbAddr, U &srcReg, UnalignReg &ureg, uint32_t postUpdateStride)

{
    DataCopyUnAlignImpl<T, postMode, U>(dstUbAddr, srcReg, ureg, postUpdateStride);
}

template <typename T, PostLiteral postMode>
__simd_callee__ inline void DataCopyUnAlignPost(__local_mem__ T *&dstUbAddr, UnalignReg &ureg, int32_t postUpdateStride)
{
    DataCopyUnAlignPostImpl<T, postMode>(dstUbAddr, ureg, postUpdateStride);
}

// vstu/vsta
template <typename T, PostLiteral postMode, typename U>
__simd_callee__ inline void DataCopyUnAlign(__local_mem__ T *&dstUbAddr, U &srcReg, UnalignReg &ureg, AddrReg &areg)
{
    DataCopyUnAlignImpl<T, postMode, U>(dstUbAddr, srcReg, ureg, areg);
}

template <typename T>
__simd_callee__ inline void DataCopyUnAlignPost(__local_mem__ T *&dstUbAddr, UnalignReg &ureg, AddrReg &areg)
{
    DataCopyUnAlignPostImpl<T>(dstUbAddr, ureg, areg);
}

// vstur/vstar
template <typename T, PostLiteral postMode, typename U>
__simd_callee__ inline void DataCopyUnAlign(__local_mem__ T *dstUbAddr, U &srcReg, UnalignReg &ureg)
{
    DataCopyUnAlignImpl(dstUbAddr, srcReg, ureg);
}

template <typename T>
__simd_callee__ inline void DataCopyUnAlignPost(__local_mem__ T *dstUbAddr, UnalignReg &ureg)
{
    DataCopyUnAlignPostImpl(dstUbAddr, ureg);
}

// vgather2
template <typename T0, typename T1, typename T2, typename T3, typename T4>
__simd_callee__ inline void DataCopyGather(
    T3 &dstReg, __local_mem__ T1 *baseAddr, T4 &index, MaskReg &mask)
{
    DataCopyGatherImpl<T0, T1, T2, T3, T4>(dstReg, baseAddr, index, mask);
}

// vgatherb
template <typename T, typename U, typename S>
__simd_callee__ inline void DataCopyGatherB(U &dstReg, __local_mem__ T *baseAddr, S &index,
    MaskReg &mask)

{
    DataCopyGatherBImpl<T, U, S>(dstReg, baseAddr, index, mask);
}

// vscatter
template <typename T, typename U, typename S, typename V>
__simd_callee__ inline void DataCopyScatter(__local_mem__ T *baseAddr, S &srcReg, V &index,
    MaskReg &mask)

{
    DataCopyScatterImpl<T, U, S, V>(baseAddr, srcReg, index, mask);
}

// pld
template <typename T, MaskDist dist>
__simd_callee__ inline void DataCopy(MaskReg &mask, __local_mem__ T *srcUbAddr, AddrReg offset)
{
    DataCopyImpl<T, dist>(mask, srcUbAddr, offset);
}

// plds
template <typename T, MaskDist dist>
__simd_callee__ inline void DataCopy(MaskReg &mask, __local_mem__ T *srcUbAddr)
{
    DataCopyImpl<T, dist>(mask, srcUbAddr);
}

template <typename T, PostLiteral postMode, MaskDist dist>
__simd_callee__ inline void DataCopy(MaskReg &mask, __local_mem__ T *&srcUbAddr, int32_t offset)
{
    DataCopyImpl<T, postMode, dist>(mask, srcUbAddr, offset);
}

// pst
template <typename T, MaskDist dist>
__simd_callee__ inline void DataCopy(__local_mem__ T *dstUbAddr, MaskReg &mask, AddrReg offset)
{
    DataCopyImpl<T, dist>(dstUbAddr, mask, offset);
}

// psts
template <typename T, MaskDist dist>
__simd_callee__ inline void DataCopy(__local_mem__ T *dstUbAddr, MaskReg &mask)
{
    DataCopyImpl<T, dist>(dstUbAddr, mask);
}

template <typename T, PostLiteral postMode, MaskDist dist>
__simd_callee__ inline void DataCopy(__local_mem__ T *&dstUbAddr, MaskReg &mask, int32_t offset)
{
    DataCopyImpl<T, postMode, dist>(dstUbAddr, mask, offset);
}

// pstu
template <typename T>
__simd_callee__ inline void DataCopyUnAlign(__local_mem__ T *&dstUbAddr, MaskReg &mask, UnalignReg &ureg)
{
    return DataCopyUnAlignImpl<T>(dstUbAddr, mask, ureg);
}
}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_KERNEL_MICRO_DATACOPY_INTERFACE_IMPL_H
