/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_COMMON_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_COMMON_IMPL_H

#include "../../../../common/common.h"

namespace AscendC {
template <typename T>
__simd_callee__ inline void LoadIfNeedCast(
    MicroAPI::RegTensor<float>& dstReg, __local_mem__ T* srcUb, MicroAPI::MaskReg& preg)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::RegTensor<T> tmpReg;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(tmpReg, srcUb);
        MicroAPI::Cast<float, T, Internal::castTraitB16ToB32>(dstReg, tmpReg, preg);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(dstReg, srcUb);
    }
}

template <typename T>
__simd_callee__ inline void LoadIfNeedCastM1(
    MicroAPI::RegTensor<float>& dstReg, __local_mem__ T* srcUb, MicroAPI::MaskReg& preg)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::RegTensor<T> castVreg;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(castVreg, srcUb);
        MicroAPI::UnPack<uint32_t, uint16_t>(
            (MicroAPI::RegTensor<uint32_t>&)castVreg, (MicroAPI::RegTensor<uint16_t>&)castVreg);
        MicroAPI::Cast<float, T, Internal::castTraitB16ToB32>(dstReg, castVreg, preg);
    } else {
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dstReg, srcUb);
    }
}

template <typename T>
__simd_callee__ inline void StoreIfNeedCastM1(
    __local_mem__ T* dstUb, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& preg)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::RegTensor<T> castVreg;
        MicroAPI::Cast<T, float, Internal::castTraitB32ToB16>(castVreg, srcReg, preg);
        MicroAPI::Pack<uint16_t, uint32_t>(
            (MicroAPI::RegTensor<uint16_t>&)castVreg, (MicroAPI::RegTensor<uint32_t>&)castVreg);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(dstUb, castVreg, preg);
    } else {
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dstUb, srcReg, preg);
    }
}

template <typename T>
__simd_callee__ inline void StoreIfNeedCast(
    __local_mem__ T* dstUb, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& preg)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::RegTensor<T> tmpReg;
        MicroAPI::Cast<T, float, Internal::castTraitB32ToB16>(tmpReg, srcReg, preg);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb, tmpReg, preg);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(dstUb, srcReg, preg);
    }
}

template <typename T>
__simd_callee__ inline void LoadE2B(MicroAPI::RegTensor<T>& dstReg, __local_mem__ T* srcUb)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B16>(dstReg, srcUb);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(dstReg, srcUb);
    }
}
} // namespace AscendC
#endif // IMPL_ACTIVATION_SOFTMAX_C310_SOFTMAX_COMMON_IMPL_H
