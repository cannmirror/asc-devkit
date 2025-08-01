/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SOFTMAX_COMMON_IMPL_H
#define AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SOFTMAX_COMMON_IMPL_H

namespace AscendC {
constexpr MicroAPI::CastTrait softmaxCastTraitF16F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING};
constexpr MicroAPI::CastTrait softmaxCastTraitF32F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void LoadIfNeedCast(
    MicroAPI::RegTensor<float>& dstReg, __local_mem__ T* srcUb, MicroAPI::MaskReg& preg)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::RegTensor<T> tmpReg;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(tmpReg, srcUb);
        MicroAPI::Cast<float, T, softmaxCastTraitF16F32>(dstReg, tmpReg, preg);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(dstReg, srcUb);
    }
};

template <typename T>
__aicore__ inline void StoreIfNeedCast(
    __local_mem__ T* dstUb, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& preg)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::RegTensor<T> tmpReg;
        MicroAPI::Cast<T, float, softmaxCastTraitF32F16>(tmpReg, srcReg, preg);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb, tmpReg, preg);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(dstUb, srcReg, preg);
    }
};

template <typename T>
__aicore__ inline void LoadE2B(MicroAPI::RegTensor<T>& dstReg, __local_mem__ T* srcUb)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B16>(dstReg, srcUb);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(dstReg, srcUb);
    }
};

};     // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_REGBASE_C310_SOFTMAX_COMMON_IMPL_H