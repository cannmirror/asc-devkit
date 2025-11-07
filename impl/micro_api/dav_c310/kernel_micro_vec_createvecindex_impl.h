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
 * \file kernel_micro_vec_createvecindex_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_CREATEVECINDEX_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_CREATEVECINDEX_IMPL_H

namespace AscendC {
namespace MicroAPI {
template <IndexOrder order = IndexOrder::INCREASE_ORDER, typename T1, typename RegT>
__simd_callee__ inline void ArangeB64Impl(RegT &dstReg, T1 scalar)
{
    using ActualT = typename RegT::ActualT;
    static_assert((SupportType<ActualT, int64_t>()),
        "ArangeB64Impl only support B64 data type");
    constexpr auto orderMode = std::integral_constant<::Order, static_cast<::Order>(order)>();
    static_assert(CheckRegTrait<RegT, RegTraitNumTwo>(), "ArangeB64Impl only support RegTraitNumTwo");
    MaskReg maskReg = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    Duplicate((RegTensor<int32_t> &)dstReg.reg[1], int32_t(0), maskReg);
    Arange<DefaultType, order>((RegTensor<int32_t> &)dstReg.reg[0], int32_t(0));
    Adds(dstReg, dstReg, scalar, maskReg);
}

template <typename T = DefaultType, IndexOrder order = IndexOrder::INCREASE_ORDER, typename T1, typename RegT>
__simd_callee__ inline void ArangeImpl(RegT &dstReg, T1 scalar)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, int8_t, int16_t, int32_t, float, half, int64_t>()),
        "current Arange data type is not supported on current device!");
    static_assert(Std::is_convertible<T1, ActualT>(), "scalar data type could be converted to RegTensor data type");
    constexpr auto orderMode = std::integral_constant<::Order, static_cast<::Order>(order)>();
    if constexpr(sizeof(ActualT) != 8) {
        vci(dstReg, scalar, orderMode);
    } else {
        if constexpr(CheckRegTrait<RegT, RegTraitNumOne>()) {
            RegTensor<ActualT, RegTraitNumTwo> traitTwoDstReg;
            ArangeB64Impl(traitTwoDstReg, scalar);
            B64TraitTwoToTaitOne(dstReg, traitTwoDstReg);
        } else if constexpr(CheckRegTrait<RegT, RegTraitNumTwo>()) {
            RegT dstTemp;
            ArangeB64Impl<order, T1, RegT>(dstTemp, scalar);
            dstReg = dstTemp;
        }
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_CREATEVECINDEX_IMPL_H