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
 * \file logical_template_impl.h
 * \brief
 */

#ifndef LIB_MATH_LOGICAL_TEMPLATE_IMPL_H
#define LIB_MATH_LOGICAL_TEMPLATE_IMPL_H
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101 || __NPU_ARCH__ == 5102)
#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
namespace AscendC {
constexpr uint32_t LOGICAL_TEMPLATE_B64_REPEAT_STRIDE = 2;

template <auto func, typename T, typename U, typename RegT, typename RegU, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__simd_vf__ inline void LogicalTemplateVF(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, 
    uint16_t repeatTime, uint32_t count, uint32_t oneRepElm)
{
    RegT dstVreg;
    RegT brcOneReg;
    RegT brcZeroReg;
    RegU src0Vreg;
    RegU src1Vreg;
    MicroAPI::MaskReg mask;
    MicroAPI::MaskReg cmpMask0;
    MicroAPI::MaskReg cmpMask1;
    MicroAPI::MaskReg cmpMask2;

    MicroAPI::Duplicate(brcOneReg, 1u);
    MicroAPI::Duplicate(brcZeroReg, 0u);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<U, Trait>(count);
        MicroAPI::LoadAlign(src0Vreg, src0 + i * oneRepElm);
        MicroAPI::LoadAlign(src1Vreg, src1 + i * oneRepElm);
        MicroAPI::CompareScalar<U, CMPMODE::NE>(cmpMask0, src0Vreg, static_cast<U>(0), mask);
        MicroAPI::CompareScalar<U, CMPMODE::NE>(cmpMask1, src1Vreg, static_cast<U>(0), mask);
        func(cmpMask2, cmpMask0, cmpMask1, mask);
        if constexpr (sizeof(U) == 2) {
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(mask, mask);
        } else if constexpr (sizeof(U) == 4 || sizeof(U) == 8) {
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(mask, mask);
            MicroAPI::MaskPack(mask, mask);
        }
        MicroAPI::Select(dstVreg, brcOneReg, brcZeroReg, cmpMask2);
        MicroAPI::StoreAlign(dst + i * oneRepElm, dstVreg, mask);
    }
}

template <auto func, typename T, typename U>
__aicore__ inline void LogicalTemplateImpl(const LocalTensor<T>& dst, const LocalTensor<U>& src0,
    const LocalTensor<U>& src1, const uint32_t count)
{
    static_assert(SupportType<T, bool>(), "only support bool data type on current device!");
    static_assert(SupportType<U, bool, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, 
        uint32_t, int32_t, uint64_t, int64_t>(), "only support bool/uint8_t/int8_t/half/bfloat16_t/"
        "uint16_t/int16_t/float/uint32_t/int32_t/uint64_t/int64_t data type on current device!");
    using RegT = MicroAPI::RegTensor<T>;
    if constexpr (sizeof(U) == 8) {
        using RegU = MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>;
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(U) * LOGICAL_TEMPLATE_B64_REPEAT_STRIDE);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        LogicalTemplateVF<func, T, U, RegT, RegU, MicroAPI::RegTraitNumTwo>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ U*)src0.GetPhyAddr(),
            (__ubuf__ U*)src1.GetPhyAddr(), repeatTime, count, oneRepElm);
    } else {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(U));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        if constexpr (Std::is_same_v<U, bool>) { 
            using RegU = MicroAPI::RegTensor<uint8_t>;
            LogicalTemplateVF<func, T, uint8_t, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ uint8_t*)src0.GetPhyAddr(),
                (__ubuf__ uint8_t*)src1.GetPhyAddr(), repeatTime, count, oneRepElm);
        } else {
            using RegU = MicroAPI::RegTensor<U>;
            LogicalTemplateVF<func, T, U, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ U*)src0.GetPhyAddr(),
                (__ubuf__ U*)src1.GetPhyAddr(), repeatTime, count, oneRepElm);
        }
    }
}

// Logical Tensor Scalar

template <auto func, typename T, typename U, typename RegT, typename RegU, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__simd_vf__ inline void LogicalTemplateBothTensorVF(__ubuf__ T* dst, __ubuf__ U* src, __ubuf__ U* scalar, 
    uint16_t repeatTime, uint32_t count, uint32_t oneRepElm)
{
    RegT dstVreg;
    RegT brcOneReg;
    RegT brcZeroReg;
    RegU srcVreg;
    RegU dupVreg;
    MicroAPI::MaskReg mask;
    MicroAPI::MaskReg cmpMask0;
    MicroAPI::MaskReg cmpMask1;
    MicroAPI::MaskReg cmpMask2;
    MicroAPI::MaskReg fullMask;

    MicroAPI::Duplicate(brcOneReg, 1u);
    MicroAPI::Duplicate(brcZeroReg, 0u);
    fullMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL, Trait>();
    if constexpr (sizeof(U) == 1) {
        MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_BRC_B8>(dupVreg, scalar);
    } else if constexpr (sizeof(U) == 2) {
        MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_BRC_B16>(dupVreg, scalar);
    } else if constexpr (sizeof(U) == 4) {
        MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_BRC_B32>(dupVreg, scalar);
    } else if constexpr (sizeof(U) == 8) {
        MicroAPI::LoadAlign(dupVreg, scalar);
        MicroAPI::Duplicate(dupVreg, dupVreg, fullMask);
    }        
    MicroAPI::CompareScalar<U, CMPMODE::NE>(cmpMask1, dupVreg, static_cast<U>(0), fullMask);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<U, Trait>(count);
        MicroAPI::LoadAlign(srcVreg, src + i * oneRepElm);
        MicroAPI::CompareScalar<U, CMPMODE::NE>(cmpMask0, srcVreg, static_cast<U>(0), mask);
        func(cmpMask2, cmpMask0, cmpMask1, mask);
        if constexpr (sizeof(U) == 2) {
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(mask, mask);
        } else if constexpr (sizeof(U) == 4 || sizeof(U) == 8) {
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(mask, mask);
            MicroAPI::MaskPack(mask, mask);
        }
        MicroAPI::Select(dstVreg, brcOneReg, brcZeroReg, cmpMask2);
        MicroAPI::StoreAlign(dst + i * oneRepElm, dstVreg, mask);
    }
}

template <auto func, typename T, typename U, typename RegT, typename RegU, const MicroAPI::RegTrait& Trait = MicroAPI::RegTraitNumOne>
__simd_vf__ inline void LogicalTemplateSingleScalarVF(__ubuf__ T* dst, __ubuf__ U* src, U scalar, 
    uint16_t repeatTime, uint32_t count, uint32_t oneRepElm)
{
    RegT dstVreg;
    RegT brcZeroReg;
    RegT brcOneReg;
    RegU srcVreg;
    RegU dupVreg;
    MicroAPI::MaskReg mask;
    MicroAPI::MaskReg fullMask;
    MicroAPI::MaskReg cmpMask0;
    MicroAPI::MaskReg cmpMask1;
    MicroAPI::MaskReg cmpMask2;

    MicroAPI::Duplicate(brcOneReg, 1u);
    MicroAPI::Duplicate(brcZeroReg, 0u);
    fullMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL, Trait>();
    MicroAPI::Duplicate(dupVreg, scalar);
    MicroAPI::CompareScalar<U, CMPMODE::NE>(cmpMask0, dupVreg, static_cast<U>(0), fullMask);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<U, Trait>(count);
        MicroAPI::LoadAlign(srcVreg, src + i * oneRepElm);
        MicroAPI::CompareScalar<U, CMPMODE::NE>(cmpMask1, srcVreg, static_cast<U>(0), mask);
        func(cmpMask2, cmpMask0, cmpMask1, mask);
        if constexpr (sizeof(U) == 2) {
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(mask, mask);
        } else if constexpr (sizeof(U) == 4 || sizeof(U) == 8) {
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(cmpMask2, cmpMask2);
            MicroAPI::MaskPack(mask, mask);
            MicroAPI::MaskPack(mask, mask);
        }
        MicroAPI::Select(dstVreg, brcOneReg, brcZeroReg, cmpMask2);
        MicroAPI::StoreAlign(dst + i * oneRepElm, dstVreg, mask);
    }
}

template <auto func, typename T, typename U, typename S, int8_t scalarTensorIndex>
__aicore__ inline void LogicalTemplateBothTensorCompute(const LocalTensor<T>& dst, const U& src0, const S& src1, const uint32_t count)
{
    using RegT = MicroAPI::RegTensor<T>;
    using ActualU = typename U::PrimType;
    using ActualS = typename S::PrimType;
    static_assert(SupportType<ActualU, bool, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, 
        uint32_t, int32_t, uint64_t, int64_t>(), "only support bool/uint8_t/int8_t/half/bfloat16_t/"
        "uint16_t/int16_t/float/uint32_t/int32_t/uint64_t/int64_t data type on current device!");
    static_assert(Std::is_same_v<ActualU, ActualS>, "The dataType ActualU and ActualS should be the same");
    static_assert((scalarTensorIndex == 0 || scalarTensorIndex == 1), "scalarTensorIndex out of range");
    if constexpr (sizeof(ActualU) == 8) {
        using RegU = MicroAPI::RegTensor<ActualU, MicroAPI::RegTraitNumTwo>;
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(ActualU) * LOGICAL_TEMPLATE_B64_REPEAT_STRIDE);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        if constexpr (scalarTensorIndex == 0) {
            LogicalTemplateBothTensorVF<func, T, ActualU, RegT, RegU, MicroAPI::RegTraitNumTwo>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ ActualU*)src1.GetPhyAddr(),
                (__ubuf__ ActualU*)src0.GetPhyAddr(), repeatTime, count, oneRepElm);
        } else {
            LogicalTemplateBothTensorVF<func, T, ActualU, RegT, RegU, MicroAPI::RegTraitNumTwo>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ ActualU*)src0.GetPhyAddr(),
                (__ubuf__ ActualU*)src1.GetPhyAddr(), repeatTime, count, oneRepElm);
        }
    } else {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(ActualU));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        if constexpr (Std::is_same_v<ActualU, bool>) {
            using RegU = MicroAPI::RegTensor<uint8_t>;
            if constexpr (scalarTensorIndex == 0) {
                LogicalTemplateBothTensorVF<func, T, uint8_t, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ uint8_t*)src1.GetPhyAddr(),
                    (__ubuf__ uint8_t*)src0.GetPhyAddr(), repeatTime, count, oneRepElm);
            } else {
                LogicalTemplateBothTensorVF<func, T, uint8_t, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ uint8_t*)src0.GetPhyAddr(),
                    (__ubuf__ uint8_t*)src1.GetPhyAddr(), repeatTime, count, oneRepElm);
            }
        } else {
            using RegU = MicroAPI::RegTensor<ActualU>;
            if constexpr (scalarTensorIndex == 0) {
                LogicalTemplateBothTensorVF<func, T, ActualU, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ ActualU*)src1.GetPhyAddr(),
                    (__ubuf__ ActualU*)src0.GetPhyAddr(), repeatTime, count, oneRepElm);
            } else {
                LogicalTemplateBothTensorVF<func, T, ActualU, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), (__ubuf__ ActualU*)src0.GetPhyAddr(),
                    (__ubuf__ ActualU*)src1.GetPhyAddr(), repeatTime, count, oneRepElm);
            }
        }
    }
}

template <auto func, typename T, typename U, typename S>
__aicore__ inline void LogicalTemplateTensorScalarCompute(const LocalTensor<T>& dst, const U& src0, const S& src1, const uint32_t count)
{
    using RegT = MicroAPI::RegTensor<T>;
    using ActualU = typename U::PrimType;
    static_assert(SupportType<ActualU, bool, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, 
        uint32_t, int32_t, uint64_t, int64_t>(), "only support bool/uint8_t/int8_t/half/bfloat16_t/"
        "uint16_t/int16_t/float/uint32_t/int32_t/uint64_t/int64_t data type on current device!");
    static_assert(Std::is_same_v<ActualU, S>, "The dataType ActualU and S should be the same");
    if constexpr (sizeof(ActualU) == 8) {
        using RegU = MicroAPI::RegTensor<ActualU, MicroAPI::RegTraitNumTwo>;
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(ActualU) * LOGICAL_TEMPLATE_B64_REPEAT_STRIDE);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        LogicalTemplateSingleScalarVF<func, T, ActualU, RegT, RegU, MicroAPI::RegTraitNumTwo>((__ubuf__ T*)dst.GetPhyAddr(), 
            (__ubuf__ ActualU*)src0.GetPhyAddr(), src1, repeatTime, count, oneRepElm);
    } else {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(ActualU));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        if constexpr (Std::is_same_v<ActualU, bool>) {
            using RegU = MicroAPI::RegTensor<uint8_t>;
            LogicalTemplateSingleScalarVF<func, T, uint8_t, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), 
                (__ubuf__ uint8_t*)src0.GetPhyAddr(), static_cast<uint8_t>(src1), repeatTime, count, oneRepElm);
        } else {
            using RegU = MicroAPI::RegTensor<ActualU>;
            LogicalTemplateSingleScalarVF<func, T, ActualU, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), 
                (__ubuf__ ActualU*)src0.GetPhyAddr(), src1, repeatTime, count, oneRepElm);
        }
    }
}

template <auto func, typename T, typename U, typename S>
__aicore__ inline void LogicalTemplateScalarTensorCompute(const LocalTensor<T>& dst, const U& src0, const S& src1, const uint32_t count)
{
    using RegT = MicroAPI::RegTensor<T>;
    using ActualS = typename S::PrimType;
    static_assert(SupportType<U, bool, uint8_t, int8_t, half, bfloat16_t, uint16_t, int16_t, float, 
        uint32_t, int32_t, uint64_t, int64_t>(), "only support bool/uint8_t/int8_t/half/bfloat16_t/"
        "uint16_t/int16_t/float/uint32_t/int32_t/uint64_t/int64_t data type on current device!");
    static_assert(Std::is_same_v<ActualS, U>, "The dataType S and ActualU should be the same");
    if constexpr (sizeof(U) == 8) {
        using RegU = MicroAPI::RegTensor<U, MicroAPI::RegTraitNumTwo>;
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(U) * LOGICAL_TEMPLATE_B64_REPEAT_STRIDE);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        LogicalTemplateSingleScalarVF<func, T, U, RegT, RegU, MicroAPI::RegTraitNumTwo>((__ubuf__ T*)dst.GetPhyAddr(), 
                (__ubuf__ U*)src1.GetPhyAddr(), src0, repeatTime, count, oneRepElm);
    } else {
        constexpr uint32_t oneRepElm = static_cast<uint32_t>(GetVecLen() / sizeof(U));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, oneRepElm));
        if constexpr (Std::is_same_v<U, bool>) {
            using RegU = MicroAPI::RegTensor<uint8_t>;
            LogicalTemplateSingleScalarVF<func, T, uint8_t, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), 
                (__ubuf__ uint8_t*)src1.GetPhyAddr(), static_cast<uint8_t>(src0), repeatTime, count, oneRepElm);
        } else {
            using RegU = MicroAPI::RegTensor<U>;
            LogicalTemplateSingleScalarVF<func, T, U, RegT, RegU>((__ubuf__ T*)dst.GetPhyAddr(), 
                (__ubuf__ U*)src1.GetPhyAddr(), src0, repeatTime, count, oneRepElm);
        }
    }
}

template <auto func, typename T, typename U, typename S, int8_t scalarTensorIndex>
__aicore__ inline void LogicalTemplateScalarImpl(const LocalTensor<T>& dst, const U& src0, const S& src1, const uint32_t count)
{
    static_assert(SupportType<T, bool>(), "only support bool data type on current device!");
    static_assert(!TypeUtils::IsInnerDefaultType<U, S>(), "One of src0 and src1 should be Tensor");
    using RegT = MicroAPI::RegTensor<T>;
    if constexpr (Std::is_same_v<S, U>) {
        LogicalTemplateBothTensorCompute<func, T, U, S, scalarTensorIndex>(dst, src0, src1, count);
    } else if constexpr (TypeUtils::IsLocalTensorType<U>() && TypeUtils::IsInnerDefaultType<S>()) {
        LogicalTemplateTensorScalarCompute<func, T, U, S>(dst, src0, src1, count);
    } else if constexpr (TypeUtils::IsLocalTensorType<S>() && TypeUtils::IsInnerDefaultType<U>()) {
        LogicalTemplateScalarTensorCompute<func, T, U, S>(dst, src0, src1, count);
    }
}

}
#endif
#endif  // IMPL_MATH_LOGICAL_TEMPLATE_LOGICAL_TEMPLATE_IMPL_H
