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
 * \file fmod_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_FMOD_FMOD_C310_IMPL_H
#define IMPL_MATH_FMOD_FMOD_C310_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"
 
namespace AscendC {
namespace FmodInternal {
union FloatS32Union {
    constexpr __aicore__ FloatS32Union(): f(0.0f) {}
    constexpr __aicore__ FloatS32Union(int32_t val): i(val) {}
    float f;
    int32_t i;
};
union FloatU32Union {
    constexpr __aicore__ FloatU32Union(): f(0.0f) {}
    constexpr __aicore__ FloatU32Union(uint32_t val): i(val) {}
    float f;
    uint32_t i;
};
constexpr FloatU32Union inf(F32_INF);
constexpr FloatU32Union negInf(F32_NEG_INF);
constexpr FloatU32Union nan(F32_NAN);
constexpr uint16_t oneRepSize = static_cast<uint16_t>(GetVecLen() / sizeof(float));

constexpr MicroAPI::CastTrait castTraitS322F32 = {
    MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, 
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT
};

constexpr FloatS32Union scaleList1[FMOD_ITERATION_NUM_MAX] = {
    FloatS32Union(0x4b800000), FloatS32Union(0x4b800000), FloatS32Union(0x57800000),
    FloatS32Union(0x63800000), FloatS32Union(0x6f800000), FloatS32Union(0x7b800000), 
    FloatS32Union(0x7b800000), FloatS32Union(0x7b800000), FloatS32Union(0x7b800000), 
    FloatS32Union(0x7b800000), FloatS32Union(0x7b800000)
};
constexpr FloatS32Union scaleList2[FMOD_ITERATION_NUM_MAX] = {
    FloatS32Union(0x3f800000), FloatS32Union(0x3f800000), FloatS32Union(0x3f800000),
    FloatS32Union(0x3f800000), FloatS32Union(0x3f800000), FloatS32Union(0x3f800000), 
    FloatS32Union(0x4b800000), FloatS32Union(0x57800000), FloatS32Union(0x63800000), 
    FloatS32Union(0x6f800000), FloatS32Union(0x7b800000)
};

template <typename T>
__simd_callee__ inline void LoadDataWithT(
    __local_mem__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> srcOrigin;
        DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcOrigin, src + offset);
        Cast<float, T, layoutZMrgZ>(dstReg, srcOrigin, mask);
    } else {
        DataCopy(dstReg, src + offset);
    }
}

template <typename T>
__simd_callee__ inline void SaveDataWithT(
    __local_mem__ T* dst, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> regT;
        MicroAPI::Cast<T, float, LayoutZMrgZRndRSatNS>(regT, srcReg, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, regT, mask);
    } else {
        MicroAPI::DataCopy(dst + offset, srcReg, mask);
    }
}

__simd_callee__ inline void GetSignBit(
    MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask)
{
    constexpr int16_t signRightNum = 31;
    MicroAPI::RegTensor<uint32_t> oneReg;
    MicroAPI::RegTensor<uint32_t> tmpReg;
    MicroAPI::Duplicate(oneReg, 1, mask);
    MicroAPI::ShiftRights(tmpReg, (MicroAPI::RegTensor<uint32_t>&)srcReg, signRightNum, mask);
    MicroAPI::And(tmpReg, tmpReg, oneReg, mask);
    MicroAPI::Cast<float, int32_t, FmodInternal::castTraitS322F32>(dstReg, (MicroAPI::RegTensor<int32_t>&)tmpReg, mask);
}

template <int32_t iterationNum>
__simd_callee__ inline void SolveScale(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& src1Reg,
    const float scale1, const float scale2, MicroAPI::MaskReg& mask)
{
    constexpr float maxValue = 3.4028235e38;
    constexpr float subnormal = 1.1754944e-38;

    MicroAPI::MaskReg subnormalMask;
    MicroAPI::RegTensor<float> bTmpReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> kReg;
    MicroAPI::RegTensor<float> signReg;

    if constexpr (iterationNum == 1) { // iter 1 (last iteration) handles subnormal case
        MicroAPI::CompareScalar<float, CMPMODE::LE>(subnormalMask, src1Reg, subnormal, mask);
        MicroAPI::Muls(tmpReg, src1Reg, scale1, subnormalMask);
        MicroAPI::Select(bTmpReg, tmpReg, src1Reg, subnormalMask);
        MicroAPI::Muls(tmpReg, dstReg, scale1, subnormalMask);
        MicroAPI::Select(dstReg, tmpReg, dstReg, subnormalMask);

        MicroAPI::Div(kReg, dstReg, bTmpReg, mask);
        MicroAPI::Truncate<float, RoundMode::CAST_RINT>(kReg, kReg, mask);
    } else {
        MicroAPI::Muls(bTmpReg, src1Reg, scale1, mask);
        if constexpr (iterationNum > 5) { // last 5 iterations do not need extra scaling
            MicroAPI::Muls(bTmpReg, bTmpReg, scale2, mask);
        }
        MicroAPI::Div(kReg, dstReg, bTmpReg, mask);
        MicroAPI::Truncate<float, RoundMode::CAST_ROUND>(kReg, kReg, mask);
    }

    // not necessary to check for inf in the final iteration
    if constexpr (iterationNum != 1) {
        MicroAPI::Mins(bTmpReg, bTmpReg, maxValue, mask);
    }
    MicroAPI::Neg(kReg, kReg, mask);
    // res = -k * bTmp + y
    MicroAPI::MulAddDst(dstReg, kReg, bTmpReg, mask);

    if constexpr (iterationNum == 1) { // iter 1 handles subnormal case
        // r = r + np.float32(np.signbit(r)) * btmp
        GetSignBit(signReg, dstReg, mask);
        MicroAPI::Mul(signReg, signReg, bTmpReg, mask);
        MicroAPI::Add(dstReg, dstReg, signReg, mask);
    }
}

// recurse from itermax to 1
template <int32_t iterationNum>
__simd_callee__ inline void SolveScaleIter (
    MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& src1Reg, MicroAPI::MaskReg& mask)
{
    SolveScale<iterationNum>(dstReg, src1Reg, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, mask);

    if constexpr (iterationNum > 1) {
        SolveScaleIter<iterationNum - 1>(dstReg, src1Reg, mask);
    }
}

template <int32_t iterationNum>
__simd_callee__ inline void SolveScale(__local_mem__ float* dst, __local_mem__ float* src, const uint16_t unitRepTimes,
    const float scale1, const float scale2, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> src1OriginReg;
    MicroAPI::RegTensor<float> src1Reg;
    MicroAPI::RegTensor<float> dstReg;
    for (uint16_t i = 0; i < unitRepTimes; i++) {
        MicroAPI::DataCopy(src1OriginReg, src + i * FmodInternal::oneRepSize);
        MicroAPI::DataCopy(dstReg, dst + i * FmodInternal::oneRepSize);
        MicroAPI::Abs(src1Reg, src1OriginReg, mask);
        SolveScale<iterationNum>(dstReg, src1Reg, scale1, scale2, mask);
        MicroAPI::DataCopy(dst + i * FmodInternal::oneRepSize, dstReg, mask);
    }
}

template <int32_t iterationNum>
__simd_callee__ inline void SolveScaleInit(__local_mem__ float* dst, __local_mem__ float* src0, __local_mem__ float* src1, 
    const uint16_t unitRepTimes, const float scale1, const float scale2, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> src0OriginReg;
    MicroAPI::RegTensor<float> src1OriginReg;
    MicroAPI::RegTensor<float> src1Reg;
    MicroAPI::RegTensor<float> dstReg;
    for (uint16_t i = 0; i < unitRepTimes; i++) {
        MicroAPI::DataCopy(src0OriginReg, src0 + i * FmodInternal::oneRepSize);
        MicroAPI::DataCopy(src1OriginReg, src1 + i * FmodInternal::oneRepSize);
        MicroAPI::Abs(dstReg, src0OriginReg, mask);
        MicroAPI::Abs(src1Reg, src1OriginReg, mask);
        SolveScale<iterationNum>(dstReg, src1Reg, scale1, scale2, mask);
        MicroAPI::DataCopy(dst + i * FmodInternal::oneRepSize, dstReg, mask);
    }
}

template <int32_t iterationNum, int32_t totalIterationNum>
__simd_callee__ inline void SolveScaleIterImpl(
    __local_mem__ float* dst, __local_mem__ float* src0, __local_mem__ float* src1, 
    const uint16_t unitRepTimes, MicroAPI::MaskReg& mask)
{
    if (iterationNum == totalIterationNum) { // first iteration, initialization
        SolveScaleInit<iterationNum>(dst, src0, src1, unitRepTimes, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, mask);
    } else {
        SolveScale<iterationNum>(dst, src1, unitRepTimes, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, mask);
    }

    if constexpr (iterationNum > 1) {
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        SolveScaleIterImpl<iterationNum - 1, totalIterationNum>(dst, src0, src1, unitRepTimes, mask);
    }
}

template <int32_t iterationNum>
__simd_callee__ inline void SolveScaleIter(__local_mem__ float* dst, __local_mem__ float* src0, __local_mem__ float* src1, 
    const uint16_t unitRepTimes, MicroAPI::MaskReg& mask)
{
    SolveScaleIterImpl<iterationNum, iterationNum>(dst, src0, src1, unitRepTimes, mask);
}
    

__simd_callee__ inline void SolveExceptionScenarios(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& src0Reg,
    MicroAPI::RegTensor<float>& src1Reg, MicroAPI::RegTensor<float> nanReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::MaskReg src0Is0CmpReg;
    MicroAPI::MaskReg src0IsNeg0CmpReg;
    MicroAPI::MaskReg src0InfCmpReg;
    MicroAPI::MaskReg src0NegInfCmpReg;
    MicroAPI::MaskReg src1InfCmpReg;
    MicroAPI::MaskReg src1NegInfCmpReg;
    MicroAPI::MaskReg srcBothInfCmpReg;
    MicroAPI::MaskReg src1Not0CmpReg;
    MicroAPI::MaskReg src1NotNeg0CmpReg;
    MicroAPI::MaskReg src1NotNanCmpReg;
    // if src1Tensor is inf return src0Reg
    MicroAPI::CompareScalar(src1InfCmpReg, src1Reg, inf.f, mask);
    MicroAPI::Select(dstReg, src0Reg, dstReg, src1InfCmpReg);
    // if src1Tensor is -inf return src0Reg
    MicroAPI::CompareScalar(src1NegInfCmpReg, src1Reg, negInf.f, mask);
    MicroAPI::Select(dstReg, src0Reg, dstReg, src1NegInfCmpReg);
    // if src0Tensor is inf
    MicroAPI::CompareScalar(src0InfCmpReg, src0Reg, inf.f, mask);
    // if src0Tensor is -inf
    MicroAPI::CompareScalar(src0NegInfCmpReg, src0Reg, negInf.f, mask);
    // if src0Tensor and src1Tensor both inf return inf
    MicroAPI::MaskOr(src0InfCmpReg, src0InfCmpReg, src0NegInfCmpReg, mask);
    MicroAPI::MaskOr(src1InfCmpReg, src1InfCmpReg, src1NegInfCmpReg, mask);
    MicroAPI::MaskAnd(srcBothInfCmpReg, src0InfCmpReg, src1InfCmpReg, mask);
    MicroAPI::Select(dstReg, nanReg, dstReg, srcBothInfCmpReg);
    // if src0Tensor is ±0 and src1Tensor is not ±0 and not nan, return src0Tensor
    MicroAPI::CompareScalar(src0Is0CmpReg, src0Reg, static_cast<float>(0), mask);
    MicroAPI::CompareScalar(src0IsNeg0CmpReg, src0Reg, static_cast<float>(-0), mask);
    MicroAPI::MaskOr(src0Is0CmpReg, src0Is0CmpReg, src0IsNeg0CmpReg, mask);

    MicroAPI::CompareScalar<float, CMPMODE::NE>(src1Not0CmpReg, src1Reg, static_cast<float>(0), mask);
    MicroAPI::CompareScalar<float, CMPMODE::NE>(src1NotNeg0CmpReg, src1Reg, static_cast<float>(-0), mask);
    MicroAPI::Compare<float, CMPMODE::NE>(src1NotNanCmpReg, src1Reg, src1Reg, mask);
    MicroAPI::MaskNot(src1NotNanCmpReg, src1NotNanCmpReg, mask);

    MicroAPI::MaskOr(src1Not0CmpReg, src1Not0CmpReg, src1NotNeg0CmpReg, mask);
    MicroAPI::MaskAnd(src1Not0CmpReg, src1Not0CmpReg, src1NotNanCmpReg, mask);

    MicroAPI::MaskAnd(src0Is0CmpReg, src0Is0CmpReg, src1Not0CmpReg, mask);
    MicroAPI::Select(dstReg, src0Reg, dstReg, src0Is0CmpReg);
}
} // namespace FmodInternal

template <int32_t iterationNum>
__simd_vf__ inline void FmodComputeIterationF32(__local_mem__ float* dstTensor, __local_mem__ float* src0Tensor,
    __local_mem__ float* src1Tensor, const uint16_t mainRepeatTimes, const uint16_t mainBlockLen,
    const uint16_t tailRepeatTimes, uint32_t tailCount)
{
    constexpr FmodInternal::FloatU32Union scale1(0x4B800000); // 2**24
    constexpr FmodInternal::FloatU32Union scale2(0x33800000); // 2**-24
    constexpr float subnormal = 1.1754944e-38;
    MicroAPI::RegTensor<float> src0OriginReg;
    MicroAPI::RegTensor<float> src1OriginReg;
    MicroAPI::RegTensor<float> src1Reg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> nanReg;
    MicroAPI::RegTensor<float> zeroReg;
    MicroAPI::RegTensor<float> n2Reg;
    MicroAPI::RegTensor<float> oneReg;
    MicroAPI::RegTensor<float> src0SignBitReg;
    MicroAPI::RegTensor<float> src0SignBitTmpReg;
    MicroAPI::RegTensor<float> dstSignBitReg;
    MicroAPI::RegTensor<float> bTmpReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg subnormalMask;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(nanReg, FmodInternal::nan.f, maskFull);
    MicroAPI::Duplicate(zeroReg, static_cast<float>(0.0), maskFull);
    MicroAPI::Duplicate(n2Reg, static_cast<float>(-2.0), maskFull);
    MicroAPI::Duplicate(oneReg, static_cast<float>(1), maskFull);

    FmodInternal::SolveScaleIter<iterationNum>(dstTensor, src0Tensor, src1Tensor, mainRepeatTimes, maskFull);

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < mainRepeatTimes; i++) {
        FmodInternal::LoadDataWithT(src0Tensor, src0OriginReg, maskFull, i * FmodInternal::oneRepSize);
        FmodInternal::LoadDataWithT(src1Tensor, src1OriginReg, maskFull, i * FmodInternal::oneRepSize);
        MicroAPI::Abs(src1Reg, src1OriginReg, maskFull);
        MicroAPI::DataCopy(dstReg, dstTensor + i * FmodInternal::oneRepSize);

        // res = res*(np.float32(asignbit)*np.float32(-2.0) + np.float32(1))
        FmodInternal::GetSignBit(src0SignBitReg, src0OriginReg, maskFull);
        MicroAPI::Mul(src0SignBitTmpReg, src0SignBitReg, n2Reg, maskFull);
        MicroAPI::Add(src0SignBitTmpReg, src0SignBitTmpReg, oneReg, maskFull);
        MicroAPI::Mul(dstReg, dstReg, src0SignBitTmpReg, maskFull);

        MicroAPI::CompareScalar<float, CMPMODE::LE>(subnormalMask, src1Reg, subnormal, maskFull);
        MicroAPI::Muls(tmpReg, src1Reg, scale1.f, subnormalMask);
        MicroAPI::Select(bTmpReg, tmpReg, src1Reg, subnormalMask);

        MicroAPI::Muls(tmpReg, dstReg, scale2.f, subnormalMask);
        MicroAPI::Select(dstReg, tmpReg, dstReg, subnormalMask);

        FmodInternal::SolveExceptionScenarios(dstReg, src0OriginReg, src1OriginReg, nanReg, maskFull);
        FmodInternal::SaveDataWithT(dstTensor, dstReg, maskFull, i * FmodInternal::oneRepSize);
    }

    for (uint16_t i = 0; i < tailRepeatTimes; i++) {
        maskReg = MicroAPI::UpdateMask<float>(tailCount);
        FmodInternal::LoadDataWithT(src0Tensor, src0OriginReg, maskReg, mainBlockLen + i * FmodInternal::oneRepSize);
        FmodInternal::LoadDataWithT(src1Tensor, src1OriginReg, maskReg, mainBlockLen + i * FmodInternal::oneRepSize);

        MicroAPI::Abs(dstReg, src0OriginReg, maskReg);
        MicroAPI::Abs(src1Reg, src1OriginReg, maskReg);
        FmodInternal::SolveScaleIter<iterationNum>(dstReg, src1Reg, maskReg);

        // res = res*(np.float32(asignbit)*np.float32(-2.0) + np.float32(1))
        FmodInternal::GetSignBit(src0SignBitReg, src0OriginReg, maskReg);
        MicroAPI::Mul(src0SignBitTmpReg, src0SignBitReg, n2Reg, maskReg);
        MicroAPI::Add(src0SignBitTmpReg, src0SignBitTmpReg, oneReg, maskReg);
        MicroAPI::Mul(dstReg, dstReg, src0SignBitTmpReg, maskReg);

        MicroAPI::CompareScalar<float, CMPMODE::LE>(subnormalMask, src1Reg, subnormal, maskReg);
        MicroAPI::Muls(tmpReg, src1Reg, scale1.f, subnormalMask);
        MicroAPI::Select(bTmpReg, tmpReg, src1Reg, subnormalMask);
        MicroAPI::Muls(tmpReg, dstReg, scale2.f, subnormalMask);
        MicroAPI::Select(dstReg, tmpReg, dstReg, subnormalMask);

        FmodInternal::SolveExceptionScenarios(dstReg, src0OriginReg, src1OriginReg, nanReg, maskReg);
        FmodInternal::SaveDataWithT(dstTensor, dstReg, maskReg, mainBlockLen + i * FmodInternal::oneRepSize);
    }
}

template <int32_t iterationNum>
__aicore__ inline void FmodComputeIteration(const LocalTensor<float>& dstTensor, const LocalTensor<float>& src0Tensor,
    const LocalTensor<float>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t count)
{
    __local_mem__ float* src0 = (__local_mem__ float*)src0Tensor.GetPhyAddr();
    __local_mem__ float* src1 = (__local_mem__ float*)src1Tensor.GetPhyAddr();
    __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
    
    const uint16_t mainRepeatTimes = static_cast<uint16_t>(count / FmodInternal::oneRepSize);
    const uint16_t mainBlockLen = mainRepeatTimes * FmodInternal::oneRepSize;

    uint32_t tailCount = count - mainBlockLen;
    const uint16_t tailRepeatTimes = static_cast<uint16_t>(CeilDivision(tailCount, FmodInternal::oneRepSize));

    FmodComputeIterationF32<iterationNum>(
        dst, src0, src1, mainRepeatTimes, mainBlockLen, tailRepeatTimes, tailCount);
}

template <typename T>
__simd_vf__ inline void FmodComputeVF(__local_mem__ T* dstTensor, __local_mem__ T* src0Tensor, __local_mem__ T* src1Tensor, 
    const uint16_t repeatTimes, uint32_t count)
{
    MicroAPI::RegTensor<float> src0Reg;
    MicroAPI::RegTensor<float> src1Reg;
    MicroAPI::RegTensor<float> negReg;
    MicroAPI::RegTensor<float> divReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> nanReg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(nanReg, FmodInternal::nan.f, maskFull);

    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg = MicroAPI::UpdateMask<float>(count);
        FmodInternal::LoadDataWithT<T>(src0Tensor, src0Reg, maskReg, i * FmodInternal::oneRepSize);
        FmodInternal::LoadDataWithT<T>(src1Tensor, src1Reg, maskReg, i * FmodInternal::oneRepSize);
        MicroAPI::Div(divReg, src0Reg, src1Reg, maskReg);
        MicroAPI::Truncate<float, RoundMode::CAST_TRUNC, MicroAPI::MaskMergeMode::ZEROING>(dstReg, divReg, maskReg);
        MicroAPI::Neg(negReg, src1Reg, maskReg);
        MicroAPI::FusedMulDstAdd(dstReg, negReg, src0Reg, maskReg);
        FmodInternal::SolveExceptionScenarios(dstReg, src0Reg, src1Reg, nanReg, maskReg);
        FmodInternal::SaveDataWithT(dstTensor, dstReg, maskReg, i * FmodInternal::oneRepSize);
    }
}
 
__aicore__ inline void FmodCompute(const LocalTensor<float> &dstTensor, const LocalTensor<float> &src0Tensor,
    const LocalTensor<float> &src1Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t count)
{
    __local_mem__ float *src0 = (__local_mem__ float *)src0Tensor.GetPhyAddr();
    __local_mem__ float *src1 = (__local_mem__ float *)src1Tensor.GetPhyAddr();
    __local_mem__ float *dst = (__local_mem__ float *)dstTensor.GetPhyAddr();
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, FmodInternal::oneRepSize));
 
    FmodComputeVF<float>(dst, src0, src1, repeatTimes, count);
}
 
__aicore__ inline void FmodCompute(const LocalTensor<half> &dstTensor, const LocalTensor<half> &src0Tensor,
    const LocalTensor<half> &src1Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t count)
{
    __local_mem__ half *src0 = (__local_mem__ half *)src0Tensor.GetPhyAddr();
    __local_mem__ half *src1 = (__local_mem__ half *)src1Tensor.GetPhyAddr();
    __local_mem__ half *dst = (__local_mem__ half *)dstTensor.GetPhyAddr();
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, FmodInternal::oneRepSize));
 
    FmodComputeVF<half>(dst, src0, src1, repeatTimes, count);
}
 
template <typename T, bool isReuseSource = false, const FmodConfig& config = DEFAULT_FMOD_CONFIG>
__aicore__ inline void FmodImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor,
    const LocalTensor<T> &src1Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
 
    CheckTensorPos(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECOUT / VECCALC", "Fmod");
    CheckTensorPos(src0Tensor, Hardware::UB, "src0Tensor", "VECIN / VECOUT / VECCALC", "Fmod");
    CheckTensorPos(src1Tensor, Hardware::UB, "src1Tensor", "VECIN / VECOUT / VECCALC", "Fmod");
    CheckTensorPos(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECOUT / VECCALC", "Fmod");
 
    CheckCalCount(calCount, "calCount", src0Tensor, "src0Tensor", "Fmod");
    CheckCalCount(calCount, "calCount", src1Tensor, "src1Tensor", "Fmod");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Fmod");
 
 
    ASCENDC_ASSERT((src0Tensor.GetSize() == src1Tensor.GetSize()),
                   { KERNEL_LOG(KERNEL_ERROR, "Input params.GetSize must be equal with each other!"); });

    if constexpr (config.algo == FmodAlgo::ITERATION_COMPENSATION) {
        static_assert(config.iterationNum >= 1 && config.iterationNum <= FMOD_ITERATION_NUM_MAX,
            "Iteration number must be in the range [1, 11].");
        static_assert(SupportType<T, float>(), "current data type is not supported on current device!");
        FmodComputeIteration<config.iterationNum>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    } else {
        static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
        FmodCompute(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    }
}
} // namespace AscendC
#endif // IMPL_MATH_FMOD_FMOD_C310_IMPL_H