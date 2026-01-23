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
 * \file sin_l300_impl.h
 * \brief
 */
#ifndef LIB_MATH_SIN_L300_IMPL_H
#define LIB_MATH_SIN_L300_IMPL_H
#include "kernel_tensor.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)
#include "sin_common_utils.h"
#endif

namespace AscendC {
namespace MicroAPI {
namespace Sin {
const uint8_t SIN_FLOAT_NOREUSE_CALC_PROCEDURE = 3;
const uint8_t SIN_FLOAT_REUSE_CALC_PROCEDURE = 2;

// define the number of x div pi
constexpr float SIN_PI_FOR_X_TODIV = 0.3183098733425140380859375;
// define the PI for compute
constexpr float SIN_PI_V2 = 3.140625;
constexpr float SIN_KPI_FIRS_PI_MULS = 0.0009670257568359375;
constexpr float SIN_KPI_TWI_PI_MULS = 6.2771141529083251953125e-7;
constexpr float SIN_KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10;
// define the number of sin compute
constexpr float SIN_RES_MULIT_SCA = 2.604926501e-6;
constexpr float SIN_RES_ADDICT_UP = -0.0001980894471;
constexpr float SIN_2ADDS = 0.008333049340;
constexpr float SIN_3ADDS = -0.1666665792;
constexpr float SIN_POINT_FIVE = 0.5;
constexpr float SIN_M4_SCA = 4.0;
constexpr float SIN_K2_SCA = -2.0;
constexpr float SIN_SCALAR_ONE = 1.0;
constexpr MicroAPI::CastTrait sinCastTraitF16F32 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                     MicroAPI::MaskMergeMode::ZEROING };
constexpr MicroAPI::CastTrait sinCastTraitF32F16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                     MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

__simd_callee__ inline void SinPolynomialApproximation(MicroAPI::RegTensor<float> &dstReg,
    MicroAPI::RegTensor<float> &srcReg, MicroAPI::RegTensor<float> &x, MicroAPI::RegTensor<float> &round,
    MicroAPI::RegTensor<float> &kpi, MicroAPI::MaskReg mask)
{
    // normalized x to [-π/2, π/2] using x = x-round(x/π)*π
    // k = round(x * invpi)
    MicroAPI::Muls(round, srcReg, SIN_PI_FOR_X_TODIV, mask);
    MicroAPI::Truncate<float, RoundMode::CAST_RINT, MicroAPI::MaskMergeMode::ZEROING>(round, round, mask);
    // x -= k * pi_0
    MicroAPI::Muls(kpi, round, SIN_PI_V2, mask);
    MicroAPI::Sub(x, srcReg, kpi, mask);
    // x -= k * pi_1
    MicroAPI::Muls(kpi, round, SIN_KPI_FIRS_PI_MULS, mask);
    MicroAPI::Sub(x, x, kpi, mask);
    // x -= k * pi_2
    MicroAPI::Muls(kpi, round, SIN_KPI_TWI_PI_MULS, mask);
    MicroAPI::Sub(x, x, kpi, mask);
    // x -= k * pi_3
    MicroAPI::Muls(kpi, round, SIN_KPI_THIR_PI_MULS, mask);
    MicroAPI::Sub(x, x, kpi, mask);

    /*
    sin(x) = (-1)^k*sin(x0)
    Finally, use sin(x) = xP(x) to calculate sin(x).
    P(x) = (((x^2 * R0 + R1) * x^2 + R2) * x^2 + R3) * x^2 + 1.0
    */
    // x^2 = mul(input_x, input_x)
    MicroAPI::Mul(kpi, x, x, mask);
    // kover2
    MicroAPI::Muls(dstReg, round, SIN_POINT_FIVE, mask);
    MicroAPI::Truncate<float, RoundMode::CAST_FLOOR, MicroAPI::MaskMergeMode::ZEROING>(dstReg, dstReg, mask);
    // kover2floorm4
    MicroAPI::Muls(dstReg, dstReg, SIN_M4_SCA, mask);
    // k2
    MicroAPI::Muls(round, round, SIN_K2_SCA, mask);
    // sign
    MicroAPI::Add(dstReg, dstReg, round, mask);
    MicroAPI::Adds(dstReg, dstReg, SIN_SCALAR_ONE, mask);
    // res_up = mul(x^2, 2.604926501e-6)
    MicroAPI::Muls(round, kpi, SIN_RES_MULIT_SCA, mask);
    MicroAPI::Adds(round, round, SIN_RES_ADDICT_UP, mask);
    // res_up = mul(res_up, x^2)
    MicroAPI::Mul(round, round, kpi, mask);
    MicroAPI::Adds(round, round, SIN_2ADDS, mask);
    // res_up = mul(res_up, x^2)
    MicroAPI::Mul(round, round, kpi, mask);
    MicroAPI::Adds(round, round, SIN_3ADDS, mask);
    // res_up = mul(res_up, x^2)
    MicroAPI::Mul(round, round, kpi, mask);
    MicroAPI::Adds(round, round, SIN_SCALAR_ONE, mask);
    // res_up = mul(res_up, input_x)
    MicroAPI::Mul(round, round, x, mask);
    MicroAPI::Mul(dstReg, round, dstReg, mask);
}

template <typename T>
__simd_vf__ inline void SinPolynomial(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount, uint16_t repeat)
{
    MicroAPI::RegTensor<T> x;
    MicroAPI::RegTensor<float> xTmp;
    MicroAPI::RegTensor<float> round;
    MicroAPI::RegTensor<float> kpi;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> srcTmp;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<float> dstTmp;
    MicroAPI::MaskReg mask;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(float);
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
    for (uint16_t i = 0; i < (uint16_t)repeat; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        if constexpr (std::is_same<T, half>::value) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * oneRepSize);
            MicroAPI::Cast<float, half, sinCastTraitF16F32>(srcTmp, srcReg, mask);
            SinPolynomialApproximation(dstTmp, srcTmp, xTmp, round, kpi, mask);
            MicroAPI::Cast<half, float, sinCastTraitF32F16>(dstReg, dstTmp, mask);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstReg, mask);
        } else {
            MicroAPI::DataCopy(srcReg, src + i * oneRepSize);
            SinPolynomialApproximation(dstReg, srcReg, xTmp, round, kpi, mask);
            MicroAPI::DataCopy(dst + i * oneRepSize, dstReg, mask);
        }
    }
}

template <typename T> __aicore__ inline void SinPolynomialImpl(__ubuf__ T *dst, __ubuf__ T *src, uint32_t calCount)
{
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(float);
    uint16_t repeat = CeilDivision(calCount, oneRepSize);
    SinPolynomial<T>(dst, src, calCount, repeat);
}
} // namespace MicroAPI
} // namespace Sin

__aicore__ inline constexpr uint32_t GetSinTmpBufferLiveNode() {
    constexpr uint32_t tmpBufferLiveNode = sizeof(float) * 2;
    return tmpBufferLiveNode;
}

template <typename T, bool isReuseSource = false, const SinConfig& config = defaultSinConfig>
__aicore__ inline void SinImpl(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    // Using the Stack Space to Allocate tmpBuffer
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    SinImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

template<typename T>
__aicore__ inline uint32_t GetSinTmpBufferSize(const LocalTensor<uint8_t>& sharedTmpBuffer) {
    uint32_t sharedTmpBufferSize = sharedTmpBuffer.GetSize() / GetSinTmpBufferLiveNode();
    return AlignUp(sharedTmpBufferSize, GetDataBlockSizeInBytes()) / sizeof(T);
}

template <typename T, bool isReuseSource = false, const SinConfig &config = defaultSinConfig>
__aicore__ inline void SinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "Sin");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "Sin");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "Sin");
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not be larger than srcTensor length %u", calCount,
            srcTensor.GetSize());
    });
    ASCENDC_ASSERT((calCount <= dstTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not be larger than dstTensor length %u", calCount,
            dstTensor.GetSize());
    });

    if constexpr (config.algo == SinAlgo::POLYNOMIAL_APPROXIMATION) {
        MicroAPI::Sin::SinPolynomialImpl((__ubuf__ T *)dstTensor.GetPhyAddr(),
            (__ubuf__ T *)srcTensor.GetPhyAddr(), calCount);
    } else if constexpr (config.algo == SinAlgo::RADIAN_REDUCTION) {
        uint32_t sharedTmpBufferSize = GetSinTmpBufferSize<T>(sharedTmpBuffer);
        uint32_t count = calCount;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sharedTmpBufferSize));
        for (uint16_t i = 0; i < repeatTimes; i++) {
            uint32_t remainCount = count - sharedTmpBufferSize * i;
            uint32_t oneRepSize = remainCount < sharedTmpBufferSize ? remainCount : sharedTmpBufferSize;
            SinRadianReductionImpl((__ubuf__ T*)dstTensor.GetPhyAddr() + i * sharedTmpBufferSize,
                (__ubuf__ T*)srcTensor.GetPhyAddr() + i * sharedTmpBufferSize,
                (__ubuf__ uint32_t*)sharedTmpBuffer.GetPhyAddr(), oneRepSize);
        }
    }
}

__aicore__ inline void SinCastFullMask(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, RoundMode castType)
{
    uint64_t newMask = 64;
    Cast<float, float, false>(dstTensor, srcTensor, castType, newMask, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

 /*!
 * \ingroup Sin
 * \brief compute Sin elementwisely
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason, only support
 * float input data type
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer: extra temporary shared space used for intermediate values among calculation process,
 * whose required space size should refer to corresponding tiling API, which is defined at sin_tiling.h.
 * Generally, the more space you allocate, the better performance you will achieve, and the performance
 * reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it is not guaranteed
 * that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: the number of elements to be processed.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 * Input data valid range should be [-65504, 65504]
 */
template<typename T, bool isReuseSource = false, const SinConfig &config = defaultSinConfig>
__aicore__ inline void Sin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    SinImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

 /*!
 * \ingroup Sin
 * \brief compute Sin elementwisely
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer: extra temporary shared space used for intermediate values among calculation process,
 * whose required space size should refer to corresponding tiling API, which is defined at sin_tiling.h.
 * Generally, the more space you allocate, the better performance you will achieve, and the performance
 * reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it is not guaranteed
 * that the shared space will be cleared after usage, the data could be anything.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 * Input data valid range should be [-65504, 65504]
 */
template<typename T, bool isReuseSource = false, const SinConfig &config = defaultSinConfig>
__aicore__ inline void Sin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    Sin<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

 /*!
 * \ingroup Sin
 * \brief compute Sin elementwisely
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] calCount: the number of elements to be processed.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 * Input data valid range should be [-65504, 65504]
 */
template<typename T, bool isReuseSource = false, const SinConfig &config = defaultSinConfig>
__aicore__ inline void Sin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    SinImpl<T, isReuseSource, config>(dstTensor, srcTensor, calCount);
}

 /*!
 * \ingroup Sin
 * \brief compute Sin elementwisely
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 * Input data valid range should be [-65504, 65504]
 */
template<typename T, bool isReuseSource = false, const SinConfig &config = defaultSinConfig>
__aicore__ inline void Sin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    Sin<T, isReuseSource, config>(dstTensor, srcTensor, srcTensor.GetSize());
}
}
#endif // LIB_MATH_SIN_L300_IMPL_H
