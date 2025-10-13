/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ascend_dequant_c310_impl.h
 * \brief
 */
#ifndef LIB_ASCEND_DEQUANT_ASCEND_DEQUANT_C310_IMPL_H
#define LIB_ASCEND_DEQUANT_ASCEND_DEQUANT_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ascend_dequant_common.h"

namespace AscendC {
constexpr uint32_t ASCENDC_DEQUANT_B32_VF_LEN = VECTOR_REG_WIDTH / sizeof(uint32_t);
template <typename dstT, typename scaleT, DeQuantMode mode>
__simd_vf__ inline void DequantPerchannelVFImpl(__local_mem__ half* dstUb, __local_mem__ int32_t* srcUb,
    __local_mem__ float* scaleUb, DequantParams params)
{
    uint32_t rowNum = params.m;
    uint32_t N = params.n;
    uint32_t calCount = params.calCount;
    uint32_t oneBlockNum = ONE_BLK_SIZE / sizeof(dstT);
    uint32_t dstInner = CeilDivision(N, oneBlockNum) * oneBlockNum;

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<int32_t> s32vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<half> b16vreg;
    MicroAPI::RegTensor<float> scaleB32Vreg0;
    MicroAPI::RegTensor<float> scaleB32Vreg1;

    uint32_t sregLower = ASCENDC_DEQUANT_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(calCount, sregLower));

    for (uint16_t i = 0; i < static_cast<uint16_t>(rowNum); ++i) {
        uint32_t sreg = calCount;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(s32vreg, srcUb + i * N + j * sregLower);
            MicroAPI::Cast<float, int32_t, MrgZRndA>(f32vreg, s32vreg, preg);

            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(scaleB32Vreg0, scaleB32Vreg1,
                scaleUb + 2 * j * sregLower); // only half of uint64_t is used

            MicroAPI::Mul(f32vreg, f32vreg, scaleB32Vreg0, preg);

            MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(b16vreg, f32vreg, preg);
            MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * dstInner + j * sregLower, b16vreg,
                preg);
        }
    }
}
template <typename dstT, typename scaleT, DeQuantMode mode>
__aicore__ inline void DequantPerchannelImpl(const LocalTensor<half>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const LocalTensor<uint64_t>& deqScale, DequantParams& params)
{
    __local_mem__ half* dstUb = (__local_mem__ half*)dstTensor.GetPhyAddr();
    __local_mem__ int32_t* srcUb = (__local_mem__ int32_t*)srcTensor.GetPhyAddr();
    __local_mem__ float* scaleUb = reinterpret_cast<__local_mem__ float*>(deqScale.GetPhyAddr());

    DequantPerchannelVFImpl<dstT, scaleT, mode>(dstUb, srcUb, scaleUb, params);
}

template <typename dstT, typename scaleT, DeQuantMode mode>
__simd_vf__ inline void DequantPerchannelVFImpl(__local_mem__ dstT* dstUb, __local_mem__ int32_t* srcUb,
    __local_mem__ scaleT* scaleUb, DequantParams params)
{
    uint32_t rowNum = params.m;
    uint32_t N = params.n;
    uint32_t calCount = params.calCount;
    uint32_t oneBlockNum = ONE_BLK_SIZE / sizeof(dstT);
    uint32_t dstInner = CeilDivision(N, oneBlockNum) * oneBlockNum;

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<int32_t> s32vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<dstT> b16vreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<float> scaleB32Vreg;

    uint32_t sregLower = ASCENDC_DEQUANT_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(calCount, sregLower));

    for (uint16_t i = 0; i < static_cast<uint16_t>(rowNum); ++i) {
        uint32_t sreg = calCount;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(s32vreg, srcUb + i * N + j * sregLower);

            MicroAPI::Cast<float, int32_t, MrgZRndA>(f32vreg, s32vreg, preg);
            if constexpr (SupportType<scaleT, bfloat16_t>()) {
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(scaleVreg,
                    scaleUb + j * sregLower);
                MicroAPI::Cast<float, bfloat16_t, layoutZMrgZ>(scaleB32Vreg, scaleVreg, preg); // bf16->fp32
            } else {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(scaleB32Vreg, scaleUb + j * sregLower);
            }

            MicroAPI::Mul(f32vreg, f32vreg, scaleB32Vreg, preg);

            if constexpr (SupportType<dstT, bfloat16_t, half>()) {
                MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(b16vreg, f32vreg, preg);
                MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * dstInner + j * sregLower, b16vreg,
                    preg);
            } else { // out is fp32
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dstInner + j * sregLower,
                    f32vreg, preg);
            }
        }
    }
}

template <typename dstT, typename scaleT, DeQuantMode mode>
__aicore__ inline void DequantPerchannelImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const LocalTensor<scaleT>& deqScale, DequantParams& params)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ int32_t* srcUb = (__local_mem__ int32_t*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)deqScale.GetPhyAddr();

    DequantPerchannelVFImpl<dstT, scaleT, mode>(dstUb, srcUb, scaleUb,params);
}

template <typename dstT, typename scaleT, DeQuantMode mode>
__simd_vf__ inline void DequantPertensorVFImpl(__local_mem__ dstT* dstUb, __local_mem__ int32_t* srcUb,
    const scaleT deqScale, DequantParams params)
{
    uint32_t rowNum = params.m;
    uint32_t N = params.n;
    uint32_t calCount = params.calCount;
    uint32_t oneBlockNum = ONE_BLK_SIZE / sizeof(dstT);
    uint32_t dstInner = CeilDivision(N, oneBlockNum) * oneBlockNum;

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<int32_t> s32vreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<bfloat16_t> b16vreg;

    uint32_t sregLower = ASCENDC_DEQUANT_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(calCount, sregLower));

    for (uint16_t i = 0; i < static_cast<uint16_t>(rowNum); ++i) {
        uint32_t sreg = calCount;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(s32vreg, srcUb + i * N + j * sregLower);
            MicroAPI::Cast<float, int32_t, MrgZRndA>(f32vreg, s32vreg, preg);
            if constexpr (SupportType<scaleT, bfloat16_t>()) {
                MicroAPI::Muls(f32vreg, f32vreg, ToFloat(deqScale), preg);
            } else {
                MicroAPI::Muls(f32vreg, f32vreg, deqScale, preg);
            }

            if constexpr (SupportType<dstT, bfloat16_t>()) {
                MicroAPI::Cast<bfloat16_t, float, LayoutZMrgZRndRSatS>(b16vreg, f32vreg, preg);
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * dstInner + j * sregLower,
                    b16vreg, preg);
            } else { // out is fp32
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(dstUb + i * dstInner + j * sregLower,
                    f32vreg, preg);
            }
        }
    }
}

template <typename dstT, typename scaleT, DeQuantMode mode>
__aicore__ inline void DequantPertensorImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<int32_t>& srcTensor,
    const scaleT deqScale, DequantParams& params)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ int32_t* srcUb = (__local_mem__ int32_t*)srcTensor.GetPhyAddr();
    DequantPertensorVFImpl<dstT, scaleT, mode>(dstUb, srcUb,deqScale, params);
}

template <typename scaleT>
__simd_callee__ inline void LoadPerTokenScale(__local_mem__ scaleT* addr, MicroAPI::RegTensor<scaleT>& vreg)
{
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B16>(vreg, addr);
    } else {
        MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_BRC_B32>(vreg, addr);
    }
}

template <typename dstT>
__simd_callee__ inline void StoreRes(__local_mem__ dstT* dstAddr, MicroAPI::RegTensor<float>& vreg,
                                MicroAPI::MaskReg& preg)
{
    if constexpr (SupportType<dstT, half, bfloat16_t>()) {
        MicroAPI::RegTensor<dstT> tempVreg;
        MicroAPI::Cast<dstT, float, LayoutZMrgZRndRSatS>(tempVreg, vreg, preg);
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr, tempVreg, preg);
    } else {
        MicroAPI::DataCopy<dstT, MicroAPI::StoreDist::DIST_NORM_B32>(dstAddr, vreg, preg);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__simd_vf__ inline void DeQuantPerTokenForS32VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const AscendDeQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(para.n, vecLen));

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<int32_t> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<float> scaleF32Vreg;
    for (uint16_t i = 0; i < static_cast<uint16_t>(rowNum); ++i) {
        LoadPerTokenScale<scaleT>(scaleUb + i, scaleVreg);
        uint32_t sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
            MicroAPI::Cast<float, int32_t, MrgZRndA>(f32Vreg, srcVreg, preg);
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                MicroAPI::Cast<float, scaleT, layoutZMrgZ>(scaleF32Vreg, scaleVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, scaleF32Vreg, preg);
            } else {
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, scaleVreg, preg);
            }
            StoreRes<dstT>(dstUb + i * para.n + j * vecLen, f32Vreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__aicore__ inline void DeQuantPerTokenForS32(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                             const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                             const AscendDeQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    DeQuantPerTokenForS32VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__simd_vf__ inline void DeQuantPerTokenForF32VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const AscendDeQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(para.n, vecLen));

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<float> scaleF32Vreg;
    for (uint16_t i = 0; i < static_cast<uint16_t>(rowNum); ++i) {
        LoadPerTokenScale<scaleT>(scaleUb + i, scaleVreg);
        uint32_t sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                MicroAPI::Cast<float, scaleT, layoutZMrgZ>(scaleF32Vreg, scaleVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, srcVreg, scaleF32Vreg, preg);
            } else {
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, srcVreg, scaleVreg, preg);
            }
            StoreRes<dstT>(dstUb + i * para.n + j * vecLen, f32Vreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__aicore__ inline void DeQuantPerTokenForF32(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                             const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                             const AscendDeQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    DeQuantPerTokenForF32VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, para);
}

template <typename T>
__simd_callee__ inline void GetPerGroupScale(__local_mem__ T* scaleUb, const int32_t start, const AscendDeQuantParam& para,
                                        const AscendDeQuantConfig& config, MicroAPI::RegTensor<T>& scaleReg)
{
    // use vgather to get perGroup scale/offset
    uint32_t groupSize = para.groupSize;
    if constexpr (SupportType<T, half, bfloat16_t>()) {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int16_t> vci_vreg;
        MicroAPI::RegTensor<uint16_t> index_vreg;
        MicroAPI::RegTensor<uint16_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint16_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int16_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint16_t> &)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(scaleReg, scaleUb, index_vreg, preg);
    } else {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<int32_t> vci_vreg;
        MicroAPI::RegTensor<uint32_t> index_vreg;
        MicroAPI::RegTensor<uint32_t> gsize_vreg;
        MicroAPI::Duplicate(gsize_vreg, static_cast<uint32_t>(groupSize));
        MicroAPI::Arange(vci_vreg, static_cast<int32_t>(start));
        MicroAPI::Div(index_vreg, (MicroAPI::RegTensor<uint32_t> &)vci_vreg, gsize_vreg, preg);
        MicroAPI::DataCopyGather(scaleReg, scaleUb, index_vreg, preg);
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__simd_vf__ inline void DeQuantPerGroupForColS32VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const AscendDeQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(para.n, vecLen));
    uint32_t sreg = para.n;
    uint16_t scaleK = static_cast<uint16_t>(CeilDivision(para.n, para.groupSize));

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<int32_t> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<scaleT> oriScaleVreg;
    MicroAPI::RegTensor<scaleT> tempVreg;
    MicroAPI::RegTensor<int32_t> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<float> scaleF32Vreg;
    MicroAPI::RegTensor<scaleT> zeroVreg;
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::MaskReg b16FullPreg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zeroVreg, static_cast<scaleT>(0), b16FullPreg);
    } else {
        MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zeroVreg, static_cast<scaleT>(0), b32FullPreg);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(rowNum); ++i) {
        sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                GetPerGroupScale<scaleT>(scaleUb + i * scaleK, j * vecLen, para, config, oriScaleVreg);
                MicroAPI::Interleave(scaleVreg, tempVreg, oriScaleVreg, zeroVreg);
                MicroAPI::Cast<float, scaleT, layoutZMrgZ>(scaleF32Vreg, scaleVreg, preg);
            } else {
                GetPerGroupScale<scaleT>(scaleUb + i * scaleK, j * vecLen, para, config, scaleF32Vreg);
            }
            MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
            MicroAPI::Cast<float, int32_t, MrgZRndA>(f32Vreg, srcVreg, preg);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, scaleF32Vreg, preg);
            StoreRes<dstT>(dstUb + i * para.n + j * vecLen, f32Vreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__aicore__ inline void DeQuantPerGroupForColS32(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                                const AscendDeQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    DeQuantPerGroupForColS32VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__simd_vf__ inline void DeQuantPerGroupForColF32VF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const AscendDeQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(para.n, vecLen));
    uint16_t scaleK = static_cast<uint16_t>(CeilDivision(para.n, para.groupSize));

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<float> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::RegTensor<scaleT> oriScaleVreg;
    MicroAPI::RegTensor<scaleT> tempVreg;
    MicroAPI::RegTensor<float> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<float> scaleF32Vreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> zeroVreg;
    if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
        MicroAPI::MaskReg b16FullPreg = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zeroVreg, static_cast<scaleT>(0), b16FullPreg);
    } else {
        MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(zeroVreg, static_cast<scaleT>(0), b32FullPreg);
    }
    for (uint16_t i = 0; i < static_cast<uint16_t>(rowNum); ++i) {
        uint32_t sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                GetPerGroupScale<scaleT>(scaleUb + i * scaleK, j * vecLen, para, config, oriScaleVreg);
                MicroAPI::Interleave(scaleVreg, tempVreg, oriScaleVreg, zeroVreg);
                MicroAPI::Cast<float, scaleT, layoutZMrgZ>(scaleF32Vreg, scaleVreg, preg);
            } else {
                GetPerGroupScale<scaleT>(scaleUb + i * scaleK, j * vecLen, para, config, scaleF32Vreg);
            }
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, srcVreg, scaleF32Vreg, preg);
            StoreRes<dstT>(dstUb + i * para.n + j * vecLen, f32Vreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__aicore__ inline void DeQuantPerGroupForColF32(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                                const AscendDeQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    DeQuantPerGroupForColF32VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__simd_callee__ inline void DeQuantPerGroupForRowTailBlock(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
                                                      __local_mem__ scaleT* scaleUb, uint16_t repeat,
                                                      uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(scaleVreg, scaleUb + j * vecLen);
                MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, scaleVreg, b32FullPreg);
            } else {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(f32ScaleVreg, scaleUb + j * vecLen);
            }
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * n + j * vecLen);
            if constexpr (SupportType<srcT, int32_t>()) {
                MicroAPI::Cast<float, int32_t, MrgZRndA>(f32Vreg, srcVreg, preg);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
            } else {
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, srcVreg, f32ScaleVreg, preg);
            }
            StoreRes<dstT>(dstUb + i * n + j * vecLen, f32Vreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__simd_vf__ inline void DeQuantPerGroupForRowVF(__local_mem__ dstT* dstUb, __local_mem__ srcT* srcUb,
    __local_mem__ scaleT* scaleUb, const AscendDeQuantParam para, uint16_t rowNum, uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / para.groupSize;
        uint32_t vecLen = ASCENDC_QUANT_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(para.n, vecLen));

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<float> f32ScaleVreg;
    MicroAPI::RegTensor<srcT> srcVreg;
    MicroAPI::RegTensor<float> f32Vreg;
    MicroAPI::MaskReg b32FullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < mainRowGroup; ++i) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
            uint32_t sreg = para.n;
            for (uint16_t k = 0; k < repeat; ++k) {
                if constexpr (SupportType<scaleT, half, bfloat16_t>()) {
                    MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_UNPACK_B16>(scaleVreg, scaleUb + i * para.n + k * vecLen);
                    MicroAPI::Cast<float, scaleT, layoutZMrgZ>(f32ScaleVreg, scaleVreg, b32FullPreg);
                } else {
                    MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(
                        f32ScaleVreg, scaleUb + i * para.n + k * vecLen);
                }
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(
                    srcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                if constexpr (SupportType<srcT, int32_t>()) {
                    MicroAPI::Cast<float, int32_t, MrgZRndA>(f32Vreg, srcVreg, preg);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, f32Vreg, f32ScaleVreg, preg);
                } else {
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(f32Vreg, srcVreg, f32ScaleVreg, preg);
                }
                StoreRes<dstT>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, f32Vreg, preg);
            }
        }
    }
    DeQuantPerGroupForRowTailBlock<dstT, srcT, scaleT, config>(
        dstUb + mainRowGroup * para.groupSize * para.n, srcUb + mainRowGroup * para.groupSize * para.n,
        scaleUb + mainRowGroup * para.n, repeat, tailRow, para.n, vecLen);
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__aicore__ inline void DeQuantPerGroupForRow(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                             const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                             const AscendDeQuantParam& para)
{
    __local_mem__ dstT* dstUb = (__local_mem__ dstT*)dstTensor.GetPhyAddr();
    __local_mem__ srcT* srcUb = (__local_mem__ srcT*)srcTensor.GetPhyAddr();
    __local_mem__ scaleT* scaleUb = (__local_mem__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t tailRow = rowNum % para.groupSize;
    DeQuantPerGroupForRowVF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, para, rowNum, tailRow);
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__aicore__ inline void AscendDeQuantPerToken(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                             const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                             const LocalTensor<scaleT>& offsetTensor, const AscendDeQuantParam& para)
{
    if constexpr (SupportType<srcT, int32_t>()) {
        DeQuantPerTokenForS32<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<srcT, float>()) {
        DeQuantPerTokenForF32<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport srcT for AscendDeQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__aicore__ inline void AscendDeQuantPerGroupForCol(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                   const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                                   const LocalTensor<scaleT>& offsetTensor, const AscendDeQuantParam& para)
{
    if constexpr (SupportType<srcT, int32_t>()) {
        DeQuantPerGroupForColS32<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else if constexpr (SupportType<srcT, float>()) {
        DeQuantPerGroupForColF32<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport srcT for AscendDeQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config>
__aicore__ inline void AscendDeQuantPerGroupForRow(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                   const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                                   const LocalTensor<scaleT>& offsetTensor, const AscendDeQuantParam& para)
{
    if constexpr (SupportType<srcT, int32_t, float>()) {
        DeQuantPerGroupForRow<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport srcT for AscendDeQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendDeQuantConfig& config, const AscendDeQuantPolicy& policy>
__aicore__ inline void AscendDequantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                         const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                         const LocalTensor<scaleT>& offsetTensor, const AscendDeQuantParam& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<srcT, int32_t, float>(),
        "AscendDequant only support int32_t/float input dtype");
    static_assert(SupportType<dstT, bfloat16_t, half, float>(),
        "AscendDequant only support bfloat16_t/half/float output dtype");
    static_assert(SupportType<scaleT, bfloat16_t, half, float>(),
        "AscendDequant only support bfloat16_t/half/float scaleT dtype");
    static_assert(((policy == AscendDeQuantPolicy::PER_TOKEN) || (policy == AscendDeQuantPolicy::PER_GROUP)),
        "unsupported policy for AscendDequant in current device!");
    ASCENDC_ASSERT((para.calCount <= srcTensor.GetSize() && para.calCount <= dstTensor.GetSize() && para.calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            para.calCount, srcTensor.GetSize(), dstTensor.GetSize());
    });
    if constexpr (policy == AscendDeQuantPolicy::PER_TOKEN) {
        AscendDeQuantPerToken<dstT, srcT, scaleT, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
    } else if constexpr (policy == AscendDeQuantPolicy::PER_GROUP) {
        static_assert(
            ((config.kDim == 0) || (config.kDim == 1)), "AscendDequant PerGroup only support kDim is axis 0/1!");
        ASCENDC_ASSERT((para.groupSize > 0 && para.groupSize % 32 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "groupSize must be an integer multiple of 32 and greater than 0 !"); });
        if constexpr (config.kDim == 1) {
            AscendDeQuantPerGroupForCol<dstT, srcT, scaleT, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
        } else {
            AscendDeQuantPerGroupForRow<dstT, srcT, scaleT, config>(dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
        }
    }
}
} //  namespace AscendC
#endif // LIB_ASCEND_DEQUANT_ASCEND_DEQUANT_C310_IMPL_H
