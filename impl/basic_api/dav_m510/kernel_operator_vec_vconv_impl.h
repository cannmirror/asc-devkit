/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_vec_template_impl.h"
namespace AscendC {
constexpr MicroAPI::CastTrait layoutZMrgZ = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN };

constexpr MicroAPI::CastTrait layoutZSatSMrgZ = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN };

constexpr MicroAPI::CastTrait layoutZSatSMrgZRndA = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };

constexpr MicroAPI::CastTrait layoutZSatSMrgZRndH = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID };

constexpr MicroAPI::CastTrait layoutZSatSMrgZRndR = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

constexpr MicroAPI::CastTrait layoutZMrgZRndR = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

constexpr MicroAPI::CastTrait layoutZMrgZRndA = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };

constexpr MicroAPI::CastTrait layoutZMrgZRndC = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL };

constexpr MicroAPI::CastTrait layoutZMrgZRndF = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR };

constexpr MicroAPI::CastTrait layoutZMrgZRndZ = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC };

constexpr MicroAPI::CastTrait MrgZRndR = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

constexpr MicroAPI::CastTrait MrgZRndA = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };

constexpr MicroAPI::CastTrait MrgZRndF = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL };

constexpr MicroAPI::CastTrait MrgZRndC = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR };

constexpr MicroAPI::CastTrait MrgZRndZ = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC };

constexpr MicroAPI::CastTrait MrgZRndRSatS = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

constexpr MicroAPI::CastTrait MrgZRndASatS = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };

constexpr MicroAPI::CastTrait MrgZRndFSatS = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL };

constexpr MicroAPI::CastTrait MrgZRndCSatS = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR };

constexpr MicroAPI::CastTrait MrgZRndZSatS = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC };

constexpr MicroAPI::CastTrait LayoutZMrgZRndRSatS = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

constexpr MicroAPI::CastTrait LayoutZMrgZRndASatS = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };

constexpr MicroAPI::CastTrait LayoutZMrgZRndRSatNS = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

constexpr MicroAPI::CastTrait LayoutZMrgZRndASatNS = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };

constexpr MicroAPI::CastTrait MrgZRndRSatNS = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

namespace CastParam {
constexpr MicroAPI::CastTrait AddReluCastTrait = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait SubReluCastTrait = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait s162HalfTrait = {
    MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait s162f32CastTrait = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait f322s16CastTrait = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait TrueHalfBlockCastTrait = {
    MicroAPI::RegLayout::ONE, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait FalseHalfBlockCastTrait = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait TrueHalfBlockHalf2S8Trait = {
    MicroAPI::RegLayout::ONE, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait FalseHalfBlockHalf2S8Trait = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait s322F32CastTrait = {
    MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait f322F16CastTrait = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename ORI_TYPE>
struct CastTypeTrait {
    using RealType = ORI_TYPE;
};

template <>
struct CastTypeTrait<int4b_t> {
    using RealType = int4x2_t;
};
}  // namespace CastParam

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
__aicore__ inline void CastIntrinsicsB64ImplVF(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint32_t calCount)
{
    constexpr uint16_t oneRepSize = 2 * GetVecLen() / sizeof(int64_t);
    uint16_t repeatTime = CeilDivision(calCount, oneRepSize);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    if constexpr (std::is_same_v<SRC_TYPE, complex64> && std::is_same_v<DST_TYPE, complex32>) {
        __VEC_SCOPE__  // complex 64 -> 32
        {
            MicroAPI::MaskReg preg;
            MicroAPI::RegTensor<SRC_TYPE, MicroAPI::RegTraitNumTwo> srcVreg;
            MicroAPI::RegTensor<DST_TYPE, MicroAPI::RegTraitNumTwo> dstVreg;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                preg = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(srcVreg, src + i * oneRepSize);
                MicroAPI::Cast<typename DST_TYPE::EleType, typename SRC_TYPE::EleType, castTrait>((MicroAPI::RegTensor<typename DST_TYPE::EleType> &)dstVreg.reg[0], 
                    (MicroAPI::RegTensor<typename SRC_TYPE::EleType> &)srcVreg.reg[0], preg);
                MicroAPI::Cast<typename DST_TYPE::EleType, typename SRC_TYPE::EleType, castTrait>((MicroAPI::RegTensor<typename DST_TYPE::EleType> &)dstVreg.reg[1], 
                    (MicroAPI::RegTensor<typename SRC_TYPE::EleType> &)srcVreg.reg[1], preg);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)dstVreg.reg[0], (MicroAPI::RegTensor<uint32_t> &)dstVreg.reg[0]);
                MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)dstVreg.reg[1], (MicroAPI::RegTensor<uint32_t> &)dstVreg.reg[1]);
                MicroAPI::MaskPack(preg, preg);
                MicroAPI::StoreAlign(dst + i * oneRepSize, dstVreg, preg);
            }
        }
    } else if constexpr (std::is_same_v<SRC_TYPE, complex64> && std::is_same_v<DST_TYPE, complex64>) {
        __VEC_SCOPE__  // complex64 -> 64
        {
            MicroAPI::MaskReg preg;
            MicroAPI::RegTensor<SRC_TYPE, MicroAPI::RegTraitNumTwo> srcVreg;
            MicroAPI::RegTensor<DST_TYPE, MicroAPI::RegTraitNumTwo> dstVreg;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                preg = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(srcVreg, src + i * oneRepSize);
                MicroAPI::Truncate<float, roundMode>((MicroAPI::RegTensor<float> &)dstVreg.reg[0], 
                    (MicroAPI::RegTensor<float> &)srcVreg.reg[0], preg);
                MicroAPI::Truncate<float, roundMode>((MicroAPI::RegTensor<float> &)dstVreg.reg[1], 
                    (MicroAPI::RegTensor<float> &)srcVreg.reg[1], preg);
                MicroAPI::StoreAlign(dst + i * oneRepSize, dstVreg, preg);
            }
        }
    } else if constexpr (std::is_same_v<SRC_TYPE, complex32> && std::is_same_v<DST_TYPE, complex64>) {
        __VEC_SCOPE__  // complex32 -> 64
        {
            MicroAPI::MaskReg preg;
            MicroAPI::RegTensor<SRC_TYPE, MicroAPI::RegTraitNumTwo> srcVreg;
            MicroAPI::RegTensor<DST_TYPE, MicroAPI::RegTraitNumTwo> dstVreg;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                preg = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(srcVreg, src + i * oneRepSize);
                MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t> &)srcVreg.reg[0], (MicroAPI::RegTensor<uint16_t> &)srcVreg.reg[0]);
                MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t> &)srcVreg.reg[1], (MicroAPI::RegTensor<uint16_t> &)srcVreg.reg[1]);
                MicroAPI::Cast<typename DST_TYPE::EleType, typename SRC_TYPE::EleType, castTrait>((MicroAPI::RegTensor<typename DST_TYPE::EleType> &)dstVreg.reg[0], 
                    (MicroAPI::RegTensor<typename SRC_TYPE::EleType> &)srcVreg.reg[0], preg);
                MicroAPI::Cast<typename DST_TYPE::EleType, typename SRC_TYPE::EleType, castTrait>((MicroAPI::RegTensor<typename DST_TYPE::EleType> &)dstVreg.reg[1], 
                    (MicroAPI::RegTensor<typename SRC_TYPE::EleType> &)srcVreg.reg[1], preg);
                MicroAPI::StoreAlign(dst + i * oneRepSize, dstVreg, preg);
            }
        }
    } else if constexpr (sizeof(DST_TYPE) == sizeof(int64_t)) {
        __VEC_SCOPE__  // B32 -> B64
        {
            MicroAPI::MaskReg preg;
            MicroAPI::RegTensor<SRC_TYPE> srcVreg;
            MicroAPI::RegTensor<DST_TYPE, MicroAPI::RegTraitNumTwo> dstVreg;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                preg = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(srcVreg, src + i * oneRepSize);
                MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, preg);
                MicroAPI::StoreAlign(dst + i * oneRepSize, dstVreg, preg);
            }
        }
    } else {
        __VEC_SCOPE__  // B64 -> B32
        {
            MicroAPI::MaskReg preg;
            MicroAPI::RegTensor<SRC_TYPE, MicroAPI::RegTraitNumTwo> srcVreg;
            MicroAPI::RegTensor<DST_TYPE> dstVreg;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                preg = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(srcVreg, src + i * oneRepSize);
                MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, preg);
                MicroAPI::StoreAlign(dst + i * oneRepSize, dstVreg, preg);
            }
        }
    }
}

template <typename DST_TYPE, typename SRC_TYPE>
__aicore__ inline void GenLoadL2(MicroAPI::RegTensor<SRC_TYPE> &srcVreg, __ubuf__ SRC_TYPE *srcAddr, MicroAPI::MaskReg &preg)
{
    if constexpr (SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(DST_TYPE) == 2) {
        MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<uint8_t> &)srcVreg, (__ubuf__ uint8_t *)srcAddr);
    } else if constexpr (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 2) {
        MicroAPI::LoadAlign<SRC_TYPE, MicroAPI::LoadDist::DIST_UNPACK_B8>(srcVreg, srcAddr);
    } else if constexpr (sizeof(SRC_TYPE) == 2 && sizeof(DST_TYPE) == 4) {
        MicroAPI::LoadAlign<SRC_TYPE, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, srcAddr);
    } else if constexpr (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4) {
        MicroAPI::LoadAlign<SRC_TYPE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcVreg, srcAddr);
    } else {
        MicroAPI::LoadAlign(srcVreg, srcAddr);
    }
}

template <typename DST_TYPE, typename SRC_TYPE>
__aicore__ inline void GenStoreL2(__ubuf__ DST_TYPE *dstAddr, MicroAPI::RegTensor<DST_TYPE> &dstVreg, MicroAPI::MaskReg &preg)
{
    if constexpr (SupportType<DST_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(SRC_TYPE) == 2) {
        MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_PACK4_B32>(
            (__ubuf__ uint8_t *)dstAddr, (MicroAPI::RegTensor<uint8_t> &)dstVreg, preg);
    } else if constexpr (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 2) {
        MicroAPI::StoreAlign<DST_TYPE, MicroAPI::StoreDist::DIST_PACK_B16>(dstAddr, dstVreg, preg);
    } else if constexpr (sizeof(DST_TYPE) == 2 && sizeof(SRC_TYPE) == 4) {
        MicroAPI::StoreAlign<DST_TYPE, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr, dstVreg, preg);
    } else if constexpr (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4) {
        MicroAPI::StoreAlign<DST_TYPE, MicroAPI::StoreDist::DIST_PACK4_B32>(dstAddr, dstVreg, preg);
    } else {
        MicroAPI::StoreAlign(dstAddr, dstVreg, preg);
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
__aicore__ inline void CastIntrinsicsImplVF(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint32_t calCount)
{
    uint16_t oneRepSize = GetVecLen() / sizeof(SRC_TYPE);
    if constexpr (sizeof(SRC_TYPE) < sizeof(DST_TYPE)) {
        oneRepSize = GetVecLen() / sizeof(DST_TYPE);
    }
    uint16_t repeatTime = CeilDivision(calCount, oneRepSize);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        if constexpr (sizeof(SRC_TYPE) < sizeof(DST_TYPE)) {
            preg = MicroAPI::UpdateMask<DST_TYPE>(sreg);
        } else {
            preg = MicroAPI::UpdateMask<SRC_TYPE>(sreg);
        }
        if constexpr (SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
            GenLoadL2<DST_TYPE, SRC_TYPE>(srcVreg, src + (i * oneRepSize) / 2, preg);
        } else {
            GenLoadL2<DST_TYPE, SRC_TYPE>(srcVreg, src + i * oneRepSize, preg);
        }
        if constexpr (std::is_same_v<SRC_TYPE, int32_t> && std::is_same_v<DST_TYPE, half>) {
            MicroAPI::Cast<float, SRC_TYPE, castTrait>((MicroAPI::RegTensor<float> &)dstVreg, srcVreg, preg);
            MicroAPI::Cast<DST_TYPE, float, castTrait>(dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, preg);
        } else if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<DST_TYPE, float>) {
            MicroAPI::Truncate<DST_TYPE, roundMode>(dstVreg, srcVreg, preg);
        } else {
            MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, preg);
        }
        if constexpr (SupportType<DST_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
            GenStoreL2<DST_TYPE, SRC_TYPE>(dst + (i * oneRepSize) / 2, dstVreg, preg);
        } else {
            GenStoreL2<DST_TYPE, SRC_TYPE>(dst + i * oneRepSize, dstVreg, preg);
        }
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
__aicore__ inline void CastIntrinsicsImpl(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint32_t calCount)
{
    constexpr bool b64Cast = SupportType<Tuple<DST_TYPE, SRC_TYPE>,
        Tuple<float, int64_t>,
        Tuple<int64_t, float>,
        Tuple<int32_t, int64_t>,
        Tuple<int64_t, int32_t>,
        Tuple<complex64, complex64>,
        Tuple<complex64, complex32>,
        Tuple<complex32, complex64>>();
    if constexpr (b64Cast) {
        VF_CALL<CastIntrinsicsB64ImplVF<DST_TYPE, SRC_TYPE, roundMode>>(dst, src, calCount);
    } else {
        VF_CALL<CastIntrinsicsImplVF<DST_TYPE, SRC_TYPE, roundMode>>(dst, src, calCount);
    }
}

// Cast::Level 2
template <typename ORI_DST_TYPE, typename ORI_SRC_TYPE>
__aicore__ inline void CastImpl(
    __ubuf__ ORI_DST_TYPE *oriDst, __ubuf__ ORI_SRC_TYPE *oriSrc, const RoundMode &roundMode, const uint32_t calCount)
{
    using DST_TYPE = typename CastParam::CastTypeTrait<ORI_DST_TYPE>::RealType;
    using SRC_TYPE = typename CastParam::CastTypeTrait<ORI_SRC_TYPE>::RealType;
    __ubuf__ DST_TYPE* dst = reinterpret_cast<__ubuf__ DST_TYPE*>(oriDst);
    __ubuf__ SRC_TYPE* src = reinterpret_cast<__ubuf__ SRC_TYPE*>(oriSrc);

    constexpr bool cast_round_all = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>, Tuple<int64_t, float>,
        Tuple<int32_t, float>, Tuple<int16_t, float>, Tuple<bfloat16_t, float>, Tuple<int32_t, half>,
        Tuple<int16_t, half>, Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4x2_t, half>, Tuple<bfloat16_t, half>,
        Tuple<half, int16_t>, Tuple<float, int32_t>,Tuple<float, int64_t>, Tuple<int32_t, bfloat16_t>,
        Tuple<half, bfloat16_t>, Tuple<fp4x2_e1m2_t, bfloat16_t>,Tuple<fp4x2_e2m1_t, bfloat16_t>,
        Tuple<half, int32_t>, Tuple<float, float>, Tuple<complex64, complex64>, Tuple<complex32, complex64>>();
    constexpr bool cast_none = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<float, half>, Tuple<float, bfloat16_t>,
        Tuple<half, int4x2_t>, Tuple<half, uint8_t>, Tuple<uint16_t, uint8_t>, Tuple<uint32_t, uint8_t>,
        Tuple<half, int8_t>, Tuple<int16_t, int8_t>, Tuple<int32_t, int8_t>, Tuple<uint8_t, uint16_t>,
        Tuple<uint32_t, uint16_t>, Tuple<float, int16_t>, Tuple<uint8_t, int16_t>, Tuple<uint32_t, int16_t>,
        Tuple<int32_t, int16_t>, Tuple<uint8_t, uint32_t>, Tuple<uint16_t, uint32_t>, Tuple<int16_t, uint32_t>,
        Tuple<int64_t, int32_t>, Tuple<int16_t, int32_t>, Tuple<uint8_t, int32_t>, Tuple<uint16_t, int32_t>,
        Tuple<int32_t, int64_t>, Tuple<half, hifloat8_t>, Tuple<float, hifloat8_t>, Tuple<float, fp8_e4m3fn_t>,
        Tuple<float, fp8_e5m2_t>, Tuple<bfloat16_t, fp4x2_e1m2_t>, Tuple<bfloat16_t, fp4x2_e2m1_t>,
        Tuple<int4x2_t, int16_t>, Tuple<int16_t, int4x2_t>, Tuple<bfloat16_t, int4x2_t>, Tuple<complex64, complex32>>();
    constexpr bool using_cast_rint =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4x2_t, half>,
        Tuple<half, float>, Tuple<half, int16_t>, Tuple<float, int32_t>, Tuple<complex32, complex64>>();
    constexpr bool cast_odd = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>, Tuple<complex32, complex64>>();
    constexpr bool cast_rint = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<fp8_e5m2_t, float>,
        Tuple<fp8_e4m3fn_t, float>>();
    constexpr bool cast_round =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<hifloat8_t, float>, Tuple<hifloat8_t, half>>();
    constexpr bool cast_hybrid =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<hifloat8_t, float>, Tuple<hifloat8_t, half>>();
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            if constexpr (cast_round_all || cast_rint) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT>(dst, src, calCount);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast rint"); });
            }
            break;
        case RoundMode::CAST_FLOOR:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_FLOOR>(dst, src, calCount);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast floor"); });
            }
            break;
        case RoundMode::CAST_CEIL:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_CEIL>(dst, src, calCount);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast ceil"); });
            }
            break;
        case RoundMode::CAST_ROUND:
            if constexpr (cast_round_all || cast_round) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ROUND>(dst, src, calCount);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast round"); });
            }
            break;
        case RoundMode::CAST_TRUNC:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_TRUNC>(dst, src, calCount);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast trunc"); });
            }
            break;
        case RoundMode::CAST_ODD:
            if constexpr (cast_odd) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ODD>(dst, src, calCount);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast odd"); });
            }
            break;
        case RoundMode::CAST_NONE:
            if constexpr (cast_none) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_NONE>(dst, src, calCount);
            } else if constexpr (using_cast_rint) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT>(dst, src, calCount);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast none"); });
            }
            break;
        case RoundMode::CAST_HYBRID:
            if constexpr (cast_hybrid) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_HYBRID>(dst, src, calCount);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast hybrid"); });
            }
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
__aicore__ inline void CastIntrinsicsB64ImplVF2(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask[],
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    MicroAPI::RegTensor<uint32_t> zeroVreg, tmpVreg;
    MicroAPI::MaskReg fullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg b32Preg = MicroAPI::MoveMask<uint32_t>();
    MicroAPI::MaskReg b64Preg, tmpPreg;
    MicroAPI::MaskInterleave<uint32_t>(b64Preg, tmpPreg, b32Preg, b32Preg);
    MicroAPI::Duplicate(zeroVreg, 0, fullPreg);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        if constexpr (sizeof(DST_TYPE) == sizeof(int64_t)) {
            // b32->b64
            MicroAPI::LoadAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<uint32_t> &)srcVreg,
                (__ubuf__ uint32_t *&)src,
                static_cast<uint32_t>(repeatParams.srcBlkStride),
                static_cast<uint32_t>(repeatParams.srcRepStride),
                b32Preg);
            MicroAPI::Interleave(
                (MicroAPI::RegTensor<uint32_t> &)srcVreg, tmpVreg, (MicroAPI::RegTensor<uint32_t> &)srcVreg, zeroVreg);
        } else {
            // b64->b32
            MicroAPI::LoadAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<uint32_t> &)srcVreg,
                (__ubuf__ uint32_t *&)src,
                static_cast<uint32_t>(repeatParams.srcBlkStride),
                static_cast<uint32_t>(repeatParams.srcRepStride),
                b64Preg);
        }
        MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, b64Preg);
        if constexpr (sizeof(DST_TYPE) == sizeof(int64_t)) {
            // b32->b64
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::StoreAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)dst,
                (MicroAPI::RegTensor<uint32_t> &)dstVreg,
                static_cast<uint32_t>(repeatParams.dstBlkStride),
                static_cast<uint32_t>(repeatParams.dstRepStride),
                b64Preg);
        } else {
            // b64->b32
            MicroAPI::DeInterleave(
                (MicroAPI::RegTensor<uint32_t> &)dstVreg, tmpVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg, zeroVreg);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::StoreAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)dst,
                (MicroAPI::RegTensor<uint32_t> &)dstVreg,
                static_cast<uint32_t>(repeatParams.dstBlkStride),
                static_cast<uint32_t>(repeatParams.dstRepStride),
                b32Preg);
        }
    }
}

template <typename DST_TYPE, typename SRC_TYPE>
__aicore__ inline void GenLoadL0(MicroAPI::RegTensor<SRC_TYPE> &srcVreg, __ubuf__ SRC_TYPE *&srcAddr,
    MicroAPI::MaskReg &preg, const UnaryRepeatParams &repeatParams)
{
    MicroAPI::LoadAlign<SRC_TYPE, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        srcVreg, srcAddr,
        static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), preg);
    if constexpr (SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(DST_TYPE) == 2) {
        MicroAPI::UnPack<uint16_t, uint8_t>(
            (MicroAPI::RegTensor<uint16_t> &)srcVreg, (MicroAPI::RegTensor<uint8_t> &)srcVreg);
        MicroAPI::UnPack<uint32_t, uint16_t>(
            (MicroAPI::RegTensor<uint32_t> &)srcVreg, (MicroAPI::RegTensor<uint16_t> &)srcVreg);
    } else if constexpr (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 2) {
        if constexpr (std::is_same_v<SRC_TYPE, int8_t>) {
            MicroAPI::UnPack<int16_t, int8_t>((MicroAPI::RegTensor<int16_t> &)srcVreg, srcVreg);
        } else {
            MicroAPI::UnPack<uint16_t, uint8_t>(
                (MicroAPI::RegTensor<uint16_t> &)srcVreg, (MicroAPI::RegTensor<uint8_t> &)srcVreg);
        }
    } else if constexpr (sizeof(SRC_TYPE) == 2 && sizeof(DST_TYPE) == 4) {
        if constexpr (std::is_same_v<SRC_TYPE, int16_t>) {
            MicroAPI::UnPack<int32_t, int16_t>((MicroAPI::RegTensor<int32_t> &)srcVreg, srcVreg);
        } else {
            MicroAPI::UnPack<uint32_t, uint16_t>(
                (MicroAPI::RegTensor<uint32_t> &)srcVreg, (MicroAPI::RegTensor<uint16_t> &)srcVreg);
        }
    } else if constexpr (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4) {
        if constexpr (std::is_same_v<SRC_TYPE, int8_t>) {
            MicroAPI::UnPack<int16_t, int8_t>((MicroAPI::RegTensor<int16_t> &)srcVreg, srcVreg);
            MicroAPI::UnPack<int32_t, int16_t>(
                (MicroAPI::RegTensor<int32_t> &)srcVreg, (MicroAPI::RegTensor<int16_t> &)srcVreg);
        } else {
            MicroAPI::UnPack<uint16_t, uint8_t>(
                (MicroAPI::RegTensor<uint16_t> &)srcVreg, (MicroAPI::RegTensor<uint8_t> &)srcVreg);
            MicroAPI::UnPack<uint32_t, uint16_t>(
                (MicroAPI::RegTensor<uint32_t> &)srcVreg, (MicroAPI::RegTensor<uint16_t> &)srcVreg);
        }
    }
}

template <typename DST_TYPE, typename SRC_TYPE>
__aicore__ inline void GenStoreL0(__ubuf__ DST_TYPE *&dstAddr, MicroAPI::RegTensor<DST_TYPE> &dstVreg,
    MicroAPI::MaskReg &preg, const UnaryRepeatParams &repeatParams)
{
    if constexpr (SupportType<DST_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(SRC_TYPE) == 2) {
        MicroAPI::Pack<uint16_t, uint32_t>(
            (MicroAPI::RegTensor<uint16_t> &)dstVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg);
        MicroAPI::Pack<uint8_t, uint16_t>(
            (MicroAPI::RegTensor<uint8_t> &)dstVreg, (MicroAPI::RegTensor<uint16_t> &)dstVreg);
    } else if constexpr (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 2) {
        MicroAPI::Pack<uint8_t, uint16_t>(
            (MicroAPI::RegTensor<uint8_t> &)dstVreg, (MicroAPI::RegTensor<uint16_t> &)dstVreg);
    } else if constexpr (sizeof(DST_TYPE) == 2 && sizeof(SRC_TYPE) == 4) {
        MicroAPI::Pack<uint16_t, uint32_t>(
            (MicroAPI::RegTensor<uint16_t> &)dstVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg);
    } else if constexpr (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4) {
        MicroAPI::Pack<uint16_t, uint32_t>(
            (MicroAPI::RegTensor<uint16_t> &)dstVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg);
        MicroAPI::Pack<uint8_t, uint16_t>(
            (MicroAPI::RegTensor<uint8_t> &)dstVreg, (MicroAPI::RegTensor<uint16_t> &)dstVreg);
    }
    MicroAPI::StoreAlign<DST_TYPE, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        dstAddr, dstVreg,
        static_cast<uint32_t>(repeatParams.dstBlkStride), static_cast<uint32_t>(repeatParams.dstRepStride), preg);
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
__aicore__ inline void CastIntrinsicsImplVF2(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask[],
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    MicroAPI::MaskReg ldPreg;
    MicroAPI::MaskReg exPreg;
    MicroAPI::MaskReg stPreg;
    MicroAPI::MaskReg dumpPreg;
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    if constexpr (sizeof(DST_TYPE) == sizeof(SRC_TYPE)) {
        ldPreg = MicroAPI::MoveMask<SRC_TYPE>();
        exPreg = ldPreg;
        stPreg = ldPreg;
    } else if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
        ldPreg = MicroAPI::MoveMask<SRC_TYPE>();
        exPreg = ldPreg;
        MicroAPI::MaskPack(stPreg, ldPreg);
        if constexpr ((SupportType<DST_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(SRC_TYPE) == 2) ||
                      (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4)) {
            MicroAPI::MaskPack(stPreg, stPreg);
        }
    } else if constexpr (sizeof(DST_TYPE) > sizeof(SRC_TYPE)) {
        stPreg = MicroAPI::MoveMask<DST_TYPE>();
        exPreg = stPreg;
        MicroAPI::MaskPack(ldPreg, stPreg);
        if constexpr ((SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(DST_TYPE) == 2) ||
                      (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4)) {
            MicroAPI::MaskPack(ldPreg, ldPreg);
            if constexpr (SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(DST_TYPE) == 2) {
                MicroAPI::MaskUnPack(stPreg, ldPreg);
                MicroAPI::MaskUnPack(exPreg, stPreg);
                MicroAPI::MaskInterleave<uint16_t>(stPreg, dumpPreg, stPreg, stPreg);
            }
        }
    }
    for (uint16_t i = 0; i < repeatTime; ++i) {
        GenLoadL0<DST_TYPE, SRC_TYPE>(srcVreg, src, ldPreg, repeatParams);
        if constexpr (std::is_same_v<SRC_TYPE, int32_t> && std::is_same_v<DST_TYPE, half>) {
            MicroAPI::Cast<float, SRC_TYPE, castTrait>((MicroAPI::RegTensor<float> &)dstVreg, srcVreg, exPreg);
            MicroAPI::Cast<DST_TYPE, float, castTrait>(dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, exPreg);
        } else if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<DST_TYPE, float>) {
            MicroAPI::Truncate<DST_TYPE, roundMode>(dstVreg, srcVreg, exPreg);
        } else {
            MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, exPreg);
        }
        GenStoreL0<DST_TYPE, SRC_TYPE>(dst, dstVreg, stPreg, repeatParams);
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__aicore__ inline void CastIntrinsicsB64ImplCounterVF(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask,
    __ubuf__ uint64_t *maskBuf, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    uint32_t countSreg = static_cast<uint32_t>(mask);
    if constexpr (!isSetMask) {
        // get SPR.MASK in VF
        MicroAPI::MaskReg sprLoadMaskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(maskBuf, sprLoadMaskReg);
        // insert membar(vec store operation) before load maskBuf[0](scalar load operation)
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        countSreg = static_cast<uint32_t>(maskBuf[0]);
    }
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(uint64_t);
    uint16_t newRepeatTimes = CeilDivision(countSreg, oneRepSize);
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    MicroAPI::RegTensor<uint32_t> zeroVreg, tmpVreg;
    MicroAPI::MaskReg b32Preg, b64Preg, tmpPreg;
    MicroAPI::MaskReg fullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroVreg, 0, fullPreg);
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        b32Preg = MicroAPI::UpdateMask<uint32_t>(countSreg);
        MicroAPI::MaskInterleave<uint32_t>(b64Preg, tmpPreg, b32Preg, b32Preg);
        if constexpr (sizeof(DST_TYPE) == sizeof(int64_t)) {
            // b32->b64
            MicroAPI::LoadAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<uint32_t> &)srcVreg,
                (__ubuf__ uint32_t *&)src,
                static_cast<uint32_t>(repeatParams.srcBlkStride),
                static_cast<uint32_t>(repeatParams.srcRepStride),
                b32Preg);
            MicroAPI::Interleave(
                (MicroAPI::RegTensor<uint32_t> &)srcVreg, tmpVreg, (MicroAPI::RegTensor<uint32_t> &)srcVreg, zeroVreg);
        } else {
            // b64->b32
            MicroAPI::LoadAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<uint32_t> &)srcVreg,
                (__ubuf__ uint32_t *&)src,
                static_cast<uint32_t>(repeatParams.srcBlkStride),
                static_cast<uint32_t>(repeatParams.srcRepStride),
                b64Preg);
        }
        MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, b64Preg);
        if constexpr (sizeof(DST_TYPE) == sizeof(int64_t)) {
            // b32->b64
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::StoreAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)dst,
                (MicroAPI::RegTensor<uint32_t> &)dstVreg,
                static_cast<uint32_t>(repeatParams.dstBlkStride),
                static_cast<uint32_t>(repeatParams.dstRepStride),
                b64Preg);
        } else {
            // b64->b32
            MicroAPI::DeInterleave(
                (MicroAPI::RegTensor<uint32_t> &)dstVreg, tmpVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg, zeroVreg);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::StoreAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)dst,
                (MicroAPI::RegTensor<uint32_t> &)dstVreg,
                static_cast<uint32_t>(repeatParams.dstBlkStride),
                static_cast<uint32_t>(repeatParams.dstRepStride),
                b32Preg);
        }
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__aicore__ inline void CastIntrinsicsImplCounterVF(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask,
    __ubuf__ uint64_t *maskBuf, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    MicroAPI::MaskReg ldPreg;
    MicroAPI::MaskReg exPreg;
    MicroAPI::MaskReg stPreg;
    MicroAPI::MaskReg dumpPreg;
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    uint32_t countSreg = static_cast<uint32_t>(mask);
    if constexpr (!isSetMask) {
        // get SPR.MASK in VF
        MicroAPI::MaskReg sprLoadMaskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(maskBuf, sprLoadMaskReg);
        // insert membar(vec store operation) before load maskBuf[0](scalar load operation)
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        countSreg = static_cast<uint32_t>(maskBuf[0]);
    }
    uint16_t oneRepSize = GetVecLen() / sizeof(SRC_TYPE);
    if constexpr (sizeof(SRC_TYPE) < sizeof(DST_TYPE)) {
        oneRepSize = GetVecLen() / sizeof(DST_TYPE);
    }
    uint16_t newRepeatTimes = CeilDivision(countSreg, oneRepSize);
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (sizeof(DST_TYPE) == sizeof(SRC_TYPE)) {
            ldPreg = MicroAPI::UpdateMask<SRC_TYPE>(countSreg);
            exPreg = ldPreg;
            stPreg = ldPreg;
        } else if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
            ldPreg = MicroAPI::UpdateMask<SRC_TYPE>(countSreg);
            exPreg = ldPreg;
            MicroAPI::MaskPack(stPreg, ldPreg);
            if constexpr ((SupportType<DST_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(SRC_TYPE) == 2) ||
                          (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4)) {
                MicroAPI::MaskPack(stPreg, stPreg);
            }
        } else if constexpr (sizeof(DST_TYPE) > sizeof(SRC_TYPE)) {
            stPreg = MicroAPI::UpdateMask<DST_TYPE>(countSreg);
            exPreg = stPreg;
            MicroAPI::MaskPack(ldPreg, stPreg);
            if constexpr ((SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(DST_TYPE) == 2) ||
                          (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4)) {
                MicroAPI::MaskPack(ldPreg, ldPreg);
                if constexpr (SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(DST_TYPE) == 2) {
                    MicroAPI::MaskUnPack(stPreg, ldPreg);
                    MicroAPI::MaskUnPack(exPreg, stPreg);
                    MicroAPI::MaskInterleave<uint16_t>(stPreg, dumpPreg, stPreg, stPreg);
                }
            }
        }
        GenLoadL0<DST_TYPE, SRC_TYPE>(srcVreg, src, ldPreg, repeatParams);
        if constexpr (std::is_same_v<SRC_TYPE, int32_t> && std::is_same_v<DST_TYPE, half>) {
            MicroAPI::Cast<float, SRC_TYPE, castTrait>((MicroAPI::RegTensor<float> &)dstVreg, srcVreg, exPreg);
            MicroAPI::Cast<DST_TYPE, float, castTrait>(dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, exPreg);
        } else if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<DST_TYPE, float>) {
            MicroAPI::Truncate<DST_TYPE, roundMode>(dstVreg, srcVreg, exPreg);
        } else {
            MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, exPreg);
        }
        GenStoreL0<DST_TYPE, SRC_TYPE>(dst, dstVreg, stPreg, repeatParams);
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__aicore__ inline void CastIntrinsicsImpl(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask[],
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    constexpr bool b64Cast = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<float, int64_t>, Tuple<int64_t, float>,
        Tuple<int32_t, int64_t>, Tuple<int64_t, int32_t>>();
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        __ubuf__ uint64_t *maskBuf = nullptr;
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
        }
        if constexpr (b64Cast) {
            VF_CALL<CastIntrinsicsB64ImplCounterVF<DST_TYPE, SRC_TYPE, roundMode, isSetMask>>(
                dst, src, mask[0], maskBuf, repeatTime, repeatParams);
        } else {
            VF_CALL<CastIntrinsicsImplCounterVF<DST_TYPE, SRC_TYPE, roundMode, isSetMask>>(
                dst, src, mask[0], maskBuf, repeatTime, repeatParams);
        }
    } else {
        if constexpr (b64Cast) {
            if constexpr (isSetMask) {
                SetVectorMask<uint32_t>(mask[1], mask[0]);
            }
            VF_CALL<CastIntrinsicsB64ImplVF2<DST_TYPE, SRC_TYPE, roundMode>>(dst, src, mask, repeatTime, repeatParams);
        } else {
            if constexpr (isSetMask) {
                if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
                    SetVectorMask<SRC_TYPE>(mask[1], mask[0]);
                } else {
                    SetVectorMask<DST_TYPE>(mask[1], mask[0]);
                }
            }
            VF_CALL<CastIntrinsicsImplVF2<DST_TYPE, SRC_TYPE, roundMode>>(
                dst, src, mask, repeatTime, repeatParams);
        }
    }
}

// Cast::Level 0 - mask bit mode
template <typename ORI_DST_TYPE, typename ORI_SRC_TYPE, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ ORI_DST_TYPE *oriDst, __ubuf__ ORI_SRC_TYPE *oriSrc, const RoundMode &roundMode,
    const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    using DST_TYPE = typename CastParam::CastTypeTrait<ORI_DST_TYPE>::RealType;
    using SRC_TYPE = typename CastParam::CastTypeTrait<ORI_SRC_TYPE>::RealType;
    __ubuf__ DST_TYPE* dst = reinterpret_cast<__ubuf__ DST_TYPE*>(oriDst);
    __ubuf__ SRC_TYPE* src = reinterpret_cast<__ubuf__ SRC_TYPE*>(oriSrc);

    constexpr bool cast_round_all = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>, Tuple<int64_t, float>,
        Tuple<int32_t, float>, Tuple<int16_t, float>, Tuple<bfloat16_t, float>, Tuple<int32_t, half>,
        Tuple<int16_t, half>, Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4x2_t, half>, Tuple<bfloat16_t, half>,
        Tuple<half, int16_t>, Tuple<float, int32_t>,Tuple<float, int64_t>, Tuple<int32_t, bfloat16_t>,
        Tuple<half, bfloat16_t>, Tuple<fp4x2_e1m2_t, bfloat16_t>,Tuple<fp4x2_e2m1_t, bfloat16_t>,
        Tuple<half, int32_t>, Tuple<float, float>>();
    constexpr bool cast_none = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<float, half>, Tuple<float, bfloat16_t>,
        Tuple<half, int4x2_t>, Tuple<half, uint8_t>, Tuple<uint16_t, uint8_t>, Tuple<uint32_t, uint8_t>,
        Tuple<half, int8_t>, Tuple<int16_t, int8_t>, Tuple<int32_t, int8_t>, Tuple<uint8_t, uint16_t>,
        Tuple<uint32_t, uint16_t>, Tuple<float, int16_t>, Tuple<uint8_t, int16_t>, Tuple<uint32_t, int16_t>,
        Tuple<int32_t, int16_t>, Tuple<uint8_t, uint32_t>, Tuple<uint16_t, uint32_t>, Tuple<int16_t, uint32_t>,
        Tuple<int64_t, int32_t>, Tuple<int16_t, int32_t>, Tuple<uint8_t, int32_t>, Tuple<uint16_t, int32_t>,
        Tuple<int32_t, int64_t>, Tuple<half, hifloat8_t>, Tuple<float, hifloat8_t>, Tuple<float, fp8_e4m3fn_t>,
        Tuple<float, fp8_e5m2_t>, Tuple<bfloat16_t, fp4x2_e1m2_t>, Tuple<bfloat16_t, fp4x2_e2m1_t>,
        Tuple<int4x2_t, int16_t>, Tuple<int16_t, int4x2_t>, Tuple<bfloat16_t, int4x2_t>>();
    constexpr bool using_cast_rint =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4x2_t, half>,
        Tuple<half, float>, Tuple<half, int16_t>, Tuple<float, int32_t>>();
    constexpr bool cast_odd = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>>();
    constexpr bool cast_rint = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<fp8_e5m2_t, float>,
        Tuple<fp8_e4m3fn_t, float>>();
    constexpr bool cast_round =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<hifloat8_t, float>, Tuple<hifloat8_t, half>>();
    constexpr bool cast_hybrid =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<hifloat8_t, float>, Tuple<hifloat8_t, half>>();
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            if constexpr (cast_round_all || cast_rint) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast rint"); });
            }
            break;
        case RoundMode::CAST_FLOOR:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_FLOOR, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast floor"); });
            }
            break;
        case RoundMode::CAST_CEIL:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_CEIL, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast ceil"); });
            }
            break;
        case RoundMode::CAST_ROUND:
            if constexpr (cast_round_all || cast_round) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ROUND, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast round"); });
            }
            break;
        case RoundMode::CAST_TRUNC:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_TRUNC, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast trunc"); });
            }
            break;
        case RoundMode::CAST_ODD:
            if constexpr (cast_odd) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ODD, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast odd"); });
            }
            break;
        case RoundMode::CAST_NONE:
            if constexpr (cast_none) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_NONE, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else if constexpr (using_cast_rint) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast none"); });
            }
            break;
        case RoundMode::CAST_HYBRID:
            if constexpr (cast_hybrid) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_HYBRID, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast hybrid"); });
            }
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__aicore__ inline void CastIntrinsicsB64ImplVF1(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask,
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    uint32_t b32Sreg = static_cast<uint32_t>(mask);
    uint32_t b64Sreg = static_cast<uint32_t>(2 * mask);
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    MicroAPI::RegTensor<uint32_t> zeroVreg, tmpVreg;
    MicroAPI::MaskReg b32Preg;
    MicroAPI::MaskReg b64Preg, tmpPreg;
    if constexpr (isSetMask) {
        b32Preg = MicroAPI::UpdateMask<uint32_t>(b32Sreg);
        b64Preg = MicroAPI::UpdateMask<uint32_t>(b64Sreg);
    } else {
        b32Preg = MicroAPI::MoveMask<uint32_t>();
        MicroAPI::MaskInterleave<uint32_t>(b64Preg, tmpPreg, b32Preg, b32Preg);
    }
    MicroAPI::MaskReg fullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroVreg, 0, fullPreg);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        if constexpr (sizeof(DST_TYPE) == sizeof(int64_t)) {
            // b32-> b64
            MicroAPI::LoadAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<uint32_t> &)srcVreg,
                (__ubuf__ uint32_t *&)src,
                static_cast<uint32_t>(repeatParams.srcBlkStride),
                static_cast<uint32_t>(repeatParams.srcRepStride),
                b32Preg);
            MicroAPI::Interleave(
                (MicroAPI::RegTensor<uint32_t> &)srcVreg, tmpVreg, (MicroAPI::RegTensor<uint32_t> &)srcVreg, zeroVreg);
        } else {
            // b64 -> b32
            MicroAPI::LoadAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<uint32_t> &)srcVreg,
                (__ubuf__ uint32_t *&)src,
                static_cast<uint32_t>(repeatParams.srcBlkStride),
                static_cast<uint32_t>(repeatParams.srcRepStride),
                b64Preg);
        }
        MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, b64Preg);
        if constexpr (sizeof(DST_TYPE) == sizeof(int64_t)) {
            // b32-> b64
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::StoreAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)dst,
                (MicroAPI::RegTensor<uint32_t> &)dstVreg,
                static_cast<uint32_t>(repeatParams.dstBlkStride),
                static_cast<uint32_t>(repeatParams.dstRepStride),
                b64Preg);
        } else {
            // b64 -> b32
            MicroAPI::DeInterleave(
                (MicroAPI::RegTensor<uint32_t> &)dstVreg, tmpVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg, zeroVreg);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::StoreAlign<uint32_t,
                MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ uint32_t *&)dst,
                (MicroAPI::RegTensor<uint32_t> &)dstVreg,
                static_cast<uint32_t>(repeatParams.dstBlkStride),
                static_cast<uint32_t>(repeatParams.dstRepStride),
                b32Preg);
        }
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__aicore__ inline void CastIntrinsicsImplVF1(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask,
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    uint32_t sreg = static_cast<uint32_t>(mask);
    MicroAPI::MaskReg ldPreg;
    MicroAPI::MaskReg exPreg;
    MicroAPI::MaskReg stPreg;
    MicroAPI::MaskReg dumpPreg;
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    if constexpr (sizeof(DST_TYPE) == sizeof(SRC_TYPE)) {
        if constexpr (isSetMask) {
            ldPreg = MicroAPI::UpdateMask<SRC_TYPE>(sreg);
        } else {
            ldPreg = MicroAPI::MoveMask<SRC_TYPE>();
        }
        exPreg = ldPreg;
        stPreg = ldPreg;
    } else if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
        if constexpr (isSetMask) {
            ldPreg = MicroAPI::UpdateMask<SRC_TYPE>(sreg);
        } else {
            ldPreg = MicroAPI::MoveMask<SRC_TYPE>();
        }
        exPreg = ldPreg;
        MicroAPI::MaskPack(stPreg, ldPreg);
        if constexpr ((SupportType<DST_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(SRC_TYPE) == 2) ||
                      (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4)) {
            MicroAPI::MaskPack(stPreg, stPreg);
        }
    } else if constexpr (sizeof(DST_TYPE) > sizeof(SRC_TYPE)) {
        if constexpr (isSetMask) {
            stPreg = MicroAPI::UpdateMask<DST_TYPE>(sreg);
        } else {
            stPreg = MicroAPI::MoveMask<DST_TYPE>();
        }
        exPreg = stPreg;
        MicroAPI::MaskPack(ldPreg, stPreg);
        if constexpr ((SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(DST_TYPE) == 2) ||
                      (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4)) {
            MicroAPI::MaskPack(ldPreg, ldPreg);
            if constexpr (SupportType<SRC_TYPE, int4x2_t, fp4x2_e2m1_t, fp4x2_e1m2_t>() && sizeof(DST_TYPE) == 2) {
                MicroAPI::MaskUnPack(stPreg, ldPreg);
                MicroAPI::MaskUnPack(exPreg, stPreg);
                MicroAPI::MaskInterleave<uint16_t>(stPreg, dumpPreg, stPreg, stPreg);
            }
        }
    }
    for (uint16_t i = 0; i < repeatTime; ++i) {
        GenLoadL0<DST_TYPE, SRC_TYPE>(srcVreg, src, ldPreg, repeatParams);
        if constexpr (std::is_same_v<SRC_TYPE, int32_t> && std::is_same_v<DST_TYPE, half>) {
            MicroAPI::Cast<float, SRC_TYPE, castTrait>((MicroAPI::RegTensor<float> &)dstVreg, srcVreg, exPreg);
            MicroAPI::Cast<DST_TYPE, float, castTrait>(dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, exPreg);
        } else if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<DST_TYPE, float>) {
            MicroAPI::Truncate<DST_TYPE, roundMode>(dstVreg, srcVreg, exPreg);
        } else {
            MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, exPreg);
        }
        GenStoreL0<DST_TYPE, SRC_TYPE>(dst, dstVreg, stPreg, repeatParams);
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__aicore__ inline void CastIntrinsicsImpl(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask,
    uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    constexpr bool b64Cast = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<float, int64_t>, Tuple<int64_t, float>,
        Tuple<int32_t, int64_t>, Tuple<int64_t, int32_t>>();
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        __ubuf__ uint64_t *maskBuf = nullptr;
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 2);
        }
        if constexpr (b64Cast) {
            VF_CALL<CastIntrinsicsB64ImplCounterVF<DST_TYPE, SRC_TYPE, roundMode, isSetMask>>(
                dst, src, mask, maskBuf, repeatTime, repeatParams);
        } else {
            VF_CALL<CastIntrinsicsImplCounterVF<DST_TYPE, SRC_TYPE, roundMode, isSetMask>>(
                dst, src, mask, maskBuf, repeatTime, repeatParams);
        }
    } else {
        if constexpr (b64Cast) {
            VF_CALL<CastIntrinsicsB64ImplVF1<DST_TYPE, SRC_TYPE, roundMode, isSetMask>>(
                dst, src, mask, repeatTime, repeatParams);
        } else {
            VF_CALL<CastIntrinsicsImplVF1<DST_TYPE, SRC_TYPE, roundMode, isSetMask>>(
                dst, src, mask, repeatTime, repeatParams);
        }
    }
}

// Cast::Level 0 - mask count mode
template <typename ORI_DST_TYPE, typename ORI_SRC_TYPE, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ ORI_DST_TYPE *oriDst, __ubuf__ ORI_SRC_TYPE *oriSrc, const RoundMode &roundMode,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    using DST_TYPE = typename CastParam::CastTypeTrait<ORI_DST_TYPE>::RealType;
    using SRC_TYPE = typename CastParam::CastTypeTrait<ORI_SRC_TYPE>::RealType;
    __ubuf__ DST_TYPE* dst = reinterpret_cast<__ubuf__ DST_TYPE*>(oriDst);
    __ubuf__ SRC_TYPE* src = reinterpret_cast<__ubuf__ SRC_TYPE*>(oriSrc);

    constexpr bool cast_round_all = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>, Tuple<int64_t, float>,
        Tuple<int32_t, float>, Tuple<int16_t, float>, Tuple<bfloat16_t, float>, Tuple<int32_t, half>,
        Tuple<int16_t, half>, Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4x2_t, half>, Tuple<bfloat16_t, half>,
        Tuple<half, int16_t>, Tuple<float, int32_t>,Tuple<float, int64_t>, Tuple<int32_t, bfloat16_t>,
        Tuple<half, bfloat16_t>, Tuple<fp4x2_e1m2_t, bfloat16_t>,Tuple<fp4x2_e2m1_t, bfloat16_t>,
        Tuple<half, int32_t>, Tuple<float, float>>();
    constexpr bool cast_none = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<float, half>, Tuple<float, bfloat16_t>,
        Tuple<half, int4x2_t>, Tuple<half, uint8_t>, Tuple<uint16_t, uint8_t>, Tuple<uint32_t, uint8_t>,
        Tuple<half, int8_t>, Tuple<int16_t, int8_t>, Tuple<int32_t, int8_t>, Tuple<uint8_t, uint16_t>,
        Tuple<uint32_t, uint16_t>, Tuple<float, int16_t>, Tuple<uint8_t, int16_t>, Tuple<uint32_t, int16_t>,
        Tuple<int32_t, int16_t>, Tuple<uint8_t, uint32_t>, Tuple<uint16_t, uint32_t>, Tuple<int16_t, uint32_t>,
        Tuple<int64_t, int32_t>, Tuple<int16_t, int32_t>, Tuple<uint8_t, int32_t>, Tuple<uint16_t, int32_t>,
        Tuple<int32_t, int64_t>, Tuple<half, hifloat8_t>, Tuple<float, hifloat8_t>, Tuple<float, fp8_e4m3fn_t>,
        Tuple<float, fp8_e5m2_t>, Tuple<bfloat16_t, fp4x2_e1m2_t>, Tuple<bfloat16_t, fp4x2_e2m1_t>,
        Tuple<int4x2_t, int16_t>, Tuple<int16_t, int4x2_t>, Tuple<bfloat16_t, int4x2_t>>();
    constexpr bool using_cast_rint =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4x2_t, half>,
        Tuple<half, float>, Tuple<half, int16_t>, Tuple<float, int32_t>>();
    constexpr bool cast_odd = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>>();
    constexpr bool cast_rint = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<fp8_e5m2_t, float>,
        Tuple<fp8_e4m3fn_t, float>>();
    constexpr bool cast_round =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<hifloat8_t, float>, Tuple<hifloat8_t, half>>();
    constexpr bool cast_hybrid =
        SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<hifloat8_t, float>, Tuple<hifloat8_t, half>>();
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            if constexpr (cast_round_all || cast_rint) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast rint"); });
            }
            break;
        case RoundMode::CAST_FLOOR:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_FLOOR, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast floor"); });
            }
            break;
        case RoundMode::CAST_CEIL:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_CEIL, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast ceil"); });
            }
            break;
        case RoundMode::CAST_ROUND:
            if constexpr (cast_round_all || cast_round) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ROUND, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast round"); });
            }
            break;
        case RoundMode::CAST_TRUNC:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_TRUNC, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast trunc"); });
            }
            break;
        case RoundMode::CAST_ODD:
            if constexpr (cast_odd) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ODD, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast odd"); });
            }
            break;
        case RoundMode::CAST_NONE:
            if constexpr (cast_none) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_NONE, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else if constexpr (using_cast_rint) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast none"); });
            }
            break;
        case RoundMode::CAST_HYBRID:
            if constexpr (cast_hybrid) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_HYBRID, isSetMask>(dst, src, mask, repeatTime, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast hybrid"); });
            }
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

// scale is stored as  1 sign bit, 8 exponent bits and 10 mantissa bits in 1971 hardware
// ===============================================================================
// | 1 signMode bit  | 8 exponent bits | 10bit mantissa bits |       000..0       |
// ===============================================================================
__aicore__ inline float GetCastDeqScale(const uint64_t deqScale)
{
    uint32_t tmp = static_cast<uint32_t>(deqScale & 0xffffffff);
    tmp = tmp & 0xffffe000;
    float ret = *(reinterpret_cast<float *>(&tmp));
    return ret;
}

__aicore__ inline int16_t GetCastDeqOffset(const uint64_t deqScale)
{
    int16_t ret = static_cast<int16_t>((deqScale >> 37) & 0x1ff);
    return ret;
}

__aicore__ inline bool GetCastDeqSignMode(const uint64_t deqScale)
{
    bool ret = static_cast<bool>(deqScale >> 46);
    return ret;
}

template <typename T, MicroAPI::HighLowPart part>
__aicore__ inline void CastDeqMulsCal(
    MicroAPI::RegTensor<float> &tmpReg, MicroAPI::RegTensor<T> &srcReg, MicroAPI::MaskReg &maskReg, const float scale)
{
    MicroAPI::Cast<float, T, CastParam::s162f32CastTrait>(tmpReg, srcReg, maskReg);
    MicroAPI::Muls(tmpReg, tmpReg, scale, maskReg);
    MicroAPI::Cast<T, float, CastParam::f322s16CastTrait>(srcReg, tmpReg, maskReg);
    MicroAPI::Pack<uint16_t, uint32_t, part>(
        (MicroAPI::RegTensor<uint16_t> &)srcReg, (MicroAPI::RegTensor<uint32_t> &)srcReg);
}

template <typename T, MicroAPI::HighLowPart part>
__aicore__ inline void CastVecDeqMulsCal(MicroAPI::RegTensor<int32_t> &tmpReg, MicroAPI::RegTensor<T> &srcReg,
    MicroAPI::MaskReg &maskReg, MicroAPI::RegTensor<float> &scaleReg)
{
    MicroAPI::Cast<float, T, CastParam::s162f32CastTrait>((MicroAPI::RegTensor<float> &)tmpReg, srcReg, maskReg);
    MicroAPI::Mul((MicroAPI::RegTensor<float> &)tmpReg, (MicroAPI::RegTensor<float> &)tmpReg, scaleReg, maskReg);
    MicroAPI::Cast<T, float, CastParam::f322s16CastTrait>(srcReg, (MicroAPI::RegTensor<float> &)tmpReg, maskReg);
    MicroAPI::Pack<uint16_t, uint32_t, part>(
        (MicroAPI::RegTensor<uint16_t> &)srcReg, (MicroAPI::RegTensor<uint32_t> &)srcReg);
}

template <typename U, typename T, bool halfBlock, bool signMode>
__aicore__ inline void CastFromS92B8(
    MicroAPI::RegTensor<T> &srcReg, MicroAPI::RegTensor<U> &dstReg, MicroAPI::MaskReg &maskReg)
{
    if constexpr (signMode) {
        if constexpr (halfBlock) {
            MicroAPI::Cast<half, T, CastParam::s162HalfTrait>((MicroAPI::RegTensor<half>&)dstReg, srcReg, maskReg);
            MicroAPI::Cast<int8_t, half, CastParam::TrueHalfBlockHalf2S8Trait>(
                (MicroAPI::RegTensor<int8_t>&)dstReg, (MicroAPI::RegTensor<half>&)dstReg, maskReg);
        } else {
            MicroAPI::Cast<half, T, CastParam::s162HalfTrait>((MicroAPI::RegTensor<half>&)dstReg, srcReg, maskReg);
            MicroAPI::Cast<int8_t, half, CastParam::FalseHalfBlockHalf2S8Trait>(
                (MicroAPI::RegTensor<int8_t>&)dstReg, (MicroAPI::RegTensor<half>&)dstReg, maskReg);
        }
    } else {
        if constexpr (halfBlock) {
            MicroAPI::Cast<uint8_t, T, CastParam::TrueHalfBlockCastTrait>((MicroAPI::RegTensor<uint8_t>&)dstReg, srcReg, maskReg);
        } else {
            MicroAPI::Cast<uint8_t, T, CastParam::FalseHalfBlockCastTrait>((MicroAPI::RegTensor<uint8_t>&)dstReg, srcReg, maskReg);
        }
    }
}

// This function is used to generate an index array
// Here is the following python pseudocode:
// BLOCK_SIZE = 32
// HALF_BLOCK_SIZE = 16
// block_id = int(index / BLOCK_SIZE)
// out[index] = block_id * BLOCK_SIZE + (index % HALF_BLOCK_SIZE) * 2 + (index // HALF_BLOCK_SIZE) % 2
__aicore__ inline void GenGatherIndex(MicroAPI::RegTensor<int8_t>& dstReg)
{
    MicroAPI::RegTensor<int8_t> arangeReg;
    MicroAPI::RegTensor<int8_t> lowHalfReg;
    MicroAPI::RegTensor<int8_t> dupReg;
    MicroAPI::RegTensor<int8_t> highHalfReg;
    MicroAPI::RegTensor<int8_t> baseOffsetReg;
    MicroAPI::MaskReg mask;
    constexpr int8_t dupConstant = 15;
    constexpr int8_t one = 1;
    constexpr int16_t highHalfRegShiftLeftScalar = 1;
    constexpr int8_t firstIndex = 0;
    constexpr int16_t arangeShiftScalar = 5;
    constexpr int16_t highHalfRegShiftRightScalar = 4;

    mask = MicroAPI::CreateMask<int8_t>();
    MicroAPI::Arange(arangeReg, firstIndex);
    MicroAPI::ShiftRights(baseOffsetReg, arangeReg, arangeShiftScalar, mask);
    MicroAPI::ShiftLefts(baseOffsetReg, baseOffsetReg, arangeShiftScalar, mask);

    MicroAPI::Duplicate(dupReg, dupConstant);
    MicroAPI::And(lowHalfReg, arangeReg, dupReg, mask);
    MicroAPI::ShiftLefts(lowHalfReg, lowHalfReg, highHalfRegShiftLeftScalar, mask);
    MicroAPI::ShiftRights(highHalfReg, arangeReg, highHalfRegShiftRightScalar, mask);
    MicroAPI::Duplicate(dupReg, one);
    MicroAPI::And(highHalfReg, highHalfReg, dupReg, mask);
    MicroAPI::Add(lowHalfReg, lowHalfReg, highHalfReg, mask);
    MicroAPI::Add(dstReg, lowHalfReg, baseOffsetReg, mask);
}

__aicore__ inline void GenVecCastDeqParam(uint64_t deqScaleAddr, MicroAPI::RegTensor<float> &scaleReg,
    MicroAPI::RegTensor<int16_t> &offsetReg, MicroAPI::MaskReg &signMask, MicroAPI::MaskReg &unSignMask,
    MicroAPI::RegTensor<int32_t> &tmpReg, MicroAPI::MaskReg &fullMask)
{
    constexpr int16_t offsetShiftLeftScalar = 18;
    constexpr int16_t offsetShiftRightScalar = 55;
    constexpr int16_t signModeShiftRightScalar = 46;
    constexpr int16_t gatherConstant = 16;
    constexpr int16_t zero = 0;
    constexpr int16_t one = 1;
    constexpr int32_t scaleMask = 0xffffe000;
    MicroAPI::RegTensor<int32_t> scaleIndexReg;
    MicroAPI::RegTensor<int16_t> offsetIndexReg;
    MicroAPI::RegTensor<int16_t> signModeReg;
    MicroAPI::LoadAlign(scaleReg, (__ubuf__ float *)deqScaleAddr);
    MicroAPI::Pack((MicroAPI::RegTensor<uint32_t> &)scaleReg, (MicroAPI::RegTensor<uint64_t> &)scaleReg);

    MicroAPI::LoadAlign(offsetReg, (__ubuf__ int16_t *)deqScaleAddr);
    signMask = MicroAPI::CreateMask<uint64_t, MicroAPI::MaskPattern::VL16>();
    MicroAPI::ShiftLefts((MicroAPI::RegTensor<uint64_t> &)offsetReg, (MicroAPI::RegTensor<uint64_t> &)offsetReg, offsetShiftLeftScalar, signMask);
    MicroAPI::ShiftRights((MicroAPI::RegTensor<uint64_t> &)offsetReg, (MicroAPI::RegTensor<uint64_t> &)offsetReg, offsetShiftRightScalar, signMask);
    MicroAPI::Pack((MicroAPI::RegTensor<uint32_t> &)offsetReg, (MicroAPI::RegTensor<uint64_t> &)offsetReg);
    MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)offsetReg, (MicroAPI::RegTensor<uint32_t> &)offsetReg);

    MicroAPI::LoadAlign(signModeReg, (__ubuf__ int16_t *)deqScaleAddr);
    MicroAPI::ShiftRights((MicroAPI::RegTensor<uint64_t> &)signModeReg, (MicroAPI::RegTensor<uint64_t> &)signModeReg, 
        signModeShiftRightScalar, signMask);
    MicroAPI::Pack((MicroAPI::RegTensor<uint32_t> &)signModeReg, (MicroAPI::RegTensor<uint64_t> &)signModeReg);
    MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)signModeReg, (MicroAPI::RegTensor<uint32_t> &)signModeReg);

    // Gen b32 fullMask to deal scaleReg (which datatype is float)
    fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Arange(scaleIndexReg, zero);
    MicroAPI::Duplicate(tmpReg, gatherConstant);
    MicroAPI::Div(tmpReg, scaleIndexReg, tmpReg, fullMask);
    MicroAPI::Muls(tmpReg, tmpReg, gatherConstant, fullMask);
    MicroAPI::Sub(scaleIndexReg, scaleIndexReg, tmpReg, fullMask);
    MicroAPI::Gather((MicroAPI::RegTensor<uint32_t> &)scaleReg, (MicroAPI::RegTensor<uint32_t> &)scaleReg, (MicroAPI::RegTensor<uint32_t> &)scaleIndexReg);
    MicroAPI::Duplicate(tmpReg, scaleMask);
    MicroAPI::And((MicroAPI::RegTensor<int32_t> &)scaleReg, (MicroAPI::RegTensor<int32_t> &)scaleReg, tmpReg, fullMask);

    // Pack lowest 16bit of mrg2ChnIndexReg, because in API "Gather", the sizeof(offsetReg) should be the same as sizeof(mrg2ChnIndexReg)
    fullMask = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Arange(offsetIndexReg, zero);
    MicroAPI::Duplicate((MicroAPI::RegTensor<int16_t> &)tmpReg, gatherConstant);
    MicroAPI::Div((MicroAPI::RegTensor<int16_t> &)tmpReg, offsetIndexReg, (MicroAPI::RegTensor<int16_t> &)tmpReg, fullMask);
    MicroAPI::Muls((MicroAPI::RegTensor<int16_t> &)tmpReg, (MicroAPI::RegTensor<int16_t> &)tmpReg, gatherConstant, fullMask);
    MicroAPI::Sub(offsetIndexReg, offsetIndexReg, (MicroAPI::RegTensor<int16_t> &)tmpReg, fullMask);
    MicroAPI::Gather((MicroAPI::RegTensor<uint16_t> &)offsetReg, (MicroAPI::RegTensor<uint16_t> &)offsetReg, (MicroAPI::RegTensor<uint16_t> &)offsetIndexReg);

    MicroAPI::Gather((MicroAPI::RegTensor<uint16_t> &)signModeReg, (MicroAPI::RegTensor<uint16_t> &)signModeReg, (MicroAPI::RegTensor<uint16_t> &)offsetIndexReg);
    // Gen mask for elements which cast to sign and cast to unsign
    MicroAPI::Duplicate((MicroAPI::RegTensor<int16_t> &)tmpReg, one);
    MicroAPI::Compare(signMask, signModeReg, (MicroAPI::RegTensor<int16_t> &)tmpReg, fullMask);
    MicroAPI::Duplicate((MicroAPI::RegTensor<int16_t> &)tmpReg, zero);
    MicroAPI::Compare(unSignMask, signModeReg, (MicroAPI::RegTensor<int16_t> &)tmpReg, fullMask);
}

template <bool halfBlock>
__aicore__ inline void GenLevel0StoreMask(MicroAPI::MaskReg &srcMask, MicroAPI::MaskReg &dstMask,
    MicroAPI::RegTensor<uint8_t> &mrg2ChnIndexReg, MicroAPI::RegTensor<uint8_t> &tmpReg, MicroAPI::MaskReg &fullMask)
{
    constexpr uint8_t cmpScalar = 1;
    if constexpr (halfBlock) {
        constexpr uint16_t scalar = 0x0100;
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t> &)tmpReg, scalar, srcMask);
    } else {
        constexpr uint16_t scalar = 0x0001;
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t> &)tmpReg, scalar, srcMask);
    }
    MicroAPI::Gather(tmpReg, tmpReg, mrg2ChnIndexReg);
    MicroAPI::CompareScalar(dstMask, tmpReg, cmpScalar, fullMask);
}

template <typename U, typename T, bool halfBlock>
__aicore__ inline void CastVecDeqImplVF(
    __ubuf__ U *dst, __ubuf__ T *src, const uint32_t calCount, uint64_t deqScaleAddr)
{
    MicroAPI::RegTensor<U> dstReg;
    MicroAPI::RegTensor<T> srcReg0, srcReg1;
    MicroAPI::RegTensor<float> scaleReg;
    MicroAPI::RegTensor<int16_t> offsetReg, signDstReg, vAndReg;
    MicroAPI::RegTensor<uint16_t> unSignDstReg;
    MicroAPI::RegTensor<int32_t> tmpReg;
    MicroAPI::RegTensor<uint8_t> mrg2ChnIndexReg;
    MicroAPI::MaskReg maskReg0, maskReg1, maskReg2, fullMask, signMask, unSignMask;
 
    constexpr int16_t s9MaxValue = 255;
    constexpr int16_t s9MinValue = -256;
    constexpr int16_t unRollConstant = 2;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t halfRepSize = GetVecLen() / unRollConstant / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, oneRepSize);
    GenVecCastDeqParam(deqScaleAddr, scaleReg, offsetReg, signMask, unSignMask, tmpReg, fullMask);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::Duplicate((MicroAPI::RegTensor<int16_t> &)vAndReg, 0x00ff);
    GenGatherIndex((MicroAPI::RegTensor<int8_t>&)mrg2ChnIndexReg);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskReg0 = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::MaskInterleave<T>(maskReg1, maskReg2, maskReg0, fullMask);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_US_B16>(srcReg0, src + i * unRollConstant * halfRepSize);
        CastVecDeqMulsCal<T, MicroAPI::HighLowPart::LOWEST>(tmpReg, srcReg0, maskReg1, scaleReg);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_US_B16>(srcReg1, src + (i * unRollConstant + 1) * halfRepSize);
        CastVecDeqMulsCal<T, MicroAPI::HighLowPart::HIGHEST>(tmpReg, srcReg1, maskReg2, scaleReg);
        maskReg1 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::H>();
        MicroAPI::Select(srcReg0, srcReg0, srcReg1, maskReg1);
        MicroAPI::Maxs(srcReg0, srcReg0, s9MinValue, maskReg0);
        MicroAPI::Mins(srcReg0, srcReg0, s9MaxValue, maskReg0);
        MicroAPI::Add(srcReg0, srcReg0, offsetReg, maskReg0);
        CastFromS92B8<int8_t, T, halfBlock, true>(srcReg0, (MicroAPI::RegTensor<int8_t> &)signDstReg, signMask);
        CastFromS92B8<uint8_t, T, halfBlock, false>(srcReg0, (MicroAPI::RegTensor<uint8_t> &)unSignDstReg, unSignMask);
        MicroAPI::Select((MicroAPI::RegTensor<int16_t> &)dstReg, signDstReg, (MicroAPI::RegTensor<int16_t> &)unSignDstReg, signMask);
        Gather(dstReg, dstReg, mrg2ChnIndexReg);
        MicroAPI::StoreAlign((__ubuf__ T*)dst + i * oneRepSize, (MicroAPI::RegTensor<T> &)dstReg, fullMask);
    }
}

template <typename U, typename T, bool halfBlock, bool signMode>
__aicore__ inline void CastDeqImplVF(__ubuf__ U *dst, __ubuf__ T *src, const uint32_t calCount, uint64_t deqScale)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<U> dstReg;
    MicroAPI::RegTensor<uint8_t> mrg2ChnIndexReg;
    MicroAPI::MaskReg maskReg0, maskReg1, maskReg2, fullMask;

    constexpr int16_t s9MaxValue = 255;
    constexpr int16_t s9MinValue = -256;
    constexpr int16_t unRollConstant = 2;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t halfRepSize = GetVecLen() / unRollConstant / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, oneRepSize);
    float scale = GetCastDeqScale(deqScale);
    uint16_t offset = GetCastDeqOffset(deqScale);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    GenGatherIndex((MicroAPI::RegTensor<int8_t>&)mrg2ChnIndexReg);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskReg0 = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::MaskInterleave<T>(maskReg1, maskReg2, maskReg0, fullMask);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_US_B16>(srcReg0, src + i * unRollConstant * halfRepSize);
        CastDeqMulsCal<T, MicroAPI::HighLowPart::LOWEST>(tmpReg, srcReg0, maskReg1, scale);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_US_B16>(srcReg1, src + (i * unRollConstant + 1) * halfRepSize);
        CastDeqMulsCal<T, MicroAPI::HighLowPart::HIGHEST>(tmpReg, srcReg1, maskReg2, scale);
        MicroAPI::Or(srcReg0, srcReg0, srcReg1, maskReg0);
        MicroAPI::Maxs(srcReg0, srcReg0, s9MinValue, maskReg0);
        MicroAPI::Mins(srcReg0, srcReg0, s9MaxValue, maskReg0);
        MicroAPI::Adds(srcReg0, srcReg0, offset, maskReg0);
        CastFromS92B8<U, T, halfBlock, signMode>(srcReg0, dstReg, maskReg0);
        Gather(dstReg, dstReg, mrg2ChnIndexReg);
        MicroAPI::StoreAlign((__ubuf__ T*)dst + i * oneRepSize, (MicroAPI::RegTensor<T>&)dstReg, fullMask);
    }
}

template <typename U, typename T>
__aicore__ inline void CastDeqS322f16ImplVF(__ubuf__ U *dst, __ubuf__ T *src, const uint32_t calCount, const half deqScale)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<U> dstReg;
    MicroAPI::MaskReg maskReg;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t repeatTime = CeilDivision(calCount, oneRepSize);
    uint32_t sreg = calCount;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::LoadAlign(srcReg, src + i * oneRepSize);
        MicroAPI::Cast<float, T, CastParam::s322F32CastTrait>(tmpReg, srcReg, maskReg);
        MicroAPI::Cast<U, float, CastParam::f322F16CastTrait>(dstReg, tmpReg, maskReg);
        MicroAPI::Muls(dstReg, dstReg, deqScale, maskReg);
        MicroAPI::StoreAlign<U, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * oneRepSize, dstReg, maskReg);
    }
}

template <typename U, typename T, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U *dst, __ubuf__ T *src, const uint32_t calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<int16_t, int8_t>, Tuple<int16_t, uint8_t>, Tuple<int32_t, half>>(),
        "Failed to check dtype in CastDeqImpl, current api support dtype combination is src: int16_t dst: int8_t/uint8_t"
        ", src:int32_t dst:half.");
    if constexpr (IsSameType<T, int32_t>::value) {
        half scale = g_deqValue;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        VF_CALL<CastDeqS322f16ImplVF<U, T>>(dst, src, calCount, scale);
    } else {
        uint64_t deqScale = Internal::g_deqScale;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        if constexpr (isVecDeq) {
            VF_CALL<CastVecDeqImplVF<U, T, halfBlock>>(dst, src, calCount, deqScale);
        } else {
            bool signMode = GetCastDeqSignMode(deqScale);
            if (signMode) {
                VF_CALL<CastDeqImplVF<U, T, halfBlock, true>>(dst, src, calCount, deqScale);
            } else {
                VF_CALL<CastDeqImplVF<U, T, halfBlock, false>>(dst, src, calCount, deqScale);
            }
        }
    }
}

template <typename U, typename T, bool isCounterMode, bool isBitMap, bool isSetMask, bool halfBlock>
__aicore__ inline void CastVecDeqLevel0ImplVF(__ubuf__ U *dst, __ubuf__ T *src, const int32_t mask,
    __ubuf__ uint64_t *tempBuf, uint8_t repeatTime, const UnaryRepeatParams &repeatParams, uint64_t deqScaleAddr)
{
    MicroAPI::RegTensor<U> dstReg;
    MicroAPI::RegTensor<T> srcReg0, srcReg1;
    MicroAPI::RegTensor<float> scaleReg;
    MicroAPI::RegTensor<int16_t> offsetReg, signDstReg;
    MicroAPI::RegTensor<uint8_t> mrg2ChnIndexReg;
    MicroAPI::RegTensor<uint16_t> unSignDstReg;
    MicroAPI::RegTensor<int32_t> tmpReg;
    MicroAPI::MaskReg maskReg0, maskReg1, maskReg2, fullMask, signMask, unSignMask, dstMask;
    constexpr int16_t unRollConstant = 2;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t halfRepSize = GetVecLen() / unRollConstant / sizeof(T);
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint32_t dstBlockElm = GetDataBlockSizeInBytes() / sizeof(U);
    constexpr int16_t s9MaxValue = 255;
    constexpr int16_t s9MinValue = -256;
    constexpr int16_t zero = 0;
    constexpr int16_t halfRepBlkSize = 4;
    uint32_t halfRepStride = halfRepBlkSize * blockElm * repeatParams.srcBlkStride;
    uint32_t sreg = static_cast<uint32_t>(mask);
    GenVecCastDeqParam(deqScaleAddr, scaleReg, offsetReg, signMask, unSignMask, tmpReg, fullMask);
    GenGatherIndex((MicroAPI::RegTensor<int8_t>&)mrg2ChnIndexReg);
    if constexpr (isCounterMode) {
        if constexpr (!isSetMask) {
            maskReg0 = MicroAPI::MoveMask<uint16_t>();
            MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg0);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
            sreg = static_cast<uint32_t>(tempBuf[0]);
        }
        repeatTime = CeilDivision(sreg, oneRepSize);
    } else {
        if constexpr (isBitMap) {
            maskReg0 = MicroAPI::MoveMask<T>();
        } else {
            if constexpr (isSetMask) {
                uint32_t sreg = static_cast<uint32_t>(mask);
                maskReg0 = MicroAPI::UpdateMask<T>(sreg);
            } else {
                maskReg0 = MicroAPI::MoveMask<T>();
            }
        }
    }
    fullMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < repeatTime; ++i) {
        if constexpr (isCounterMode) {
            maskReg0 = MicroAPI::UpdateMask<T>(sreg);
        }
        MicroAPI::MaskInterleave<T>(maskReg1, maskReg2, maskReg0, fullMask);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg0, src + i * blockElm * repeatParams.srcRepStride,
            static_cast<uint32_t>(repeatParams.srcBlkStride), maskReg1);
        MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t> &)srcReg0, (MicroAPI::RegTensor<uint16_t> &)srcReg0);
        CastVecDeqMulsCal<T, MicroAPI::HighLowPart::LOWEST>(tmpReg, srcReg0, maskReg1, scaleReg);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg1, src + i * blockElm * repeatParams.srcRepStride + halfRepStride, 
            static_cast<uint32_t>(repeatParams.srcBlkStride), maskReg2);
        MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t> &)srcReg1, (MicroAPI::RegTensor<uint16_t> &)srcReg1);
        CastVecDeqMulsCal<T, MicroAPI::HighLowPart::HIGHEST>(tmpReg, srcReg1, maskReg2, scaleReg);
        maskReg1 = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::H>();
        MicroAPI::Select(srcReg0, srcReg0, srcReg1, maskReg1);
        MicroAPI::Maxs(srcReg0, srcReg0, s9MinValue, maskReg0);
        MicroAPI::Mins(srcReg0, srcReg0, s9MaxValue, maskReg0);
        MicroAPI::Add(srcReg0, srcReg0, offsetReg, maskReg0);
        CastFromS92B8<int8_t, T, halfBlock, true>(srcReg0, (MicroAPI::RegTensor<int8_t> &)signDstReg, signMask);
        CastFromS92B8<uint8_t, T, halfBlock, false>(srcReg0, (MicroAPI::RegTensor<uint8_t> &)unSignDstReg, unSignMask);
        MicroAPI::Select((MicroAPI::RegTensor<int16_t> &)dstReg, signDstReg, (MicroAPI::RegTensor<int16_t> &)unSignDstReg, signMask);
        MicroAPI::Gather(dstReg, dstReg, mrg2ChnIndexReg);
        GenLevel0StoreMask<halfBlock>(maskReg0, dstMask, mrg2ChnIndexReg, (MicroAPI::RegTensor<uint8_t> &)tmpReg, fullMask);
        MicroAPI::StoreAlign<U, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(dst + i * dstBlockElm * repeatParams.dstRepStride,
            dstReg, static_cast<uint32_t>(repeatParams.dstBlkStride), dstMask);
    }
}

template <typename U, typename T, bool isCounterMode, bool isBitMap, bool isSetMask, bool halfBlock, bool signMode>
__aicore__ inline void CastDeqLevel0ImplVF(__ubuf__ U *dst, __ubuf__ T *src, const int32_t mask,
    __ubuf__ uint64_t *tempBuf, uint8_t repeatTime, const UnaryRepeatParams &repeatParams, uint64_t deqScale)
{
    MicroAPI::RegTensor<T> srcReg0, srcReg1;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<U> dstReg;
    MicroAPI::RegTensor<uint8_t> mrg2ChnIndexReg;
    MicroAPI::MaskReg maskReg0, maskReg1, maskReg2, fullMask, dstMask;
    constexpr int16_t unRollConstant = 2;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t halfRepSize = GetVecLen() / unRollConstant / sizeof(T);
    constexpr uint32_t blockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint32_t dstBlockElm = GetDataBlockSizeInBytes() / sizeof(U);
    float scale = GetCastDeqScale(deqScale);
    uint16_t offset = GetCastDeqOffset(deqScale);
    constexpr int16_t s9MaxValue = 255;
    constexpr int16_t s9MinValue = -256;
    constexpr int16_t halfRepBlkSize = 4;
    constexpr int8_t zero = 0;
    uint32_t halfRepStride = halfRepBlkSize * blockElm * repeatParams.srcBlkStride;
    uint32_t sreg = static_cast<uint32_t>(mask);
    if constexpr (isCounterMode) {
        if constexpr (!isSetMask) {
            maskReg0 = MicroAPI::MoveMask<uint16_t>();
            MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg0);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
            sreg = static_cast<uint32_t>(tempBuf[0]);
        }
        repeatTime = CeilDivision(sreg, oneRepSize);
    } else {
        if constexpr (isBitMap) {
            maskReg0 = MicroAPI::MoveMask<T>();
        } else {
            if constexpr (isSetMask) {
                uint32_t sreg = static_cast<uint32_t>(mask);
                maskReg0 = MicroAPI::UpdateMask<T>(sreg);
            } else {
                maskReg0 = MicroAPI::MoveMask<T>();
            }
        }
    }
    fullMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
    GenGatherIndex((MicroAPI::RegTensor<int8_t>&)mrg2ChnIndexReg);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        if constexpr (isCounterMode) {
            maskReg0 = MicroAPI::UpdateMask<T>(sreg);
        }
        MicroAPI::MaskInterleave<T>(maskReg1, maskReg2, maskReg0, fullMask);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg0, src + i * blockElm * repeatParams.srcRepStride,
            static_cast<uint32_t>(repeatParams.srcBlkStride), maskReg1);
        MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t> &)srcReg0, (MicroAPI::RegTensor<uint16_t> &)srcReg0);
        CastDeqMulsCal<T, MicroAPI::HighLowPart::LOWEST>(tmpReg, srcReg0, maskReg1, scale);
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg1, src + i * blockElm * repeatParams.srcRepStride + halfRepStride,
            static_cast<uint32_t>(repeatParams.srcBlkStride), maskReg2);
        MicroAPI::UnPack((MicroAPI::RegTensor<uint32_t> &)srcReg1, (MicroAPI::RegTensor<uint16_t> &)srcReg1);
        CastDeqMulsCal<T, MicroAPI::HighLowPart::HIGHEST>(tmpReg, srcReg1, maskReg2, scale);
        MicroAPI::Or(srcReg0, srcReg0, srcReg1, maskReg0);
        MicroAPI::Maxs(srcReg0, srcReg0, s9MinValue, maskReg0);
        MicroAPI::Mins(srcReg0, srcReg0, s9MaxValue, maskReg0);
        MicroAPI::Adds(srcReg0, srcReg0, offset, maskReg0);
        CastFromS92B8<U, T, halfBlock, signMode>(srcReg0, dstReg, maskReg0);
        MicroAPI::Gather((MicroAPI::RegTensor<uint8_t> &)dstReg, (MicroAPI::RegTensor<uint8_t> &)dstReg, mrg2ChnIndexReg);
        GenLevel0StoreMask<halfBlock>(maskReg0, dstMask, mrg2ChnIndexReg, (MicroAPI::RegTensor<uint8_t> &)tmpReg, fullMask);
        MicroAPI::StoreAlign<U, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(dst + i * dstBlockElm * repeatParams.dstRepStride, dstReg,
            static_cast<uint32_t>(repeatParams.dstBlkStride), dstMask);
    }
}

template <typename U, typename T, bool isCounterMode, bool isBitMap, bool isSetMask>
__aicore__ inline void CastDeqS322f16Level0ImplVF(__ubuf__ U *dst, __ubuf__ T *src, const int32_t mask,
    __ubuf__ uint64_t *tempBuf, uint8_t repeatTime, const UnaryRepeatParams &repeatParams, const half deqScale)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<U> dstReg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg dstMask;
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(T);
    constexpr uint32_t srcBlockElm = GetDataBlockSizeInBytes() / sizeof(T);
    constexpr uint32_t dstBlockElm = GetDataBlockSizeInBytes() / sizeof(U);
    uint32_t sreg = mask;
    if constexpr (isCounterMode) {
        if constexpr (!isSetMask) {
            maskReg = MicroAPI::MoveMask<uint16_t>();
            MicroAPI::StoreAlign<uint64_t, MicroAPI::MaskDist::DIST_PACK>(tempBuf, maskReg);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
            sreg = static_cast<uint32_t>(tempBuf[0]);
        }
        repeatTime = CeilDivision(sreg, oneRepSize);
    } else {
        if constexpr (isBitMap) {
            maskReg = MicroAPI::MoveMask<T>();
        } else {
            if constexpr (isSetMask) {
                uint32_t sreg = static_cast<uint32_t>(mask);
                maskReg = MicroAPI::UpdateMask<T>(sreg);
            } else {
                maskReg = MicroAPI::MoveMask<T>();
            }
        }
        MicroAPI::MaskPack(dstMask, maskReg);
    }
    for (uint16_t i = 0; i < repeatTime; ++i) {
        if constexpr (isCounterMode) {
            maskReg = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::MaskPack(dstMask, maskReg);
        }
        MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcReg, src + i * srcBlockElm * repeatParams.srcRepStride,
            static_cast<uint32_t>(repeatParams.srcBlkStride), maskReg);
        MicroAPI::Cast<float, T, CastParam::s322F32CastTrait>(tmpReg, srcReg, maskReg);
        MicroAPI::Cast<U, float, CastParam::f322F16CastTrait>(dstReg, tmpReg, maskReg);
        MicroAPI::Muls(dstReg, dstReg, deqScale, maskReg);
        MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)dstReg, (MicroAPI::RegTensor<uint32_t>&)dstReg);
        MicroAPI::StoreAlign<U, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(dst + i * dstBlockElm * repeatParams.dstRepStride, dstReg,
            static_cast<uint32_t>(repeatParams.dstBlkStride), dstMask);
    }
}

template <typename U, typename T, bool isSetMask = true, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(
    __ubuf__ U *dst, __ubuf__ T *src, const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<int16_t, int8_t>, Tuple<int16_t, uint8_t>, Tuple<int32_t, half>>(),
        "Failed to check dtype in CastDeqImpl, current api support dtype combination is src: int16_t dst: int8_t/uint8_t"
        ", src:int32_t dst:half.");
    bool isCounterMode = Internal::IsCounterMode();
    __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 4);
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }
    if constexpr (IsSameType<T, int32_t>::value) {
        half scale = g_deqValue;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        if (isCounterMode) {
            VF_CALL<CastDeqS322f16Level0ImplVF<U, T, true, true, isSetMask>>(
                dst, src, mask[0], tempBuf, repeatTime, repeatParams, scale);
        } else {
            VF_CALL<CastDeqS322f16Level0ImplVF<U, T, false, true, isSetMask>>(
                dst, src, mask[0], tempBuf, repeatTime, repeatParams, scale);
        }
    } else {
        uint64_t deqScale = Internal::g_deqScale;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        bool signMode = GetCastDeqSignMode(deqScale);
        if (isCounterMode) {
            if constexpr (isVecDeq) {
                VF_CALL<CastVecDeqLevel0ImplVF<U, T, true, true, isSetMask, halfBlock>>(
                    dst, src, mask[0], tempBuf, repeatTime, repeatParams, deqScale);
            } else {
                if (signMode) {
                    VF_CALL<CastDeqLevel0ImplVF<U, T, true, true, isSetMask, halfBlock, true>>(
                        dst, src, mask[0], tempBuf, repeatTime, repeatParams, deqScale);
                } else {
                    VF_CALL<CastDeqLevel0ImplVF<U, T, true, true, isSetMask, halfBlock, false>>(
                        dst, src, mask[0], tempBuf, repeatTime, repeatParams, deqScale);
                }
            }
        } else {
            if constexpr (isVecDeq) {
                VF_CALL<CastVecDeqLevel0ImplVF<U, T, false, true, isSetMask, halfBlock>>(
                    dst, src, 0, tempBuf, repeatTime, repeatParams, deqScale);
            } else {
                if (signMode) {
                    VF_CALL<CastDeqLevel0ImplVF<U, T, false, true, isSetMask, halfBlock, true>>(
                        dst, src, 0, tempBuf, repeatTime, repeatParams, deqScale);
                } else {
                    VF_CALL<CastDeqLevel0ImplVF<U, T, false, true, isSetMask, halfBlock, false>>(
                        dst, src, 0, tempBuf, repeatTime, repeatParams, deqScale);
                }
            }
        }
    }
    AscendCUtils::FreeTemporaryBuffer(tempBuf);
}

template <typename U, typename T, bool isSetMask = true, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(
    __ubuf__ U *dst, __ubuf__ T *src, const int32_t mask, uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<int16_t, int8_t>, Tuple<int16_t, uint8_t>, Tuple<int32_t, half>>(),
        "Failed to check dtype in CastDeqImpl, current api support dtype combination is src: int16_t dst: int8_t/uint8_t"
        ", src:int32_t dst:half.");
    bool isCounterMode = Internal::IsCounterMode();
    __ubuf__ uint64_t *tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(GetRuntimeUBSize(), 4);
    if constexpr (IsSameType<T, int32_t>::value) {
        half scale = g_deqValue;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        if (isCounterMode) {
            VF_CALL<CastDeqS322f16Level0ImplVF<U, T, true, false, isSetMask>>(
                dst, src, mask, tempBuf, repeatTime, repeatParams, scale);
        } else {
            VF_CALL<CastDeqS322f16Level0ImplVF<U, T, false, false, isSetMask>>(
                dst, src, mask, tempBuf, repeatTime, repeatParams, scale);
        }
    } else {
        uint64_t deqScale = Internal::g_deqScale;
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        bool signMode = GetCastDeqSignMode(deqScale);
        if (isCounterMode) {
            if constexpr (isVecDeq) {
                VF_CALL<CastVecDeqLevel0ImplVF<U, T, true, false, isSetMask, halfBlock>>(
                    dst, src, mask, tempBuf, repeatTime, repeatParams, deqScale);
            } else {
                if (signMode) {
                    VF_CALL<CastDeqLevel0ImplVF<U, T, true, false, isSetMask, halfBlock, true>>(
                        dst, src, mask, tempBuf, repeatTime, repeatParams, deqScale);
                } else {
                    VF_CALL<CastDeqLevel0ImplVF<U, T, true, false, isSetMask, halfBlock, false>>(
                        dst, src, mask, tempBuf, repeatTime, repeatParams, deqScale);
                }
            }
        } else {
            if constexpr (isVecDeq) {
                VF_CALL<CastVecDeqLevel0ImplVF<U, T, false, false, isSetMask, halfBlock>>(
                    dst, src, mask, tempBuf, repeatTime, repeatParams, deqScale);
            } else {
                if (signMode) {
                    VF_CALL<CastDeqLevel0ImplVF<U, T, false, false, isSetMask, halfBlock, true>>(
                        dst, src, mask, tempBuf, repeatTime, repeatParams, deqScale);
                } else {
                    VF_CALL<CastDeqLevel0ImplVF<U, T, false, false, isSetMask, halfBlock, false>>(
                        dst, src, mask, tempBuf, repeatTime, repeatParams, deqScale);
                }
            }
        }
    }
    AscendCUtils::FreeTemporaryBuffer(tempBuf);
}

namespace MicroAPIAddReluCast {
template <typename T1, typename T2, typename RegT, typename RegU>
__aicore__ inline void AddReluCast(RegT &dstReg, RegU &src0Reg, RegU &src1Reg, MicroAPI::MaskReg &mask)
{
    MicroAPI::Add(src0Reg, src0Reg, src1Reg, mask);
    MicroAPI::Maxs(src0Reg, src0Reg, static_cast<T2>(0), mask);
    if constexpr (IsSameType<T2, float>::value) {
        MicroAPI::Cast<T1, T2, CastParam::AddReluCastTrait>(dstReg, src0Reg, mask);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint16_t> &)dstReg, (MicroAPI::RegTensor<uint32_t> &)dstReg);
    } else {
        if constexpr (IsSameType<T2, int16_t>::value) {
            MicroAPI::RegTensor<half> tmpReg;
            MicroAPI::Cast<half, int16_t, CastParam::s162HalfTrait>(tmpReg, src0Reg, mask);
            MicroAPI::Cast<int8_t, half, CastParam::AddReluCastTrait>(dstReg, tmpReg, mask);
        } else {
            MicroAPI::Cast<T1, T2, CastParam::AddReluCastTrait>(dstReg, src0Reg, mask);
        }
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint8_t> &)dstReg, (MicroAPI::RegTensor<uint16_t> &)dstReg);
    }
}
}  // namespace MicroAPIAddReluCast

template <typename T1, typename T2> constexpr __aicore__ inline void CheckAddReluCastSupportType()
{
    static_assert(SupportType<Tuple<T1, T2>, Tuple<half, float>, Tuple<int8_t, half>, Tuple<int8_t, int16_t>>(),
        "Failed to check dtype in AddReluCast, current api support dtype combination is src: float, dst: half; "
        "src: half, dst: int8_t; src: int16_t, dst: int8_t.");
}
// AddReluCast::Level 0 - mask count mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ T1 *dst, __ubuf__ T2 *src0, __ubuf__ T2 *src1, const uint64_t mask,
    uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    CheckAddReluCastSupportType<T1, T2>();
    constexpr auto func = MicroAPIAddReluCast::AddReluCast<T1, T2, MicroAPI::RegTensor<T1>, MicroAPI::RegTensor<T2>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

// AddReluCast::Level 0 - mask bit mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ T1 *dst, __ubuf__ T2 *src0, __ubuf__ T2 *src1, const uint64_t mask[],
    uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    CheckAddReluCastSupportType<T1, T2>();
    constexpr auto func = MicroAPIAddReluCast::AddReluCast<T1, T2, MicroAPI::RegTensor<T1>, MicroAPI::RegTensor<T2>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

// AddReluCast::Level 2
template <typename T1, typename T2>
__aicore__ inline void AddReluCastImpl(__ubuf__ T1 *dst, __ubuf__ T2 *src0, __ubuf__ T2 *src1, const uint32_t calCount)
{
    static_assert(SupportType<Tuple<T1, T2>, Tuple<half, float>, Tuple<int8_t, half>, Tuple<int8_t, int16_t>,
            Tuple<float, int64_t>, Tuple<int32_t, int64_t>>(), "Failed to check dtype in AddReluCast, current api "
            "support dtype combination is src: float, dst: half; src: half, dst: int8_t; src: int16_t, dst: int8_t; "
            "src: int64_t, dst : int32_t / float.");
    uint32_t sreg = static_cast<uint32_t>(calCount);
    const T2 scalarValue = 0;
    if constexpr (sizeof(T2) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T1> vDstReg0;
            MicroAPI::RegTensor<T2, MicroAPI::RegTraitNumTwo> vDstReg1;
            MicroAPI::RegTensor<T2, MicroAPI::RegTraitNumTwo> vSrcReg0;
            MicroAPI::RegTensor<T2, MicroAPI::RegTraitNumTwo> vSrcReg1;
            MicroAPI::MaskReg mask;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                mask = MicroAPI::UpdateMask<T2, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(vSrcReg0, src0 + i * sregLower);
                MicroAPI::LoadAlign(vSrcReg1, src1 + i * sregLower);
                MicroAPI::Add(vDstReg1, vSrcReg0, vSrcReg1, mask);
                MicroAPI::Maxs(vDstReg1, vDstReg1, scalarValue, mask);
                MicroAPI::Cast<T1, T2, CastParam::AddReluCastTrait>(vDstReg0, vDstReg1, mask);
                MicroAPI::StoreAlign(dst + i * sregLower, vDstReg0, mask);
            }
        }
    } else {
        constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T2));
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T1> dst0Reg;
            MicroAPI::RegTensor<T2> dst1Reg;
            MicroAPI::RegTensor<T2> src0Reg;
            MicroAPI::RegTensor<T2> src1Reg;
            MicroAPI::MaskReg preg;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                preg = MicroAPI::UpdateMask<T2>(sreg);
                MicroAPI::LoadAlign<T2>(src0Reg, src0 + i * sregLower);
                MicroAPI::LoadAlign<T2>(src1Reg, src1 + i * sregLower);
                MicroAPI::Add<T2>(dst1Reg, src0Reg, src1Reg, preg);
                MicroAPI::Maxs<T2>(dst1Reg, dst1Reg, scalarValue, preg);
                if constexpr (IsSameType<T2, float>::value) {
                    MicroAPI::Cast<T1, T2, CastParam::AddReluCastTrait>(dst0Reg, dst1Reg, preg);
                    MicroAPI::StoreAlign<T1, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, dst0Reg, preg);
                } else {
                    if constexpr (IsSameType<T2, int16_t>::value) {
                        MicroAPI::RegTensor<half> tmpReg;
                        MicroAPI::Cast<half, int16_t, CastParam::s162HalfTrait>(tmpReg, dst1Reg, preg);
                        MicroAPI::Cast<int8_t, half, CastParam::AddReluCastTrait>(dst0Reg, tmpReg, preg);
                    } else {
                        MicroAPI::Cast<T1, T2, CastParam::AddReluCastTrait>(dst0Reg, dst1Reg, preg);
                    }
                    MicroAPI::StoreAlign<T1, MicroAPI::StoreDist::DIST_PACK_B16>(dst + i * sregLower, dst0Reg, preg);
                }
            }
        }
    }
}

namespace MicroAPISubReluCast {
template <typename T1, typename T2, typename RegT, typename RegU>
__aicore__ inline void SubReluCast(RegT &dstReg, RegU &src0Reg, RegU &src1Reg, MicroAPI::MaskReg &mask)
{
    MicroAPI::Sub(src0Reg, src0Reg, src1Reg, mask);
    MicroAPI::Maxs<T2>(src0Reg, src0Reg, static_cast<T2>(0), mask);
    if constexpr (IsSameType<T2, float>::value) {
        MicroAPI::Cast<T1, T2, layoutZSatSMrgZRndR>(dstReg, src0Reg, mask);
        MicroAPI::Pack((MicroAPI::RegTensor<uint16_t> &)dstReg, (MicroAPI::RegTensor<uint32_t> &)dstReg);
    } else {
        if constexpr (IsSameType<T2, int16_t>::value) {
            MicroAPI::RegTensor<half> tmpReg;
            MicroAPI::Cast<half, int16_t, MrgZRndRSatS>(tmpReg, src0Reg, mask);
            MicroAPI::Cast<int8_t, half, layoutZSatSMrgZRndR>(dstReg, tmpReg, mask);
        } else {
            MicroAPI::Cast<T1, T2, layoutZSatSMrgZRndR>(dstReg, src0Reg, mask);
        }
        MicroAPI::Pack((MicroAPI::RegTensor<uint8_t> &)dstReg, (MicroAPI::RegTensor<uint16_t> &)dstReg);
    }
}
}  // namespace MicroAPISubReluCast

template <typename T1, typename T2>
constexpr __aicore__ inline void CheckSubReluCastSupportType()
{
    static_assert(SupportType<Tuple<T1, T2>, Tuple<half, float>, Tuple<int8_t, half>, Tuple<int8_t, int16_t>>(),
        "Failed to check dtype in SubReluCast, current api support dtype combination is src: float, dst: half; "
        "src: half, dst: int8_t; src: int16_t, dst: int8_t.");
}

// SubReluCast::Level 0 - mask count mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ T1 *dst, __ubuf__ T2 *src0, __ubuf__ T2 *src1, const uint64_t mask,
    uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    CheckSubReluCastSupportType<T1, T2>();
    constexpr auto func = MicroAPISubReluCast::SubReluCast<T1, T2, MicroAPI::RegTensor<T1>, MicroAPI::RegTensor<T2>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, false>(dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
}

// SubReluCast::Level 0 - mask bit mode
template <typename T1, typename T2, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ T1 *dst, __ubuf__ T2 *src0, __ubuf__ T2 *src1, const uint64_t mask[],
    uint8_t repeatTime, const BinaryRepeatParams &repeatParams)
{
    CheckSubReluCastSupportType<T1, T2>();
    constexpr auto func = MicroAPISubReluCast::SubReluCast<T1, T2, MicroAPI::RegTensor<T1>, MicroAPI::RegTensor<T2>>;
    Internal::VecBinaryImplTemplate<func, isSetMask, true>(dst, src0, src1, mask, 0, repeatTime, repeatParams);
}

// SubReluCast::Level 2
template <typename T1, typename T2>
__aicore__ inline void SubReluCastImpl(__ubuf__ T1* dst, __ubuf__ T2* src0, __ubuf__ T2* src1, const uint32_t calCount)
{
    static_assert(SupportType<Tuple<T1, T2>, Tuple<half, float>, Tuple<int8_t, half>, Tuple<int8_t, int16_t>,
        Tuple<float, int64_t>, Tuple<int32_t, int64_t>>(), "Failed to check dtype in SubReluCast, current api support "
        "dtype combination is src: float, dst: half; src: half, dst: int8_t; src: int16_t, dst: int8_t; src: int64_t, "
        "dst : int32_t / float.");
    uint32_t sreg = static_cast<uint32_t>(calCount);
    const T2 scalarValue = 0;

    if constexpr (sizeof(T2) == 8) {
        constexpr uint32_t sregLower = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT * 2);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T1> vDstReg0;
            MicroAPI::RegTensor<T2, MicroAPI::RegTraitNumTwo> vDstReg1;
            MicroAPI::RegTensor<T2, MicroAPI::RegTraitNumTwo> vSrcReg0;
            MicroAPI::RegTensor<T2, MicroAPI::RegTraitNumTwo> vSrcReg1;
            MicroAPI::MaskReg mask;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                mask = MicroAPI::UpdateMask<T2, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(vSrcReg0, src0 + i * sregLower);
                MicroAPI::LoadAlign(vSrcReg1, src1 + i * sregLower);
                MicroAPI::Sub(vDstReg1, vSrcReg0, vSrcReg1, mask);
                MicroAPI::Maxs(vDstReg1, vDstReg1, scalarValue, mask);
                MicroAPI::Cast<T1, T2, CastParam::SubReluCastTrait>(vDstReg0, vDstReg1, mask);
                MicroAPI::StoreAlign(dst + i * sregLower, vDstReg0, mask);
            }
        }
    } else {
        const uint32_t repeatStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T2));
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T2> src0Reg;
            MicroAPI::RegTensor<T2> src1Reg;
            MicroAPI::RegTensor<T1> dstReg;
            MicroAPI::MaskReg mask;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                mask = MicroAPI::UpdateMask<T2>(sreg);
                MicroAPI::LoadAlign(src0Reg, src0 + i * repeatStride);
                MicroAPI::LoadAlign(src1Reg, src1 + i * repeatStride);
                MicroAPI::Sub(src0Reg, src0Reg, src1Reg, mask);
                MicroAPI::Maxs<T2>(src0Reg, src0Reg, scalarValue, mask);
                if constexpr (IsSameType<T2, float>::value) {
                    MicroAPI::Cast<T1, T2, layoutZSatSMrgZRndR>(dstReg, src0Reg, mask);
                    MicroAPI::StoreAlign<T1, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * repeatStride, dstReg, mask);
                } else {
                    if constexpr (IsSameType<T2, int16_t>::value) {
                        MicroAPI::RegTensor<half> tmpReg;
                        MicroAPI::Cast<half, int16_t, MrgZRndRSatS>(tmpReg, src0Reg, mask);
                        MicroAPI::Cast<int8_t, half, layoutZSatSMrgZRndR>(dstReg, tmpReg, mask);
                    } else {
                        MicroAPI::Cast<T1, T2, layoutZSatSMrgZRndR>(dstReg, src0Reg, mask);
                    }
                    MicroAPI::StoreAlign<T1, MicroAPI::StoreDist::DIST_PACK_B16>(dst + i * repeatStride, dstReg, mask);
                }
            }
        }
    }
}

//  castDequantValue bit arrange
//  =========================================================================
//  | unused 17bit | 1bit signMode | 9bit offset | unused 5bit | 32bit scale|
//  =========================================================================
__aicore__ inline uint64_t MakeDeqScaleConfig(float scale, int16_t offset, bool signMode)
{
    constexpr uint64_t signModeBit = 46;
    constexpr uint64_t offsetMask = 0x1ff;
    constexpr uint64_t offsetBit = 37;
    uint64_t config = ((static_cast<uint64_t>(signMode) << signModeBit) | ((offset & offsetMask) << offsetBit) |
                       *(reinterpret_cast<uint32_t *>(&scale)));
    return config;
}

__aicore__ inline void SetDeqScaleImpl(float scale, int16_t offset, bool signMode)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    Internal::g_deqScale = MakeDeqScaleConfig(scale, offset, signMode);
}

template <typename T> __aicore__ inline void SetDeqScaleImpl(const LocalTensor<T> &vdeqTensor, const VdeqInfo &vdeqInfo)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (uint8_t i = 0; i < VDEQ_TENSOR_SIZE; ++i) {
        float scale = vdeqInfo.vdeqScale[i];
        int16_t offset = vdeqInfo.vdeqOffset[i];
        bool signMode = vdeqInfo.vdeqSignMode[i];
        vdeqTensor.SetValue(i, static_cast<T>(MakeDeqScaleConfig(scale, offset, signMode)));
    }
    Internal::g_deqScale = reinterpret_cast<uint64_t>(vdeqTensor.GetPhyAddr());
}

template <typename T> __aicore__ inline void SetDeqScaleImpl(T config)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    g_deqValue = config;
}
// Truncate::Level2
template <typename T, RoundMode roundMode>
__aicore__ inline void TruncateImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(), "Failed to check dtype in Truncate, current api "
        "support dtype is src and dst both: half, float, bfloat16_t.");
    static_assert(SupportEnum<roundMode, RoundMode::CAST_RINT, RoundMode::CAST_FLOOR, RoundMode::CAST_CEIL,
        RoundMode::CAST_ROUND, RoundMode::CAST_TRUNC>(), "Failed to check dtype in Truncate, "
        "current api support roundMode is CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vDstReg;
        MicroAPI::RegTensor<T> vSrcReg;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::LoadAlign(vSrcReg, src + i * sregLower);
            MicroAPI::Truncate<T, roundMode>(vDstReg, vSrcReg, mask);
            MicroAPI::StoreAlign(dst + i * sregLower, vDstReg, mask);
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
