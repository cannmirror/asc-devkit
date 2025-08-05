/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_out_with_layout.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_TILE_COPY_OUI_COPY_OUT_WITH_LAYOUT_H
#define ACT_INCLUDE_MATMUL_TILE_COPY_OUI_COPY_OUT_WITH_LAYOUT_H

#include "../tile_copy_policy.h"

namespace Act {
namespace Gemm {
namespace Tile {
template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910B, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<
        SrcTrait::tPos == AscendC::TPosition::CO1 && DstTrait::tPos == AscendC::TPosition::GM       // CO1->GM
        && (OutputType::format == CubeFormat::ND || OutputType::format == CubeFormat::ND_ALIGN)>> { // ND/ND_ALIGN
    using DstTensor = AscendC::GlobalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord)
    {
#if __CCE_AICORE__ == 220
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsV220 params;
        params.nSize =
            AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape), Get<1>(dstShape) - Get<1>(coord));
        params.mSize =
            AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape), Get<0>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<0>(dstStride);
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }

        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }
};

template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<Arch::Ascend910B, CopyWithLayout, OutputType, DstTrait, SrcTrait,
            AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::CO1 &&
                                      DstTrait::tPos == AscendC::TPosition::GM    // CO1->GM
                                      && OutputType::format == CubeFormat::NZ>> { // NZ
    using DstTensor = AscendC::GlobalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord)
    {
#if __CCE_AICORE__ == 220
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsV220 params;
        params.nSize = AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape),
                                                   Get<1, 0>(dstShape) * Get<1, 1>(dstShape) - Get<1>(coord));
        params.mSize = AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape),
                                                   Get<0, 0>(dstShape) * Get<0, 1>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<1, 1>(dstStride) * sizeof(DstT) / 32; // 32: c0 byte size
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }
};

template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<
        SrcTrait::tPos == AscendC::TPosition::CO1 && DstTrait::tPos == AscendC::TPosition::GM       // CO1->GM
        && (OutputType::format == CubeFormat::ND || OutputType::format == CubeFormat::ND_ALIGN)>> { // ND/ND_ALIGN
    using DstTensor = AscendC::GlobalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord)
    {
#if defined(__DAV_C310__)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize =
            AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape), Get<1>(dstShape) - Get<1>(coord));
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.nSize = AscendC::CeilAlign(params.nSize, AscendC::ONE_BLK_SIZE / sizeof(DstT));
        }
        params.mSize =
            AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape), Get<0>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<0>(dstStride);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.dstStride = AscendC::CeilAlign(params.dstStride, AscendC::ONE_BLK_SIZE / sizeof(DstT));
        }
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }

        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<Arch::Ascend910_95, CopyWithLayout, OutputType, DstTrait, SrcTrait,
            AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::CO1 &&
                                      DstTrait::tPos == AscendC::TPosition::GM    // CO1->GM
                                      && OutputType::format == CubeFormat::NZ>> { // NZ
    using DstTensor = AscendC::GlobalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord)
    {
#if defined(__DAV_C310__)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        params.nSize = AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape),
                                                   Get<1, 0>(dstShape) * Get<1, 1>(dstShape) - Get<1>(coord));
        params.mSize = AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape),
                                                   Get<0, 0>(dstShape) * Get<0, 1>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<1, 1>(dstStride) * AscendC::BLOCK_CUBE / (AscendC::ONE_BLK_SIZE / sizeof(DstT));
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::GlobalTensor<DstT> dstTensor;
        dstTensor.SetGlobalBuffer(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<AscendC::PhyPosIsL0C(SrcTrait::tPos) && AscendC::PhyPosIsUB(DstTrait::tPos) // CO1->UB
                              && OutputType::format == CubeFormat::ND>> {                                 // ND
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord, uint8_t subIdx = 0)
    {
#if defined(__DAV_C310__)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;
        static constexpr AscendC::FixpipeConfig cfgUb = {AscendC::CO2Layout::ROW_MAJOR, true}; // isToUB is true

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310<cfgUb.format> params;
        params.nSize =
            AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape), Get<1>(dstShape) - Get<1>(coord));
        params.mSize =
            AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape), Get<0>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<0>(dstStride);
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.dualDstCtl = 0;
        params.subBlockId = subIdx;

        AscendC::LocalTensor<DstT> dstTensor;
        dstTensor.SetAddr(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, cfgUb>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

template <class OutputType, class DstTrait, class SrcTrait>
struct Copy<
    Arch::Ascend910_95, CopyWithLayout, OutputType, DstTrait, SrcTrait,
    AscendC::Std::enable_if_t<AscendC::PhyPosIsL0C(SrcTrait::tPos) && AscendC::PhyPosIsUB(DstTrait::tPos) // CO1->UB
                              && OutputType::format == CubeFormat::NZ>> {                                 // NZ
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::LocalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord, uint8_t subIdx = 0)
    {
#if defined(__DAV_C310__)
        using SrcT = typename SrcTrait::LiteType;
        using DstT = typename DstTrait::LiteType;
        static constexpr AscendC::FixpipeConfig cfgUb = {AscendC::CO2Layout::NZ, true}; // isToUB is true

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        auto dstStride = dst.GetTensorTrait().GetLayout().GetStride();

        AscendC::FixpipeParamsC310<cfgUb.format> params;
        params.nSize = AscendC::Std::min<int, int>(Get<1, 0>(srcShape) * Get<1, 1>(srcShape),
                                                   Get<1, 0>(dstShape) * Get<1, 1>(dstShape) - Get<1>(coord));
        params.mSize = AscendC::Std::min<int, int>(Get<0, 0>(srcShape) * Get<0, 1>(srcShape),
                                                   Get<0, 0>(dstShape) * Get<0, 1>(dstShape) - Get<0>(coord));
        params.srcStride = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        params.dstStride = Get<1, 1>(dstStride) * AscendC::BLOCK_CUBE / (AscendC::ONE_BLK_SIZE / sizeof(DstT));
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.dualDstCtl = 0;
        params.subBlockId = subIdx;

        AscendC::LocalTensor<DstT> dstTensor;
        dstTensor.SetAddr(dst.address_);
        AscendC::LocalTensor<SrcT> srcTensor;
        srcTensor.SetAddr(src.address_);

        auto offset = dst.GetTensorTrait().GetLayout()(coord);
        AscendC::Fixpipe<DstT, SrcT, cfgUb>(dstTensor[offset], srcTensor, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
