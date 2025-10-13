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
 * \file broadcast_gather_c310_impl.h
 * \brief
 */
#ifndef IMPL_PAD_BROADCAST_BROADCAST_GATHER_C310_IMPL_H
#define IMPL_PAD_BROADCAST_BROADCAST_GATHER_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"

namespace AscendC {
template <typename T>
__aicore__ inline void VfGenIndex(__ubuf__ T *indexUb, uint32_t *sizeI, uint32_t *strideI, uint32_t offset)
{
    MicroAPI::RegTensor<T> v0;
    MicroAPI::RegTensor<T> v1;
    MicroAPI::RegTensor<T> v2;

    MicroAPI::RegTensor<T> voffset;
    MicroAPI::RegTensor<T> vr0;

    MicroAPI::RegTensor<T> vd0;
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::RegTensor<T> vd2;

    MicroAPI::RegTensor<T> vi0;
    MicroAPI::RegTensor<T> vi1;
    MicroAPI::RegTensor<T> vi2;

    MicroAPI::RegTensor<T> vs0;
    MicroAPI::RegTensor<T> vs1;
    MicroAPI::RegTensor<T> vs2;

    MicroAPI::MaskReg p0;

    p0 = MicroAPI::CreateMask<T>();
    MicroAPI::Arange(v0, 0);

    MicroAPI::Duplicate(v1, (T)sizeI[2], p0);
    MicroAPI::Div(vd0, v0, v1, p0);
    MicroAPI::Mul(v2, vd0, v1, p0);
    MicroAPI::Sub(vi2, v0, v2, p0);

    MicroAPI::Duplicate(v1, (T)sizeI[1], p0);
    MicroAPI::Div(vd1, vd0, v1, p0);
    MicroAPI::Mul(v2, vd1, v1, p0);
    MicroAPI::Sub(vi1, vd0, v2, p0);

    MicroAPI::Duplicate(v1, (T)sizeI[0], p0);
    MicroAPI::Div(vd2, vd1, v1, p0);
    MicroAPI::Mul(v2, vd2, v1, p0);
    MicroAPI::Sub(vi0, vd1, v2, p0);

    MicroAPI::Duplicate(vs0, (T)strideI[0], p0);
    MicroAPI::Duplicate(vs1, (T)strideI[1], p0);
    MicroAPI::Duplicate(vs2, (T)strideI[2], p0);

    MicroAPI::Mul(vr0, vs2, vi2, p0);
    MicroAPI::MulAddDst(vr0, vs1, vi1, p0);
    MicroAPI::MulAddDst(vr0, vs0, vi0, p0);

    MicroAPI::Duplicate(voffset, (T)offset, p0);
    MicroAPI::Add(vr0, vr0, voffset, p0);

    MicroAPI::DataCopy(indexUb, vr0, p0);
}

template <typename T>
__aicore__ inline void VfGenIndexForFourDim(__ubuf__ T *indexUb, uint32_t *sizeI, uint32_t *strideI, uint32_t offset)
{
    MicroAPI::RegTensor<T> v0;
    MicroAPI::RegTensor<T> v1;
    MicroAPI::RegTensor<T> v2;

    MicroAPI::RegTensor<T> voffset;
    MicroAPI::RegTensor<T> vr0;

    MicroAPI::RegTensor<T> vd0;
    MicroAPI::RegTensor<T> vd1;
    MicroAPI::RegTensor<T> vd2;
    MicroAPI::RegTensor<T> vd3;

    MicroAPI::RegTensor<T> vi0;
    MicroAPI::RegTensor<T> vi1;
    MicroAPI::RegTensor<T> vi2;
    MicroAPI::RegTensor<T> vi3;

    MicroAPI::RegTensor<T> vs0;
    MicroAPI::RegTensor<T> vs1;
    MicroAPI::RegTensor<T> vs2;
    MicroAPI::RegTensor<T> vs3;

    MicroAPI::MaskReg p0;

    p0 = MicroAPI::CreateMask<T>();
    MicroAPI::Arange(v0, 0);

    MicroAPI::Duplicate(v1, (T)sizeI[3], p0);
    MicroAPI::Div(vd0, v0, v1, p0);
    MicroAPI::Mul(v2, vd0, v1, p0);
    MicroAPI::Sub(vi3, v0, v2, p0);

    MicroAPI::Duplicate(v1, (T)sizeI[2], p0);
    MicroAPI::Div(vd1, vd0, v1, p0);
    MicroAPI::Mul(v2, vd1, v1, p0);
    MicroAPI::Sub(vi2, vd0, v2, p0);

    MicroAPI::Duplicate(v1, (T)sizeI[1], p0);
    MicroAPI::Div(vd2, vd1, v1, p0);
    MicroAPI::Mul(v2, vd2, v1, p0);
    MicroAPI::Sub(vi1, vd1, v2, p0);

    MicroAPI::Duplicate(v1, (T)sizeI[0], p0);
    MicroAPI::Div(vd3, vd2, v1, p0);
    MicroAPI::Mul(v2, vd3, v1, p0);
    MicroAPI::Sub(vi0, vd2, v2, p0);

    MicroAPI::Duplicate(vs0, (T)strideI[0], p0);
    MicroAPI::Duplicate(vs1, (T)strideI[1], p0);
    MicroAPI::Duplicate(vs2, (T)strideI[2], p0);
    MicroAPI::Duplicate(vs3, (T)strideI[3], p0);

    MicroAPI::Mul(vr0, vs3, vi3, p0);
    MicroAPI::MulAddDst(vr0, vs2, vi2, p0);
    MicroAPI::MulAddDst(vr0, vs1, vi1, p0);
    MicroAPI::MulAddDst(vr0, vs0, vi0, p0);

    MicroAPI::Duplicate(voffset, (T)offset, p0);
    MicroAPI::Add(vr0, vr0, voffset, p0);

    MicroAPI::DataCopy(indexUb, vr0, p0);
}

template <typename T>
__aicore__ inline void VfGatherBrc(__ubuf__ T *dstUb, __ubuf__ T *srcUb, __ubuf__ T *indexUb, uint16_t *size,
    uint16_t *srcStride, uint32_t main, uint32_t tail)
{
    MicroAPI::UnalignReg u0;
    MicroAPI::RegTensor<T> vindex0;
    MicroAPI::RegTensor<T> vindex;
    MicroAPI::RegTensor<T> vstride0;
    MicroAPI::RegTensor<T> vstride1;
    MicroAPI::RegTensor<T> vstride2;
    MicroAPI::RegTensor<T> vbase0;
    MicroAPI::RegTensor<T> vbase1;
    MicroAPI::RegTensor<T> vbase2;
    MicroAPI::RegTensor<T> voffset0;
    MicroAPI::RegTensor<T> voffset1;
    MicroAPI::RegTensor<T> voffset2;

    MicroAPI::RegTensor<T> vd0;
    MicroAPI::RegTensor<T> vd1;

    MicroAPI::MaskReg pa;
    MicroAPI::MaskReg p0;
    MicroAPI::MaskReg p1;
    pa = MicroAPI::CreateMask<T>();
    uint32_t main1 = main;
    uint32_t tail1 = tail;
    p0 = MicroAPI::UpdateMask<T>(main);
    p1 = MicroAPI::UpdateMask<T>(tail);
    MicroAPI::Duplicate(vstride0, (T)srcStride[0], pa);
    MicroAPI::Duplicate(vstride1, (T)srcStride[1], pa);
    MicroAPI::Duplicate(vstride2, (T)srcStride[2], pa);
    MicroAPI::DataCopy(vindex0, indexUb);
    for (uint16_t i0 = 0; i0 < size[0]; ++i0) {
        MicroAPI::Muls(voffset0, vstride0, (T)i0, p0);
        MicroAPI::Add(vbase0, voffset0, vindex0, p0);
        for (uint16_t i1 = 0; i1 < size[1]; ++i1) {
            MicroAPI::Muls(voffset1, vstride1, (T)i1, p0);
            MicroAPI::Add(vbase1, vbase0, voffset1, p0);
            for (uint16_t i2 = 0; i2 < size[2]; ++i2) {
                MicroAPI::Muls(voffset2, vstride2, (T)i2, p0);
                MicroAPI::Add(vindex, vbase1, voffset2, p0);
                MicroAPI::DataCopyGather(vd0, srcUb, vindex, p0);
                MicroAPI::DataCopyUnAlign(dstUb, vd0, u0, main1);
            }
            MicroAPI::Muls(voffset2, vstride2, (T)size[2], p1);
            MicroAPI::Add(vindex, vbase1, voffset2, p1);
            MicroAPI::DataCopyGather(vd1, srcUb, vindex, p1);
            MicroAPI::DataCopyUnAlign(dstUb, vd1, u0, tail1);
        }
    }
    MicroAPI::DataCopyUnAlignPost(dstUb, u0, 0);
}

template <typename T>
__aicore__ inline void VfGatherBrcForFourDim(__ubuf__ T *dstUb, __ubuf__ T *srcUb, __ubuf__ T *indexUb, uint16_t *size,
    uint16_t *srcStride, uint32_t main, uint32_t tail)
{
    MicroAPI::UnalignReg u0;
    MicroAPI::RegTensor<T> vindex0;
    MicroAPI::RegTensor<T> vindex;
    MicroAPI::RegTensor<T> vstride0;
    MicroAPI::RegTensor<T> vstride1;
    MicroAPI::RegTensor<T> vstride2;
    MicroAPI::RegTensor<T> vstride3;
    MicroAPI::RegTensor<T> vbase0;
    MicroAPI::RegTensor<T> vbase1;
    MicroAPI::RegTensor<T> vbase2;
    MicroAPI::RegTensor<T> voffset0;
    MicroAPI::RegTensor<T> voffset1;
    MicroAPI::RegTensor<T> voffset2;
    MicroAPI::RegTensor<T> voffset3;

    MicroAPI::RegTensor<T> vd0;
    MicroAPI::RegTensor<T> vd1;

    MicroAPI::MaskReg pa;
    MicroAPI::MaskReg p0;
    MicroAPI::MaskReg p1;
    pa = MicroAPI::CreateMask<T>();
    uint32_t main1 = main;
    uint32_t tail1 = tail;
    p0 = MicroAPI::UpdateMask<T>(main);
    p1 = MicroAPI::UpdateMask<T>(tail);
    MicroAPI::Duplicate(vstride0, (T)srcStride[0], pa);
    MicroAPI::Duplicate(vstride1, (T)srcStride[1], pa);
    MicroAPI::Duplicate(vstride2, (T)srcStride[2], pa);
    MicroAPI::Duplicate(vstride3, (T)srcStride[3], pa);
    MicroAPI::DataCopy(vindex0, indexUb);
    for (uint16_t i0 = 0; i0 < size[0]; ++i0) {
        MicroAPI::Muls(voffset0, vstride0, (T)i0, p0);
        MicroAPI::Add(vbase0, voffset0, vindex0, p0);
        for (uint16_t i1 = 0; i1 < size[1]; ++i1) {
            MicroAPI::Muls(voffset1, vstride1, (T)i1, p0);
            MicroAPI::Add(vbase1, vbase0, voffset1, p0);
            for (uint16_t i2 = 0; i2 < size[2]; ++i2) {
                MicroAPI::Muls(voffset2, vstride2, (T)i2, p0);
                MicroAPI::Add(vbase2, vbase1, voffset2, p0);
                for (uint16_t i3 = 0; i3 < size[3]; ++i3) {
                    MicroAPI::Muls(voffset3, vstride3, (T)i3, p0);
                    MicroAPI::Add(vindex, vbase2, voffset3, p0);
                    MicroAPI::DataCopyGather(vd0, srcUb, vindex, p0);
                    MicroAPI::DataCopyUnAlign(dstUb, vd0, u0, main1);
                }
                MicroAPI::Muls(voffset3, vstride3, (T)size[3], p1);
                MicroAPI::Add(vindex, vbase2, voffset3, p1);
                MicroAPI::DataCopyGather(vd1, srcUb, vindex, p1);
                MicroAPI::DataCopyUnAlign(dstUb, vd1, u0, tail1);
            }
        }
    }
    MicroAPI::DataCopyUnAlignPost(dstUb, u0, 0);
}

template <typename T> __aicore__ inline void GenGatherIndex(__ubuf__ T *indexUb, uint32_t *size, uint32_t *srcStride)
{
    constexpr uint32_t VF_LEN = GetVecLen() / sizeof(T);
    uint32_t sizeI[3];
    uint32_t srcStrideI[3];
    srcStrideI[0] = srcStride[0];
    srcStrideI[1] = srcStride[1];
    srcStrideI[2] = srcStride[2];

    if (size[2] * size[1] * size[0] < VF_LEN) {
        sizeI[0] = size[0];
        sizeI[1] = size[1];
        sizeI[2] = size[2];
    } else if (size[2] * size[1] < VF_LEN) {
        sizeI[0] = VF_LEN / (size[2] * size[1]);
        sizeI[1] = size[1];
        sizeI[2] = size[2];
    } else if (size[2] < VF_LEN) {
        sizeI[0] = 1;
        sizeI[1] = VF_LEN / size[2];
        sizeI[2] = size[2];
    } else {
        sizeI[0] = 1;
        sizeI[1] = 1;
        sizeI[2] = VF_LEN;
    }

    VF_CALL<VfGenIndex<T>>(indexUb, sizeI, srcStrideI, 0);
}

template <typename T> __aicore__ inline void GenGatherIndexForFourDim(__ubuf__ T *indexUb, uint32_t *size, uint32_t *srcStride)
{
    constexpr uint32_t VF_LEN = GetVecLen() / sizeof(T);
    uint32_t sizeI[4];
    uint32_t srcStrideI[4];
    srcStrideI[0] = srcStride[0];
    srcStrideI[1] = srcStride[1];
    srcStrideI[2] = srcStride[2];
    srcStrideI[3] = srcStride[3];

    if (size[3] * size[2] * size[1] * size[0] < VF_LEN) {
        sizeI[0] = size[0];
        sizeI[1] = size[1];
        sizeI[2] = size[2];
        sizeI[3] = size[3];
    } else if (size[3] * size[2] * size[1] < VF_LEN) {
        sizeI[0] = VF_LEN / (size[3] * size[2] * size[1]);
        sizeI[1] = size[1];
        sizeI[2] = size[2];
        sizeI[3] = size[3];
    } else if (size[3] * size[2] < VF_LEN) {
        sizeI[0] = 1;
        sizeI[1] = VF_LEN / (size[3] * size[2]);
        sizeI[2] = size[2];
        sizeI[3] = size[3];
    } else if (size[3] < VF_LEN) {
        sizeI[0] = 1;
        sizeI[1] = 1;
        sizeI[2] = VF_LEN / size[3];
        sizeI[3] = size[3];
    } else {
        sizeI[0] = 1;
        sizeI[1] = 1;
        sizeI[2] = 1;
        sizeI[3] = VF_LEN;
    }

    VF_CALL<VfGenIndexForFourDim<T>>(indexUb, sizeI, srcStrideI, 0);
}

template <typename T>
__aicore__ inline void GatherWrapper(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t *size, uint32_t *srcStride)
{
    constexpr uint32_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t sizeI[3];
    uint16_t srcStrideI[3];
    uint32_t main;
    uint32_t tail;
    uint32_t vlTile0;
    uint32_t vlTile1;
    uint32_t vlTile2;

    if (size[2] * size[1] < VF_LEN) {
        vlTile2 = size[2];
        vlTile1 = size[1];
        vlTile0 = VF_LEN / (vlTile2 * vlTile1);
        sizeI[0] = 1;
        sizeI[1] = 1;
        sizeI[2] = static_cast<uint16_t>(size[0] / vlTile0);
        srcStrideI[0] = 0;
        srcStrideI[1] = 0;
        srcStrideI[2] = static_cast<uint16_t>(srcStride[0] * vlTile0);
        main = vlTile2 * vlTile1 * vlTile0;
        tail = size[2] * size[1] * size[0] - sizeI[2] * main;
    } else if (size[2] < VF_LEN) {
        vlTile2 = size[2];
        vlTile1 = VF_LEN / (vlTile2);
        sizeI[0] = 1;
        sizeI[1] = size[0];
        sizeI[2] = size[1] / vlTile1;
        srcStrideI[0] = 0;
        srcStrideI[1] = static_cast<uint16_t>(srcStride[0]);
        srcStrideI[2] = static_cast<uint16_t>(srcStride[1] * vlTile1);
        main = vlTile2 * vlTile1;
        tail = size[2] * size[1] - sizeI[2] * main;
    } else {
        vlTile2 = VF_LEN;
        sizeI[0] = size[0];
        sizeI[1] = size[1];
        sizeI[2] = size[2] / vlTile2;
        srcStrideI[0] = static_cast<uint16_t>(srcStride[0]);
        srcStrideI[1] = static_cast<uint16_t>(srcStride[1]);
        srcStrideI[2] = static_cast<uint16_t>(srcStride[2] * vlTile2);
        main = vlTile2;
        tail = size[2] - sizeI[2] * main;
    }
    constexpr uint32_t U16_MAX = 65536;
    ASCENDC_ASSERT((sizeI[2] <= U16_MAX), { KERNEL_LOG(KERNEL_ERROR, "shape should less than uint16 max"); });
    ASCENDC_ASSERT((sizeI[1] <= U16_MAX), { KERNEL_LOG(KERNEL_ERROR, "shape should less than uint16 max"); });
    ASCENDC_ASSERT((sizeI[0] <= U16_MAX), { KERNEL_LOG(KERNEL_ERROR, "shape should less than uint16 max"); });
    LocalTensor<T> indexUb;
    PopStackBuffer<T, TPosition::LCM>(indexUb);
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        GenGatherIndex((__ubuf__ int32_t *)indexUb.GetPhyAddr(), size, srcStride);
        VF_CALL<VfGatherBrc<uint32_t>>((__ubuf__ uint32_t *)dstUb, (__ubuf__ uint32_t *)srcUb,
            (__ubuf__ uint32_t *)indexUb.GetPhyAddr(), sizeI, srcStrideI, main, tail);
    } else {
        GenGatherIndex((__ubuf__ int16_t *)indexUb.GetPhyAddr(), size, srcStride);
        VF_CALL<VfGatherBrc<uint16_t>>((__ubuf__ uint16_t *)dstUb, (__ubuf__ uint16_t *)srcUb,
            (__ubuf__ uint16_t *)indexUb.GetPhyAddr(), sizeI, srcStrideI, main, tail);
    }
}

template <typename T>
__aicore__ inline void GatherWrapperForFourDim(__ubuf__ T *dstUb, __ubuf__ T *srcUb, uint32_t *size, uint32_t *srcStride)
{
    constexpr uint32_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t sizeI[4];
    uint16_t srcStrideI[4];
    uint32_t main;
    uint32_t tail;
    uint32_t vlTile0;
    uint32_t vlTile1;
    uint32_t vlTile2;
    uint32_t vlTile3;

    if (size[3] * size[2] * size[1] < VF_LEN) {
        vlTile3 = size[3];
        vlTile2 = size[2];
        vlTile1 = size[1];
        vlTile0 = VF_LEN / (vlTile3 * vlTile2 * vlTile1);
        sizeI[0] = 1;
        sizeI[1] = 1;
        sizeI[2] = 1;
        sizeI[3] = static_cast<uint16_t>(size[0] / vlTile0);
        srcStrideI[0] = 0;
        srcStrideI[1] = 0;
        srcStrideI[2] = 0;
        srcStrideI[3] = static_cast<uint16_t>(srcStride[0] * vlTile0);
        main = vlTile3 * vlTile2 * vlTile1 * vlTile0;
        tail = size[3] * size[2] * size[1] * size[0] - sizeI[3] * main;
    } else if (size[3] * size[2] < VF_LEN) {
        vlTile3 = size[3];
        vlTile2 = size[2];
        vlTile1 = VF_LEN / (vlTile2 * vlTile3);
        sizeI[0] = 1;
        sizeI[1] = 1;
        sizeI[2] = size[0];
        sizeI[3] = static_cast<uint16_t>(size[1] / vlTile1);
        srcStrideI[0] = 0;
        srcStrideI[1] = 0;
        srcStrideI[2] = static_cast<uint16_t>(srcStride[0]);
        srcStrideI[3] = static_cast<uint16_t>(srcStride[1] * vlTile1);
        main = vlTile3 * vlTile2 * vlTile1;
        tail = size[3] * size[2] * size[1] - sizeI[3] * main;
    } else if (size[3] < VF_LEN) {
        vlTile3 = size[3];
        vlTile2 = VF_LEN / vlTile3;
        sizeI[0] = 1;
        sizeI[1] = size[0];;
        sizeI[2] = size[1];
        sizeI[3] = static_cast<uint16_t>(size[2] / vlTile2);
        srcStrideI[0] = 0;
        srcStrideI[1] = static_cast<uint16_t>(srcStride[0]);
        srcStrideI[2] = static_cast<uint16_t>(srcStride[1]);
        srcStrideI[3] = static_cast<uint16_t>(srcStride[2] * vlTile2);
        main = vlTile3 * vlTile2;
        tail = size[3] * size[2] - sizeI[3] * main;
    } else {
        vlTile3 = VF_LEN;
        sizeI[0] = size[0];
        sizeI[1] = size[1];
        sizeI[2] = size[2];
        sizeI[3] = static_cast<uint16_t>(size[3] / vlTile3);
        srcStrideI[0] = static_cast<uint16_t>(srcStride[0]);
        srcStrideI[1] = static_cast<uint16_t>(srcStride[1]);
        srcStrideI[2] = static_cast<uint16_t>(srcStride[2]);
        srcStrideI[3] = static_cast<uint16_t>(srcStride[3] * vlTile3);
        main = vlTile3;
        tail = size[3] - sizeI[3] * main;
    }
    constexpr uint32_t U16_MAX = 65536;
    ASCENDC_ASSERT((sizeI[3] <= U16_MAX), { KERNEL_LOG(KERNEL_ERROR, "shape should less than uint16 max"); });
    ASCENDC_ASSERT((sizeI[2] <= U16_MAX), { KERNEL_LOG(KERNEL_ERROR, "shape should less than uint16 max"); });
    ASCENDC_ASSERT((sizeI[1] <= U16_MAX), { KERNEL_LOG(KERNEL_ERROR, "shape should less than uint16 max"); });
    ASCENDC_ASSERT((sizeI[0] <= U16_MAX), { KERNEL_LOG(KERNEL_ERROR, "shape should less than uint16 max"); });
    LocalTensor<T> indexUb;
    PopStackBuffer<T, TPosition::LCM>(indexUb);
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        GenGatherIndexForFourDim((__ubuf__ int32_t *)indexUb.GetPhyAddr(), size, srcStride);
        VF_CALL<VfGatherBrcForFourDim<uint32_t>>((__ubuf__ uint32_t *)dstUb, (__ubuf__ uint32_t *)srcUb,
            (__ubuf__ uint32_t *)indexUb.GetPhyAddr(), sizeI, srcStrideI, main, tail);
    } else {
        GenGatherIndexForFourDim((__ubuf__ int16_t *)indexUb.GetPhyAddr(), size, srcStride);
        VF_CALL<VfGatherBrcForFourDim<uint16_t>>((__ubuf__ uint16_t *)dstUb, (__ubuf__ uint16_t *)srcUb,
            (__ubuf__ uint16_t *)indexUb.GetPhyAddr(), sizeI, srcStrideI, main, tail);
    }
}
} // namespace AscendC
#endif // IMPL_PAD_BROADCAST_BROADCAST_GATHER_C310_IMPL_H