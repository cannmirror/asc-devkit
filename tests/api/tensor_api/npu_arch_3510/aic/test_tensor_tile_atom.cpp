/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "tensor_api/stub/cce_stub.h"
#include "include/tensor_api/tensor.h"

class Tensor_Api_Atom : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(Tensor_Api_Atom, CopyGM2L1Operation)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 1024;

    __gm__ float src[TILE_LENGTH] = {0};
    __cbuf__ float dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(Int<20>{}, Int<30>{});
    auto srcLayout = MakeFrameLayout<NDExtLayoutPtn, LayoutTraitDefault<float>>(11, 12);
    auto dstLayout = MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<float>>(11, 12);
    auto gmSrc = MakeTensor(MakeMemPtr<Location::GM>(src), srcLayout);
    auto l1Dst = MakeTensor(MakeMemPtr<Location::L1>(dst), dstLayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, CopyGM2L1TraitDefault{});
    atomCopy.Call(l1Dst, gmSrc);

    atomCopy.Call(l1Dst, gmSrc, coord);

    CopyAtom<CopyTraits<CopyGM2L1, CopyGM2L1TraitDefault>>{}.Call(l1Dst, gmSrc);

    CopyAtom<CopyTraits<CopyGM2L1, CopyGM2L1TraitDefault>>{}.Call(l1Dst, gmSrc, coord);

    Copy(CopyAtom<CopyTraits<CopyGM2L1, CopyGM2L1TraitDefault>>{}, l1Dst, gmSrc);

    Copy(CopyAtom<CopyTraits<CopyGM2L1, CopyGM2L1TraitDefault>>{}, l1Dst, gmSrc, coord);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL12L0Operation)
{
    using namespace AscendC::Te;
    using namespace AscendC::Std;

    constexpr uint32_t TILE_LENGTH = 128;

    __cbuf__ float l1Src[TILE_LENGTH] = {0};
    __ca__ float l0aDst[TILE_LENGTH] = {0};
    __cb__ float l0bDst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(Int<20>{}, Int<30>{});
    auto shape = MakeShape(MakeShape(Int<11>{}, Int<12>{}), MakeShape(Int<13>{}, Int<14>{}));
    auto stride = MakeStride(MakeStride(Int<15>{}, Int<16>{}), MakeStride(Int<17>{}, Int<18>{}));

    auto srcL1 = MakeTensor(MakeMemPtr<Location::L1>(l1Src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<>>(11, 12));
    auto dstL0A = MakeTensor(MakeMemPtr<Location::L0A>(l0aDst), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<>>(11, 12));
    auto dstL0B = MakeTensor(MakeMemPtr<Location::L0B>(l0bDst), MakeFrameLayout<ZNLayoutPtn, LayoutTraitDefault<>>(11, 12));

    auto atomCopyA = MakeCopy(CopyL12L0A{}, CopyL12L0ATraitDefault{});
    atomCopyA.Call(dstL0A, srcL1);

    atomCopyA.Call(dstL0A, srcL1, coord);

    CopyAtom<CopyTraits<CopyL12L0A, CopyL12L0ATraitDefault>>{}.Call(dstL0A, srcL1);

    CopyAtom<CopyTraits<CopyL12L0A, CopyL12L0ATraitDefault>>{}.Call(dstL0A, srcL1, coord);

    Copy(CopyAtom<CopyTraits<CopyL12L0A, CopyL12L0ATraitDefault>>{}, dstL0A, srcL1);

    Copy(CopyAtom<CopyTraits<CopyL12L0A, CopyL12L0ATraitDefault>>{}, dstL0A, srcL1, coord);

    auto atomCopyB = MakeCopy(CopyL12L0B{}, CopyL12L0BTraitDefault{});
    atomCopyB.Call(dstL0B, srcL1);
    atomCopyB.Call(dstL0B, srcL1, coord);

    CopyAtom<CopyTraits<CopyL12L0B, CopyL12L0BTraitDefault>>{}.Call(dstL0B, srcL1);

    CopyAtom<CopyTraits<CopyL12L0B, CopyL12L0BTraitDefault>>{}.Call(dstL0B, srcL1, coord);

    Copy(CopyAtom<CopyTraits<CopyL12L0B, CopyL12L0BTraitDefault>>{}, dstL0B, srcL1);

    Copy(CopyAtom<CopyTraits<CopyL12L0B, CopyL12L0BTraitDefault>>{}, dstL0B, srcL1, coord);

    EXPECT_EQ(l0aDst[0], 0);
    EXPECT_EQ(l0bDst[0], 0);
}

