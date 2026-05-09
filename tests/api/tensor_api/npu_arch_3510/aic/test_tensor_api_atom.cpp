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

    void SetUp()
    { AscendC::SetGCoreType(AscendC::MIX_TYPE); }
    void TearDown()
    { AscendC::SetGCoreType(AscendC::MIX_TYPE); }
};


TEST_F(Tensor_Api_Atom, CopyL0C2GMOperation)
{
    using namespace AscendC::Te;
    using namespace AscendC::Std;

    constexpr uint32_t TILE_LENGTH = 128;

    __cc__ float src[TILE_LENGTH] = {0};
    __gm__ float dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));
    constexpr uint32_t m = 64;
    constexpr uint32_t n = 32;
    auto l0cSrc = MakeTensor(MakeMemPtr<Location::L0C>(src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<float, 16>>(m, n));
    auto gmDst = MakeTensor(MakeMemPtr<Location::GM>(dst), MakeFrameLayout<NDExtLayoutPtn>(m, n));

    auto atomCopy = MakeCopy(CopyL0C2GM{}, CopyL0C2GMTraitDefault{});
    atomCopy.Call(gmDst, l0cSrc);

    CopyAtom<CopyTraits<CopyL0C2GM>>{}.Call(gmDst, l0cSrc);

    Copy(CopyAtom<CopyTraits<CopyL0C2GM>>{}, gmDst, l0cSrc);
    
    FixpipeParams params;
    Copy(CopyAtom<CopyTraits<CopyL0C2GM>>{}, gmDst, l0cSrc, params);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL0C2GMWithOperation)
{
    using namespace AscendC;
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 128;

    __cc__ float src[TILE_LENGTH] = {0};
    __gm__ float dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));
    constexpr uint32_t m = 32;
    constexpr uint32_t n = 64;
    auto l0cSrc = MakeTensor(MakeMemPtr<Location::L0C>(src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<float, 16>>(m, n));
    auto gmDst = MakeTensor(MakeMemPtr<Location::GM>(dst), MakeFrameLayout<NDExtLayoutPtn>(m, n));

    auto atomCopy = MakeCopy(CopyL0C2GM{});
    atomCopy.with(FixpipeParams{}).Call(gmDst, l0cSrc);

    CopyAtom<CopyTraits<CopyL0C2GM>>{}.with(FixpipeParams{}).Call(gmDst, l0cSrc);

    Copy(CopyAtom<CopyTraits<CopyL0C2GM, CopyL0C2GMTraitDefault>>{}.with(FixpipeParams{}), gmDst, l0cSrc);

    EXPECT_EQ(dst[0], 0);
}


TEST_F(Tensor_Api_Atom, CopyL0C2UBOperation)
{
    using namespace AscendC::Te;
    using namespace AscendC::Std;

    constexpr uint32_t TILE_LENGTH = 32 * 32;
    using type = float;

    __cc__ type src[TILE_LENGTH] = {0};
    __ubuf__ type dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));
    auto l0cSrc =
        MakeTensor(MakeMemPtr<Location::L0C>(src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type, 16>>(32, 32));
    auto ubDst = MakeTensor(MakeMemPtr<Location::UB>(dst), MakeFrameLayout<NDExtLayoutPtn>(32, 32));

    auto atomCopy = MakeCopy(CopyL0C2UB{}, CopyL0C2UBTraitDefault{});
    atomCopy.Call(ubDst, l0cSrc);

    CopyAtom<CopyTraits<CopyL0C2UB, CopyL0C2UBTraitDefault>>{}.Call(ubDst, l0cSrc);

    Copy(CopyAtom<CopyTraits<CopyL0C2UB, CopyL0C2UBTraitDefault>>{}, ubDst, l0cSrc);

    FixpipeParams params;
    Copy(CopyAtom<CopyTraits<CopyL0C2UB, CopyL0C2UBTraitDefault>>{}, ubDst, l0cSrc, params);


    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL0C2UBWithOperation)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 32 * 32;
    using type = float;

    __cc__ type src[TILE_LENGTH] = {0};
    __ubuf__ type dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));
    auto l0cSrc =
        MakeTensor(MakeMemPtr<Location::L0C>(src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type, 16>>(32, 32));
    auto ubDst = MakeTensor(MakeMemPtr<Location::UB>(dst), MakeFrameLayout<NDExtLayoutPtn>(32, 32));

    auto atomCopy = MakeCopy(CopyL0C2UB{});
    atomCopy.with(FixpipeParams{}).Call(ubDst, l0cSrc);

    CopyAtom<CopyTraits<CopyL0C2UB, CopyL0C2UBTraitDefault>>{}.with(FixpipeParams{}).Call(ubDst, l0cSrc);

    Copy(CopyAtom<CopyTraits<CopyL0C2UB, CopyL0C2UBTraitDefault>>{}.with(FixpipeParams{}), ubDst, l0cSrc);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL12UBND2ND)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __cbuf__ type src[m * n] = {0};
    __ubuf__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto l1Tensor = MakeTensor(MakeMemPtr<Location::L1>(src), MakeFrameLayout<NDExtLayoutPtn>(m, n));
    auto ubTensor = MakeTensor(MakeMemPtr<Location::UB>(dst), MakeFrameLayout<NDExtLayoutPtn>(m, n));

    Copy(CopyAtom<CopyTraits<CopyL12UB, CopyL12UBTraitDefault>>{}, ubTensor, l1Tensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL12UBDN2DN)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __cbuf__ type src[m * n] = {0};
    __ubuf__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto l1Tensor = MakeTensor(MakeMemPtr<Location::L1>(src), MakeFrameLayout<DNExtLayoutPtn>(m, n));
    auto ubTensor = MakeTensor(MakeMemPtr<Location::UB>(dst), MakeFrameLayout<DNExtLayoutPtn>(m, n));

    Copy(CopyAtom<CopyTraits<CopyL12UB, CopyL12UBTraitDefault>>{}, ubTensor, l1Tensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL12UBNZ2NZ)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __cbuf__ type src[m * n] = {0};
    __ubuf__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto l1Tensor =
        MakeTensor(MakeMemPtr<Location::L1>(src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type>>(m, n));
    auto ubTensor =
        MakeTensor(MakeMemPtr<Location::UB>(dst), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type>>(m, n));

    Copy(CopyAtom<CopyTraits<CopyL12UB, CopyL12UBTraitDefault>>{}, ubTensor, l1Tensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyGM2UBND2ND)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __gm__ type src[m * n] = {0};
    __ubuf__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto gmTensor = MakeTensor(MakeMemPtr<Location::GM>(src), MakeFrameLayout<NDExtLayoutPtn>(m, n));
    auto ubTensor = MakeTensor(MakeMemPtr<Location::UB>(dst), MakeFrameLayout<NDExtLayoutPtn>(m, n));

    Copy(CopyAtom<CopyTraits<CopyGM2UB, CopyGM2UBTraitDefault>>{}, ubTensor, gmTensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyGM2UBDN2DN)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __gm__ type src[m * n] = {0};
    __ubuf__ type dst[m * n] = {0};


    auto gmTensor = MakeTensor(MakeMemPtr<Location::GM>(src), MakeFrameLayout<DNExtLayoutPtn>(m, n));
    auto ubTensor = MakeTensor(MakeMemPtr<Location::UB>(dst), MakeFrameLayout<DNExtLayoutPtn>(m, n));

    Copy(CopyAtom<CopyTraits<CopyGM2UB, CopyGM2UBTraitDefault>>{}, ubTensor, gmTensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyGM2UBNZ2NZ)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __gm__ type src[m * n] = {0};
    __ubuf__ type dst[m * n] = {0};

    auto gmTensor =
        MakeTensor(MakeMemPtr<Location::GM>(src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type>>(m, n));
    auto ubTensor =
        MakeTensor(MakeMemPtr<Location::UB>(dst), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type>>(m, n));

    Copy(CopyAtom<CopyTraits<CopyGM2UB, CopyGM2UBTraitDefault>>{}, ubTensor, gmTensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyUB2L1ND2ND)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __ubuf__ type src[m * n] = {0};
    __cbuf__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));
    ;

    auto ubTensor = MakeTensor(MakeMemPtr<Location::UB>(src), MakeFrameLayout<NDExtLayoutPtn>(m, n));
    auto l1Tensor = MakeTensor(MakeMemPtr<Location::L1>(dst), MakeFrameLayout<NDExtLayoutPtn>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2L1, CopyUB2L1TraitDefault>>{}, l1Tensor, ubTensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyUB2L1DN2DN)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __ubuf__ type src[m * n] = {0};
    __cbuf__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto ubTensor = MakeTensor(MakeMemPtr<Location::UB>(src), MakeFrameLayout<DNExtLayoutPtn>(m, n));
    auto l1Tensor = MakeTensor(MakeMemPtr<Location::L1>(dst), MakeFrameLayout<DNExtLayoutPtn>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2L1, CopyUB2L1TraitDefault>>{}, l1Tensor, ubTensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyUB2L1NZ2NZ)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __ubuf__ type src[m * n] = {0};
    __cbuf__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto ubTensor =
        MakeTensor(MakeMemPtr<Location::UB>(src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type>>(m, n));
    auto l1Tensor =
        MakeTensor(MakeMemPtr<Location::L1>(dst), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type>>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2L1, CopyUB2L1TraitDefault>>{}, l1Tensor, ubTensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyUB2GMND2ND)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __ubuf__ type src[m * n] = {0};
    __gm__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto ubTensor = MakeTensor(MakeMemPtr<Location::UB>(src), MakeFrameLayout<NDExtLayoutPtn>(m, n));
    auto gmTensor = MakeTensor(MakeMemPtr<Location::GM>(dst), MakeFrameLayout<NDExtLayoutPtn>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2GM, CopyUB2GMTraitDefault>>{}, gmTensor, ubTensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyUB2GMDN2DN)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __ubuf__ type src[m * n] = {0};
    __gm__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto ubTensor = MakeTensor(MakeMemPtr<Location::UB>(src), MakeFrameLayout<DNExtLayoutPtn>(m, n));
    auto gmTensor = MakeTensor(MakeMemPtr<Location::GM>(dst), MakeFrameLayout<DNExtLayoutPtn>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2GM, CopyUB2GMTraitDefault>>{}, gmTensor, ubTensor);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyUB2GMNZ2NZ)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    using type = int8_t;

    __ubuf__ type src[m * n] = {0};
    __gm__ type dst[m * n] = {0};

    auto coord = MakeCoord(MakeCoord(Int<0>{}, Int<0>{}), MakeCoord(Int<0>{}, Int<0>{}));

    auto ubTensor =
        MakeTensor(MakeMemPtr<Location::UB>(src), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type>>(m, n));
    auto gmTensor =
        MakeTensor(MakeMemPtr<Location::GM>(dst), MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<type>>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2GM, CopyUB2GMTraitDefault>>{}, gmTensor, ubTensor);

    EXPECT_EQ(dst[0], 0);
}
