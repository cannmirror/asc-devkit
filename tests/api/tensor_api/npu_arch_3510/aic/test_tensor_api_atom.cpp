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
#include "include/experimental/tensor_api/tensor.h"

class Tensor_Api_Atom : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(Tensor_Api_Atom, CopyL0C2UBOperation)
{
    using namespace AscendC::Te;
    using namespace AscendC::Std;

    constexpr uint32_t TILE_LENGTH = 32 * 32;

    __cc__ float src[TILE_LENGTH] = {0};
    __ubuf__ float dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(Int<0>{}, Int<0>{});
    auto l0cSrc = MakeTensor(MakeL0CmemPtr(src), MakeL0CLayout(32, 32));
    auto ubDst = MakeTensor(MakeUBmemPtr(dst), MakeNDLayout<float>(32, 32));

    auto atomCopy = MakeCopy(CopyL0C2UB{}, FixpipeTraitDefault{});
    atomCopy.Call(ubDst, l0cSrc);

    atomCopy.Call(ubDst, l0cSrc, coord);

    CopyAtom<CopyTraits<CopyL0C2UB, FixpipeTraitDefault>>{}.Call(ubDst, l0cSrc);

    CopyAtom<CopyTraits<CopyL0C2UB, FixpipeTraitDefault>>{}.Call(ubDst, l0cSrc, coord);

    Copy(CopyAtom<CopyTraits<CopyL0C2UB, FixpipeTraitDefault>>{}, ubDst, l0cSrc);

    Copy(CopyAtom<CopyTraits<CopyL0C2UB, FixpipeTraitDefault>>{}, ubDst, l0cSrc, coord);

    FixpipeParams params;
    Copy(CopyAtom<CopyTraits<CopyL0C2UB, FixpipeTraitDefault>>{}, ubDst, l0cSrc, params);

    Copy(CopyAtom<CopyTraits<CopyL0C2UB, FixpipeTraitDefault>>{}, ubDst, l0cSrc, coord, params);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL0C2UBWithOperation)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 32 * 32;

    __cc__ float src[TILE_LENGTH] = {0};
    __ubuf__ float dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(Int<0>{}, Int<0>{});
    auto l0cSrc = MakeTensor(MakeL0CmemPtr(src), MakeL0CLayout(32, 32));
    auto ubDst = MakeTensor(MakeUBmemPtr(dst), MakeNDLayout<float>(32, 32));

    auto atomCopy = MakeCopy(CopyL0C2UB{});
    atomCopy.with(12).Call(ubDst, l0cSrc);

    atomCopy.with(23).Call(ubDst, l0cSrc, coord);

    CopyAtom<CopyTraits<CopyL0C2UB>>{}.with(34).Call(ubDst, l0cSrc);

    CopyAtom<CopyTraits<CopyL0C2UB>>{}.with(45).Call(ubDst, l0cSrc, coord);

    Copy(CopyAtom<CopyTraits<CopyL0C2UB>>{}.with(56), ubDst, l0cSrc);

    Copy(CopyAtom<CopyTraits<CopyL0C2UB>>{}.with(67), ubDst, l0cSrc, coord);

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

    auto l1Tensor = MakeTensor(MakeL1memPtr(src), MakeNDLayout<type>(m, n));
    auto ubTensor = MakeTensor(MakeUBmemPtr(dst), MakeNDLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyL12UB, DataCopyTraitDefault>>{}, ubTensor, l1Tensor);

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

    auto l1Tensor = MakeTensor(MakeL1memPtr(src), MakeDNLayout<type>(m, n));
    auto ubTensor = MakeTensor(MakeUBmemPtr(dst), MakeDNLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyL12UB, DataCopyTraitDefault>>{}, ubTensor, l1Tensor);

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

    auto l1Tensor = MakeTensor(MakeL1memPtr(src), MakeNzLayout<type>(m, n));
    auto ubTensor = MakeTensor(MakeUBmemPtr(dst), MakeNzLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyL12UB, DataCopyTraitDefault>>{}, ubTensor, l1Tensor);

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

    auto gmTensor = MakeTensor(MakeGMmemPtr(src), MakeNDLayout<type>(m, n));
    auto ubTensor = MakeTensor(MakeUBmemPtr(dst), MakeNDLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyGM2UB, DataCopyTraitDefault>>{}, ubTensor, gmTensor);

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

    auto gmTensor = MakeTensor(MakeGMmemPtr(src), MakeDNLayout<type>(m, n));
    auto ubTensor = MakeTensor(MakeUBmemPtr(dst), MakeDNLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyGM2UB, DataCopyTraitDefault>>{}, ubTensor, gmTensor);

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

    auto gmTensor = MakeTensor(MakeGMmemPtr(src), MakeNzLayout<type>(m, n));
    auto ubTensor = MakeTensor(MakeUBmemPtr(dst), MakeNzLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyGM2UB, DataCopyTraitDefault>>{}, ubTensor, gmTensor);

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

    auto ubTensor = MakeTensor(MakeUBmemPtr(src), MakeNDLayout<type>(m, n));
    auto l1Tensor = MakeTensor(MakeL1memPtr(dst), MakeNDLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2L1, DataCopyTraitDefault>>{}, l1Tensor, ubTensor);

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

    auto ubTensor = MakeTensor(MakeUBmemPtr(src), MakeDNLayout<type>(m, n));
    auto l1Tensor = MakeTensor(MakeL1memPtr(dst), MakeDNLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2L1, DataCopyTraitDefault>>{}, l1Tensor, ubTensor);

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

    auto ubTensor = MakeTensor(MakeUBmemPtr(src), MakeNzLayout<type>(m, n));
    auto l1Tensor = MakeTensor(MakeL1memPtr(dst), MakeNzLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2L1, DataCopyTraitDefault>>{}, l1Tensor, ubTensor);

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

    auto ubTensor = MakeTensor(MakeUBmemPtr(src), MakeNDLayout<type>(m, n));
    auto gmTensor = MakeTensor(MakeGMmemPtr(dst), MakeNDLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2GM, DataCopyTraitDefault>>{}, gmTensor, ubTensor);

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

    auto ubTensor = MakeTensor(MakeUBmemPtr(src), MakeDNLayout<type>(m, n));
    auto gmTensor = MakeTensor(MakeGMmemPtr(dst), MakeDNLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2GM, DataCopyTraitDefault>>{}, gmTensor, ubTensor);

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

    auto ubTensor = MakeTensor(MakeUBmemPtr(src), MakeNzLayout<type>(m, n));
    auto gmTensor = MakeTensor(MakeGMmemPtr(dst), MakeNzLayout<type>(m, n));

    Copy(CopyAtom<CopyTraits<CopyUB2GM, DataCopyTraitDefault>>{}, gmTensor, ubTensor);

    EXPECT_EQ(dst[0], 0);
}
