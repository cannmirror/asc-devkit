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

class Tensor_Api_Cube_Copy_3510 : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override 
    {
        AscendC::SetGCoreType(1);
    }

    void TearDown() override 
    {
        AscendC::SetGCoreType(0);
    }
};

namespace {

template <typename LocationTag, typename Pointer, typename Layout>
auto MakeTensorAt(Pointer ptr, const Layout& layout)
{
    return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<LocationTag>(ptr), layout);
}

template <typename CopyOp, typename Trait, typename DstTensor, typename SrcTensor>
void RunCopyCallPaths(const DstTensor& dst, const SrcTensor& src)
{
    using namespace AscendC::Te;

    auto atom = MakeCopy(CopyOp{}, Trait{});
    atom.Call(dst, src);

    CopyAtom<CopyTraits<CopyOp, Trait>>{}.Call(dst, src);
    Copy(CopyAtom<CopyTraits<CopyOp, Trait>>{}, dst, src);
}

template <typename CopyOp, typename Trait, typename DstTensor, typename SrcTensor>
void RunCopyWithPaths(const DstTensor& dst, const SrcTensor& src)
{
    using namespace AscendC::Te;

    auto atom = CopyAtom<CopyTraits<CopyOp, Trait>>{}.with();
    atom.Call(dst, src);
    Copy(atom, dst, src);
}

} // namespace

TEST_F(Tensor_Api_Cube_Copy_3510, CopyGM2L1RoutesToCubeArchCopy)
{
    using namespace AscendC::Te;

    constexpr uint32_t m = 32;
    constexpr uint32_t n = 32;
    __gm__ float src[m * n] = {0};
    __cbuf__ float dst[m * n] = {0};

    auto gmTensor = MakeTensorAt<Location::GM>(src, MakeFrameLayout<NDExtLayoutPtn, LayoutTraitDefault<float>>(m, n));
    auto l1Tensor = MakeTensorAt<Location::L1>(dst, MakeFrameLayout<NZLayoutPtn, LayoutTraitDefault<float>>(m, n));

    RunCopyCallPaths<CopyGM2L1, CopyGM2L1TraitDefault>(l1Tensor, gmTensor);
    RunCopyWithPaths<CopyGM2L1, CopyGM2L1TraitDefault>(l1Tensor, gmTensor);

    EXPECT_EQ(dst[0], 0);
}
