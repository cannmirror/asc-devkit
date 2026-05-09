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

class Tensor_Api_Tensor : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(Tensor_Api_Tensor, ViewEngineOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 4;
    __gm__ float data[TILE_LENGTH] = {1, 2, 3, 4};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeLayout(MakeShape(2, 2), MakeStride(2, 1));
    auto tensor = MakeTensor(ptr, layout);

    auto engine = tensor.Engine();
    EXPECT_EQ(engine.Begin(), ptr);
    EXPECT_EQ(engine.Begin()[0], 1);
    EXPECT_EQ(engine.Begin()[3], 4);

    ViewEngine<decltype(ptr)> viewEngine(ptr);
    EXPECT_EQ(viewEngine.Begin(), ptr);
    EXPECT_EQ(viewEngine.Begin()[1], 2);

    ConstViewEngine<decltype(ptr)> constViewEngine(ptr);
    EXPECT_EQ(constViewEngine.Begin(), ptr);
    EXPECT_EQ(constViewEngine.Begin()[1], 2);

    ViewEngine<decltype(ptr)> defaultEngine;
    EXPECT_EQ(defaultEngine.Begin().Get(), nullptr);

    ConstViewEngine<decltype(ptr)> defaultConstEngine;
    EXPECT_EQ(defaultConstEngine.Begin().Get(), nullptr);

    constexpr uint32_t TILE_LENGTH_3D = 8;
    __gm__ float data3d[TILE_LENGTH_3D] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto ptr3d = MakeMemPtr<Location::GM>(data3d);
    auto layout3d = MakeLayout(MakeShape(2, 2, 2), MakeStride(4, 2, 1));
    auto tensor3d = MakeTensor(ptr3d, layout3d);

    auto engine3d = tensor3d.Engine();
    EXPECT_EQ(engine3d.Begin(), ptr3d);
    EXPECT_EQ(engine3d.Begin()[0], 1);
    EXPECT_EQ(engine3d.Begin()[7], 8);

    ViewEngine<decltype(ptr3d)> viewEngine3d(ptr3d);
    EXPECT_EQ(viewEngine3d.Begin(), ptr3d);
    EXPECT_EQ(viewEngine3d.Begin()[1], 2);

    ConstViewEngine<decltype(ptr3d)> constViewEngine3d(ptr3d);
    EXPECT_EQ(constViewEngine3d.Begin(), ptr3d);
    EXPECT_EQ(constViewEngine3d.Begin()[1], 2);
}

TEST_F(Tensor_Api_Tensor, ViewEngineOperationInt)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 4;
    __gm__ float data[TILE_LENGTH] = {5, 6, 7, 8};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeLayout(MakeShape(AscendC::Std::Int<1>{}, AscendC::Std::Int<4>{}),
                             MakeStride(AscendC::Std::Int<4>{}, AscendC::Std::Int<1>{}));
    auto tensor = MakeTensor(ptr, layout);

    auto engine = tensor.Engine();
    EXPECT_EQ(engine.Begin(), ptr);
    EXPECT_EQ(engine.Begin()[0], 5);
    EXPECT_EQ(engine.Begin()[3], 8);

    ViewEngine<decltype(ptr)> viewEngine(ptr);
    EXPECT_EQ(viewEngine.Begin(), ptr);
    EXPECT_EQ(viewEngine.Begin()[1], 6);

    ConstViewEngine<decltype(ptr)> constViewEngine(ptr);
    EXPECT_EQ(constViewEngine.Begin(), ptr);
    EXPECT_EQ(constViewEngine.Begin()[1], 6);

    constexpr uint32_t TILE_LENGTH_3D = 8;
    __gm__ float data3d[TILE_LENGTH_3D] = {9, 10, 11, 12, 13, 14, 15, 16};
    auto ptr3d = MakeMemPtr<Location::GM>(data3d);
    auto layout3d = MakeLayout(MakeShape(AscendC::Std::Int<2>{}, AscendC::Std::Int<2>{}, AscendC::Std::Int<2>{}),
                               MakeStride(AscendC::Std::Int<4>{}, AscendC::Std::Int<2>{}, AscendC::Std::Int<1>{}));
    auto tensor3d = MakeTensor(ptr3d, layout3d);

    auto engine3d = tensor3d.Engine();
    EXPECT_EQ(engine3d.Begin(), ptr3d);
    EXPECT_EQ(engine3d.Begin()[0], 9);
    EXPECT_EQ(engine3d.Begin()[7], 16);

    ViewEngine<decltype(ptr3d)> viewEngine3d(ptr3d);
    EXPECT_EQ(viewEngine3d.Begin(), ptr3d);
    EXPECT_EQ(viewEngine3d.Begin()[1], 10);

    ConstViewEngine<decltype(ptr3d)> constViewEngine3d(ptr3d);
    EXPECT_EQ(constViewEngine3d.Begin(), ptr3d);
    EXPECT_EQ(constViewEngine3d.Begin()[1], 10);
}

TEST_F(Tensor_Api_Tensor, IteratorMakeMemPtrOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 128;
    __gm__ float gmData[TILE_LENGTH] = {0};
    __ubuf__ float ubData[TILE_LENGTH] = {0};
    __cbuf__ float l1Data[TILE_LENGTH] = {0};
    __ca__ float l0aData[TILE_LENGTH] = {0};
    __cb__ float l0bData[TILE_LENGTH] = {0};
    __cc__ float l0cData[TILE_LENGTH] = {0};
    __biasbuf__ float biasData[TILE_LENGTH] = {0};
    __fbuf__ float fixbufData[TILE_LENGTH] = {0};

    auto gmPtr = MakeMemPtr<Location::GM>(gmData);
    auto ubPtr = MakeMemPtr<Location::UB>(ubData);
    auto l1Ptr = MakeMemPtr<Location::L1>(l1Data);
    auto l0aPtr = MakeMemPtr<Location::L0A>(l0aData);
    auto l0bPtr = MakeMemPtr<Location::L0B>(l0bData);
    auto l0cPtr = MakeMemPtr<Location::L0C>(l0cData);
    auto biasPtr = MakeMemPtr<Location::BIAS>(biasData);
    auto fixbufPtr = MakeMemPtr<Location::FIXBUF>(fixbufData);

    EXPECT_EQ(gmPtr.Get(), gmData);
    EXPECT_EQ(ubPtr.Get(), ubData);
    EXPECT_EQ(l1Ptr.Get(), l1Data);
    EXPECT_EQ(l0aPtr.Get(), l0aData);
    EXPECT_EQ(l0bPtr.Get(), l0bData);
    EXPECT_EQ(l0cPtr.Get(), l0cData);
    EXPECT_EQ(biasPtr.Get(), biasData);
    EXPECT_EQ(fixbufPtr.Get(), fixbufData);

    EXPECT_EQ(MakeMemPtr<Location::GM>(gmPtr), gmPtr);
    EXPECT_EQ(MakeMemPtr<Location::UB>(ubPtr), ubPtr);
    EXPECT_EQ(MakeMemPtr<Location::L1>(l1Ptr), l1Ptr);
    EXPECT_EQ(MakeMemPtr<Location::L0A>(l0aPtr), l0aPtr);
    EXPECT_EQ(MakeMemPtr<Location::L0B>(l0bPtr), l0bPtr);
    EXPECT_EQ(MakeMemPtr<Location::L0C>(l0cPtr), l0cPtr);
    EXPECT_EQ(MakeMemPtr<Location::BIAS>(biasPtr), biasPtr);
    EXPECT_EQ(MakeMemPtr<Location::FIXBUF>(fixbufPtr), fixbufPtr);
}

 TEST_F(Tensor_Api_Tensor, IteratorMakeMemPatternPtrOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 128;
    __gm__ float gmData[TILE_LENGTH] = {0};
    __ubuf__ float ubData[TILE_LENGTH] = {0};
    __cbuf__ float l1Data[TILE_LENGTH] = {0};
    __ca__ float l0aData[TILE_LENGTH] = {0};
    __cb__ float l0bData[TILE_LENGTH] = {0};
    __cc__ float l0cData[TILE_LENGTH] = {0};
    __biasbuf__ float biasData[TILE_LENGTH] = {0};
    __fbuf__ float fixbufData[TILE_LENGTH] = {0};

    auto gmPtr = MakeMemPtr<Location::GM>(gmData);
    auto ubPtr = MakeMemPtr<Location::UB>(ubData);
    auto l1Ptr = MakeMemPtr<Location::L1>(l1Data);
    auto l0aPtr = MakeMemPtr<Location::L0A>(l0aData);
    auto l0bPtr = MakeMemPtr<Location::L0B>(l0bData);
    auto l0cPtr = MakeMemPtr<Location::L0C>(l0cData);
    auto biasPtr = MakeMemPtr<Location::BIAS>(biasData);
    auto fixbufPtr = MakeMemPtr<Location::FIXBUF>(fixbufData);
    EXPECT_EQ(gmPtr.Get(), gmData);
    EXPECT_EQ(ubPtr.Get(), ubData);
    EXPECT_EQ(l1Ptr.Get(), l1Data);
    EXPECT_EQ(l0aPtr.Get(), l0aData);
    EXPECT_EQ(l0bPtr.Get(), l0bData);
    EXPECT_EQ(l0cPtr.Get(), l0cData);
    EXPECT_EQ(biasPtr.Get(), biasData);
    EXPECT_EQ(fixbufPtr.Get(), fixbufData);
}

TEST_F(Tensor_Api_Tensor, ByteOffsetMakeMemPatternPtrOperation)
{
    using namespace AscendC::Te;

    struct FloatTrait {
        using type = float;
    };

    constexpr uint64_t BYTE_OFFSET = 128;
    auto ubPtr = MakeMemPtr<Location::UB, FloatTrait>(BYTE_OFFSET);
    auto l1Ptr = MakeMemPtr<Location::L1, FloatTrait>(BYTE_OFFSET);
    auto l0aPtr = MakeMemPtr<Location::L0A, FloatTrait>(BYTE_OFFSET);
    auto l0bPtr = MakeMemPtr<Location::L0B, FloatTrait>(BYTE_OFFSET);
    auto l0cPtr = MakeMemPtr<Location::L0C, FloatTrait>(BYTE_OFFSET);
    auto biasPtr = MakeMemPtr<Location::BIAS, FloatTrait>(BYTE_OFFSET);
    auto fixbufPtr = MakeMemPtr<Location::FIXBUF, FloatTrait>(BYTE_OFFSET);

    EXPECT_EQ(ubPtr.Get(), reinterpret_cast<__ubuf__ float*>(get_imm(0) + BYTE_OFFSET));
    EXPECT_EQ(l1Ptr.Get(), reinterpret_cast<__cbuf__ float*>(get_imm(0) + BYTE_OFFSET));
    EXPECT_EQ(l0aPtr.Get(), reinterpret_cast<__ca__ float*>(get_imm(0) + BYTE_OFFSET));
    EXPECT_EQ(l0bPtr.Get(), reinterpret_cast<__cb__ float*>(get_imm(0) + BYTE_OFFSET));
    EXPECT_EQ(l0cPtr.Get(), reinterpret_cast<__cc__ float*>(get_imm(0) + BYTE_OFFSET));
    EXPECT_EQ(biasPtr.Get(), reinterpret_cast<__biasbuf__ float*>(get_imm(0) + BYTE_OFFSET));
    EXPECT_EQ(fixbufPtr.Get(), reinterpret_cast<__fbuf__ float*>(get_imm(0) + BYTE_OFFSET));
}

TEST_F(Tensor_Api_Tensor, IteratorGetOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 4;
    __gm__ float data[TILE_LENGTH] = {1, 2, 3, 4};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto offsetPtr = ptr + 2;

    EXPECT_EQ(ptr.Get(), data);
    EXPECT_EQ(offsetPtr.Get(), data + 2);
}

TEST_F(Tensor_Api_Tensor, IteratorOperatorOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 4;
    __gm__ float data[TILE_LENGTH] = {1, 2, 3, 4};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto next = ptr + 1;
    auto far = ptr + 3;

    EXPECT_EQ(*ptr, 1);
    EXPECT_EQ(ptr[2], 3);

    *ptr = 10;
    next[1] = 20;
    EXPECT_EQ(data[0], 10);
    EXPECT_EQ(data[2], 20);

    EXPECT_TRUE(ptr == ptr);
    EXPECT_TRUE(ptr != next);
    EXPECT_TRUE(ptr < next);
    EXPECT_TRUE(next > ptr);
    EXPECT_TRUE(ptr <= next);
    EXPECT_TRUE(far >= next);
}

TEST_F(Tensor_Api_Tensor, LocalTensorOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 6;
    __gm__ float data[TILE_LENGTH] = {0, 1, 2, 3, 4, 5};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeLayout(MakeShape(2, 3), MakeStride(3, 1));
    auto tensor = MakeTensor(ptr, layout);

    EXPECT_EQ(tensor.Tensor().Data(), ptr);
    EXPECT_EQ(tensor.Engine().Begin(), ptr);
    EXPECT_EQ(tensor.Layout().Size(), 6);
    EXPECT_EQ(tensor.Data(), ptr);

    auto shape = tensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape), 2);
    EXPECT_EQ(AscendC::Std::get<1>(shape), 3);

    auto stride = tensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride), 3);
    EXPECT_EQ(AscendC::Std::get<1>(stride), 1);

    EXPECT_EQ(tensor.Size(), 6);
    EXPECT_EQ(tensor.Capacity(), 6);
}

TEST_F(Tensor_Api_Tensor, LocalTensorOperationInt)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 6;
    __gm__ float data[TILE_LENGTH] = {6, 7, 8, 9, 10, 11};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeLayout(MakeShape(AscendC::Std::Int<3>{}, AscendC::Std::Int<2>{}),
                             MakeStride(AscendC::Std::Int<2>{}, AscendC::Std::Int<1>{}));
    auto tensor = MakeTensor(ptr, layout);

    EXPECT_EQ(tensor.Layout().Size(), 6);

    auto shape = tensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape).value, 3);
    EXPECT_EQ(AscendC::Std::get<1>(shape).value, 2);

    auto stride = tensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride).value, 2);
    EXPECT_EQ(AscendC::Std::get<1>(stride).value, 1);
}

TEST_F(Tensor_Api_Tensor, LocalTensorOperation3D)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 8;
    __gm__ float data[TILE_LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeLayout(MakeShape(2, 2, 2), MakeStride(4, 2, 1));
    auto tensor = MakeTensor(ptr, layout);

    EXPECT_EQ(tensor.Tensor().Data(), ptr);
    EXPECT_EQ(tensor.Engine().Begin(), ptr);
    EXPECT_EQ(tensor.Layout().Size(), 8);
    EXPECT_EQ(tensor.Data(), ptr);

    auto shape = tensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape), 2);
    EXPECT_EQ(AscendC::Std::get<1>(shape), 2);
    EXPECT_EQ(AscendC::Std::get<2>(shape), 2);

    auto stride = tensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride), 4);
    EXPECT_EQ(AscendC::Std::get<1>(stride), 2);
    EXPECT_EQ(AscendC::Std::get<2>(stride), 1);

    EXPECT_EQ(tensor.Size(), 8);
    EXPECT_EQ(tensor.Capacity(), 8);
}

TEST_F(Tensor_Api_Tensor, LocalTensorConstOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 6;
    __gm__ float data[TILE_LENGTH] = {0, 1, 2, 3, 4, 5};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeLayout(MakeShape(2, 3), MakeStride(3, 1));
    const auto constTensor = MakeTensor(ptr, layout);

    EXPECT_EQ(constTensor.Tensor().Data(), ptr);
    EXPECT_EQ(constTensor.Engine().Begin(), ptr);
    EXPECT_EQ(constTensor.Layout().Size(), 6);
    EXPECT_EQ(constTensor.Data(), ptr);

    auto shape = constTensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape), 2);
    EXPECT_EQ(AscendC::Std::get<1>(shape), 3);

    auto stride = constTensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride), 3);
    EXPECT_EQ(AscendC::Std::get<1>(stride), 1);

    EXPECT_EQ(constTensor.Size(), 6);
    EXPECT_EQ(constTensor.Capacity(), 6);
}

TEST_F(Tensor_Api_Tensor, LocalTensorConstOperationInt)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 6;
    __gm__ float data[TILE_LENGTH] = {10, 11, 12, 13, 14, 15};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeLayout(MakeShape(AscendC::Std::Int<1>{}, AscendC::Std::Int<6>{}),
                             MakeStride(AscendC::Std::Int<6>{}, AscendC::Std::Int<1>{}));
    const auto constTensor = MakeTensor(ptr, layout);

    EXPECT_EQ(constTensor.Layout().Size(), 6);

    auto shape = constTensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape).value, 1);
    EXPECT_EQ(AscendC::Std::get<1>(shape).value, 6);

    auto stride = constTensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride).value, 6);
    EXPECT_EQ(AscendC::Std::get<1>(stride).value, 1);
}

TEST_F(Tensor_Api_Tensor, LocalTensorConstOperation3D)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 8;
    __gm__ float data[TILE_LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeLayout(MakeShape(AscendC::Std::Int<2>{}, AscendC::Std::Int<2>{}, AscendC::Std::Int<2>{}),
                             MakeStride(AscendC::Std::Int<4>{}, AscendC::Std::Int<2>{}, AscendC::Std::Int<1>{}));
    const auto constTensor = MakeTensor(ptr, layout);

    EXPECT_EQ(constTensor.Tensor().Data(), ptr);
    EXPECT_EQ(constTensor.Engine().Begin(), ptr);
    EXPECT_EQ(constTensor.Layout().Size(), 8);
    EXPECT_EQ(constTensor.Data(), ptr);

    auto shape = constTensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape).value, 2);
    EXPECT_EQ(AscendC::Std::get<1>(shape).value, 2);
    EXPECT_EQ(AscendC::Std::get<2>(shape).value, 2);

    auto stride = constTensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride).value, 4);
    EXPECT_EQ(AscendC::Std::get<1>(stride).value, 2);
    EXPECT_EQ(AscendC::Std::get<2>(stride).value, 1);

    EXPECT_EQ(constTensor.Size(), 8);
    EXPECT_EQ(constTensor.Capacity(), 8);
}

TEST_F(Tensor_Api_Tensor, LocalTensorCoordViewOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 12;
    __gm__ float data[TILE_LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeFrameLayout<NDLayoutPtn, LayoutTraitDefault<float, 1>>(3, 4);
    auto tensor = MakeTensor(ptr, layout);

    auto subTensor = tensor(MakeCoord(1, 1));
    EXPECT_EQ(subTensor.Data(), ptr + layout(MakeCoord(1, 1)));

    auto shape = subTensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape), 2);
    EXPECT_EQ(AscendC::Std::get<1>(shape), 3);

    auto stride = subTensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride), 4);
    EXPECT_EQ(AscendC::Std::get<1>(stride), 1);

    EXPECT_EQ(subTensor[MakeCoord(0, 0)], 5);
    EXPECT_EQ(subTensor[MakeCoord(1, 2)], 11);
}

TEST_F(Tensor_Api_Tensor, LocalTensorCoordViewOperation3D)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 24;
    __gm__ float data[TILE_LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakePatternLayout<NDLayoutPtn, LayoutTraitDefault<float, 1>>(
        MakeShape(2, 3, 4), MakeStride(12, 4, 1));
    auto tensor = MakeTensor(ptr, layout);

    auto subTensor = tensor(MakeCoord(1, 1, 2));
    EXPECT_EQ(subTensor.Data(), ptr + layout(MakeCoord(1, 1, 2)));

    auto shape = subTensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shape), 2);
    EXPECT_EQ(AscendC::Std::get<2>(shape), 2);

    auto stride = subTensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride), 12);
    EXPECT_EQ(AscendC::Std::get<1>(stride), 4);
    EXPECT_EQ(AscendC::Std::get<2>(stride), 1);

    EXPECT_EQ(subTensor[MakeCoord(0, 0, 0)], 18);
    EXPECT_EQ(subTensor[MakeCoord(0, 1, 1)], 23);
}

TEST_F(Tensor_Api_Tensor, LocalTensorSliceOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 48;
    __gm__ float data[TILE_LENGTH] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeFrameLayout<ScaleANDLayoutPtn, LayoutTraitDefault<float, 2>>(6, 8);
    auto tensor = MakeTensor(ptr, layout);
    auto coord = MakeCoord(1, 2);

    auto shapeSlice = Slice(tensor, coord, MakeShape(3, 3));
    EXPECT_EQ(shapeSlice.Data(), ptr + layout(coord));
    auto shapeSliceShape = shapeSlice.Shape();
    auto shapeSliceStride = shapeSlice.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<0>(shapeSliceShape)), 1);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(shapeSliceShape)), 3);
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<1>(shapeSliceShape)), 1);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(shapeSliceShape)), 3);

    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<0>(shapeSliceStride)), 0);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(shapeSliceStride)), 8);
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<1>(shapeSliceStride)), 0);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(shapeSliceStride)), 1);

    EXPECT_EQ(shapeSlice[MakeCoord(MakeCoord(0, 0), MakeCoord(0, 0))], 10);
    EXPECT_EQ(shapeSlice[MakeCoord(MakeCoord(0, 2), MakeCoord(0, 1))], 27);
}

TEST_F(Tensor_Api_Tensor, LocalTensorSliceWithLayoutInfoOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 48;
    __gm__ float data[TILE_LENGTH] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto layout = MakeFrameLayout<ScaleANDLayoutPtn, LayoutTraitDefault<float, 2>>(6, 8);

    auto tensor = MakeTensor(ptr, layout);
    auto coord = MakeCoord(MakeCoord(0, 1), MakeCoord(0, 2));
    auto infoLayout = MakeFrameLayout<ScaleANDLayoutPtn, LayoutTraitDefault<float, 2>>(2, 4);

    auto layoutSlice = Slice(tensor, coord, infoLayout);
    EXPECT_EQ(layoutSlice.Data(), ptr + layout(coord));

    auto layoutSliceShape = layoutSlice.Shape();
    auto layoutSliceStride = layoutSlice.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<0>(layoutSliceShape)), 1);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layoutSliceShape)), 2);
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<1>(layoutSliceShape)), 1);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(layoutSliceShape)), 4);

    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<0>(layoutSliceStride)), 0);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<0>(layoutSliceStride)), 8);
    EXPECT_EQ(AscendC::Std::get<0>(AscendC::Std::get<1>(layoutSliceStride)), 0);
    EXPECT_EQ(AscendC::Std::get<1>(AscendC::Std::get<1>(layoutSliceStride)), 1);

    EXPECT_EQ(layoutSlice[MakeCoord(MakeCoord(0, 0), MakeCoord(0, 0))], 10);
    EXPECT_EQ(layoutSlice[MakeCoord(MakeCoord(0, 1), MakeCoord(0, 3))], 21);
}

TEST_F(Tensor_Api_Tensor, MakeTensorShapeStrideBranchOperation)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 6;
    __gm__ float data[TILE_LENGTH] = {10, 11, 12, 13, 14, 15};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto tensor = MakeTensor(ptr, MakeShape(2, 3), MakeStride(3, 1));

    EXPECT_EQ(tensor.Data(), ptr);
    EXPECT_EQ(tensor.Layout().Size(), 6);

    auto shape = tensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape), 2);
    EXPECT_EQ(AscendC::Std::get<1>(shape), 3);

    auto stride = tensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride), 3);
    EXPECT_EQ(AscendC::Std::get<1>(stride), 1);
}

TEST_F(Tensor_Api_Tensor, MakeTensorShapeStrideBranchOperationInt)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 6;
    __gm__ float data[TILE_LENGTH] = {10, 11, 12, 13, 14, 15};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto tensor = MakeTensor(ptr, MakeShape(AscendC::Std::Int<2>{}, AscendC::Std::Int<3>{}),
                             MakeStride(AscendC::Std::Int<3>{}, AscendC::Std::Int<1>{}));

    EXPECT_EQ(tensor.Data(), ptr);
    EXPECT_EQ(tensor.Layout().Size(), 6);

    auto shape = tensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape).value, 2);
    EXPECT_EQ(AscendC::Std::get<1>(shape).value, 3);

    auto stride = tensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride).value, 3);
    EXPECT_EQ(AscendC::Std::get<1>(stride).value, 1);
}

TEST_F(Tensor_Api_Tensor, MakeTensorShapeStrideBranchOperationInt3D)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 8;
    __gm__ float data[TILE_LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto tensor = MakeTensor(ptr, MakeShape(AscendC::Std::Int<2>{}, AscendC::Std::Int<2>{}, AscendC::Std::Int<2>{}),
                             MakeStride(AscendC::Std::Int<4>{}, AscendC::Std::Int<2>{}, AscendC::Std::Int<1>{}));

    EXPECT_EQ(tensor.Data(), ptr);
    EXPECT_EQ(tensor.Layout().Size(), 8);

    auto shape = tensor.Shape();
    EXPECT_EQ(AscendC::Std::get<0>(shape).value, 2);
    EXPECT_EQ(AscendC::Std::get<1>(shape).value, 2);
    EXPECT_EQ(AscendC::Std::get<2>(shape).value, 2);

    auto stride = tensor.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(stride).value, 4);
    EXPECT_EQ(AscendC::Std::get<1>(stride).value, 2);
    EXPECT_EQ(AscendC::Std::get<2>(stride).value, 1);
}
