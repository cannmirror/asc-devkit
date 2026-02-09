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


class Tensor_Api_Layout : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

template <size_t v>
using Int = AscendC::Std::integral_constant<size_t, v>; 

TEST_F(Tensor_Api_Layout, ShapeAndStrideOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shapeInit{11, 22, 33};
    Stride<int, int, int> strideInit{44, 55, 66};

    Shape<int, int, int> shapeMake = MakeShape(10, 20, 30);
    Stride<int, int, int> strideMake = MakeStride(40, 50, 60);

    auto shapeMakeInt = MakeShape(Int<111>{}, Int<222>{}, Int<333>{});
    auto strideMakeInt = MakeStride(Int<444>{}, Int<555>{}, Int<666>{});

    EXPECT_EQ(AscendC::Std::get<0>(shapeInit), 11);
    EXPECT_EQ(AscendC::Std::get<1>(shapeInit), 22);
    EXPECT_EQ(AscendC::Std::get<2>(shapeInit), 33);

    EXPECT_EQ(AscendC::Std::get<0>(strideInit), 44);
    EXPECT_EQ(AscendC::Std::get<1>(strideInit), 55);
    EXPECT_EQ(AscendC::Std::get<2>(strideInit), 66);

    EXPECT_EQ(AscendC::Std::get<0>(shapeMake), 10);
    EXPECT_EQ(AscendC::Std::get<1>(shapeMake), 20);
    EXPECT_EQ(AscendC::Std::get<2>(shapeMake), 30);

    EXPECT_EQ(AscendC::Std::get<0>(strideMake), 40);
    EXPECT_EQ(AscendC::Std::get<1>(strideMake), 50);
    EXPECT_EQ(AscendC::Std::get<2>(strideMake), 60);

    EXPECT_EQ(AscendC::Std::get<0>(shapeMakeInt).value, 111);
    EXPECT_EQ(AscendC::Std::get<1>(shapeMakeInt).value, 222);
    EXPECT_EQ(AscendC::Std::get<2>(shapeMakeInt).value, 333);

    EXPECT_EQ(AscendC::Std::get<0>(strideMakeInt).value, 444);
    EXPECT_EQ(AscendC::Std::get<1>(strideMakeInt).value, 555);
    EXPECT_EQ(AscendC::Std::get<2>(strideMakeInt).value, 666);
}

TEST_F(Tensor_Api_Layout, IsTupleOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shapeInit{11, 22, 33};
    Stride<int, int, int> strideInit{44, 55, 66};

    Shape<int, int, int> shapeMake = MakeShape(10, 20, 30);
    Stride<int, int, int> strideMake = MakeStride(40, 50, 60);

    auto shapeMakeInt = MakeShape(Int<111>{}, Int<222>{}, Int<333>{});
    auto strideMakeInt = MakeStride(Int<444>{}, Int<555>{}, Int<666>{});

    EXPECT_EQ(AscendC::Std::is_tuple<decltype(shapeInit)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(strideInit)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(shapeMake)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(strideMake)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(shapeMakeInt)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(strideMakeInt)>::value, true);

    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shapeInit)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(strideInit)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shapeMake)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(strideMake)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shapeMakeInt)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(strideMakeInt)>, true);
}

TEST_F(Tensor_Api_Layout, LayoutOperation)
{
    using namespace AscendC::Te;

    Shape<int,int,int> shape = MakeShape(10, 20, 30);
    Stride<int,int,int> stride = MakeStride(1, 100, 200);

    auto layoutMake = MakeLayout(shape, stride);

    Layout<Shape<int, int, int>, Stride<int, int, int>> layoutInit(shape, stride);

    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.Shape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(layoutMake.Shape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(layoutMake.Shape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.Stride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(layoutMake.Stride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(layoutMake.Stride()), 200);


    EXPECT_EQ(AscendC::Std::get<0>(layoutInit.Shape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(layoutInit.Shape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(layoutInit.Shape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(layoutInit.Stride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(layoutInit.Stride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(layoutInit.Stride()), 200);

    EXPECT_EQ(layoutMake.Rank(), 3);
    EXPECT_EQ(layoutMake.Rank<0>(), 1);
    EXPECT_EQ(Rank(layoutMake), 3);

    auto shapeTuple = GetShape(Select<1,2>(layoutMake));
    EXPECT_EQ(AscendC::Std::get<0>(shapeTuple), 20);
    EXPECT_EQ(Size(layoutMake), 6000);
    EXPECT_EQ(layoutMake.Size(), 6000);
    EXPECT_EQ(layoutMake.Size(), 6000);
    EXPECT_EQ(Coshape(layoutMake), 7710);
    EXPECT_EQ(Cosize(layoutMake), 7710);
}

TEST_F(Tensor_Api_Layout, IsLayoutOperation)
{
    using namespace AscendC::Te;

    Shape<int,int,int> shape = MakeShape(10, 20, 30);
    Stride<int,int,int> stride = MakeStride(1, 100, 200);

    auto layoutMake = MakeLayout(shape, stride);

    Layout<Shape<int, int, int>, Stride<int, int, int>> layoutInit(shape, stride);

    EXPECT_EQ(is_layout<decltype(shape)>::value, false);
    EXPECT_EQ(is_layout<decltype(stride)>::value, false);
    EXPECT_EQ(is_layout<decltype(layoutMake)>::value, true);
    EXPECT_EQ(is_layout<decltype(layoutInit)>::value, true);

    EXPECT_EQ(is_layout_v<decltype(shape)>, false);
    EXPECT_EQ(is_layout_v<decltype(stride)>, false);
    EXPECT_EQ(is_layout_v<decltype(layoutMake)>, true);
    EXPECT_EQ(is_layout_v<decltype(layoutInit)>, true);
}

TEST_F(Tensor_Api_Layout, MakeLayoutByShapeOperation)
{
    using namespace AscendC::Te;

    Shape<int,int,int> shape = MakeShape(2, 3, 4);
    Stride<int,int,int> stride = MakeStride(1, 2, 6);

    auto layoutMake1 = MakeLayout(shape, stride);
    auto layoutMake2 = MakeLayout(shape);

    EXPECT_EQ(AscendC::Std::get<1>(layoutMake1.Stride()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(layoutMake2.Stride()), 4);
    EXPECT_EQ(Size(layoutMake1), 24);
    EXPECT_EQ(Size(layoutMake2), 24);
    EXPECT_EQ(layoutMake1.Size(), 24);
    EXPECT_EQ(layoutMake2.Size(), 24);
    EXPECT_EQ(layoutMake2.Capacity(), 24);
}

TEST_F(Tensor_Api_Layout, LayoutSizeOperation)
{
    using namespace AscendC::Te;

    using shape = Shape<Int<16>, Int<16>>;
    using stride = Stride<Int<1>, Int<16>>;
    Layout<shape, stride> layoutMake;
    Tile<int, int> tile = MakeTile(1,2);
    Coord<int, int> coord = MakeCoord(1,2);

    EXPECT_EQ(layoutMake.size, 256);
    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.Shape()), 16);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape(layoutMake)), 16);
    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.Stride()), 1);
    EXPECT_EQ(AscendC::Std::get<0>(GetStride(layoutMake)), 1);
    EXPECT_EQ(layoutMake.Size(), 256);
}

TEST_F(Tensor_Api_Layout, StaticLayoutOperation)
{
    using namespace AscendC::Te;
    
    using TwoDimT = AscendC::Std::tuple<AscendC::Std::Int<3>, AscendC::Std::Int<4>>;
    using TwoDimU = AscendC::Std::tuple<AscendC::Std::Int<2>, AscendC::Std::Int<1>>;
    EXPECT_EQ((StaticLayoutSize<TwoDimT, TwoDimU>::size), 6);

    using FourDimT = AscendC::Std::tuple<AscendC::Std::tuple<AscendC::Std::Int<3>, AscendC::Std::Int<4>>,
                                        AscendC::Std::tuple<AscendC::Std::Int<1>, AscendC::Std::Int<2>>>;
    using FourDimU = AscendC::Std::tuple<AscendC::Std::tuple<AscendC::Std::Int<4>, AscendC::Std::Int<5>>,
                                        AscendC::Std::tuple<AscendC::Std::Int<2>, AscendC::Std::Int<3>>>;
    EXPECT_EQ((StaticLayoutSize<FourDimT, FourDimU>::size), 20);
}
