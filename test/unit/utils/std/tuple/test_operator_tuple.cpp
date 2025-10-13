/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "kernel_operator.h"

class TupleTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "TupleTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "TupleTest TearDownTestCase" << std::endl;
    }
};

TEST_F(TupleTest, InitTupleValue)
{
    // 1.1 tuple定义
    AscendC::Std::tuple<uint32_t, float, bool> testTuple;
    EXPECT_EQ(sizeof(testTuple), 12);

    // 1.2 tuple定义并初始化
    AscendC::Std::tuple<uint32_t, float, bool> testTupleInit{11, 2.2, true};
    EXPECT_EQ(sizeof(testTupleInit), 12);

    // 1.3 tuple定义,类型为Tensor
    AscendC::Std::tuple<AscendC::LocalTensor<half>, AscendC::GlobalTensor<float>> testTupleInitTensor;
    EXPECT_EQ(sizeof(testTupleInitTensor), sizeof(AscendC::GlobalTensor<float>) + sizeof(AscendC::LocalTensor<half>));

    // 1.4 tuple定义,类型为Tensor，并初始化
    AscendC::LocalTensor<float> srcLocal;
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::Std::tuple<AscendC::LocalTensor<float>, AscendC::GlobalTensor<half>> testTupleInitTensorInit{srcLocal, srcGlobal};
    EXPECT_EQ(sizeof(testTupleInitTensorInit), sizeof(AscendC::LocalTensor<float>) + sizeof(AscendC::GlobalTensor<half>));

    // 1.5 多种类型定义
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double> testMultiType;
    EXPECT_EQ(sizeof(testMultiType), 56);

    // 1.6 多种类型初始化
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double> testMultiTypeInit{
        1, 2, 3, 4.4, 5, 6, 7, 8, 9, 10, 11.11, 12.12, 13.13 };
    EXPECT_EQ(sizeof(testMultiTypeInit), 56);

    // 1.7 多种复合类型定义
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double, \
    AscendC::LocalTensor<AscendC::int4b_t>, AscendC::GlobalTensor<bfloat16_t>> testMultiTensorType;
    EXPECT_EQ(sizeof(testMultiTensorType), 55 + sizeof(AscendC::int4b_t) + sizeof(AscendC::LocalTensor<AscendC::int4b_t>) + sizeof(AscendC::GlobalTensor<bfloat16_t>));

    // 1.7 64个变量聚合
    AscendC::Std::tuple<uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t,\
        uint32_t, float, bool, uint32_t> variableTuple64;
    EXPECT_EQ(sizeof(variableTuple64), 256);

    // 1.8 拷贝构造
    AscendC::Std::tuple<uint32_t, float, bool> testCopyTuple = testTupleInit;
    EXPECT_EQ(sizeof(testCopyTuple), 12);

    // 1.9 分步构造
    AscendC::Std::tuple<uint32_t, float, bool> testCopySplitTuple;
    testCopySplitTuple = testTupleInit;
    EXPECT_EQ(sizeof(testCopySplitTuple), 12);
    
    // 1.10 变量赋值
    uint32_t aaa = 11;
    float bbb = 2.2;
    bool ccc = true;
    AscendC::Std::tuple<uint32_t, float, bool> InitTupleUsingVariable{aaa, bbb, ccc};
    EXPECT_EQ(sizeof(InitTupleUsingVariable), 12);
}

TEST_F(TupleTest, TupleSize)
{
    // 2.1 获取tuple元素个数
    AscendC::Std::tuple<uint32_t, float, bool> testTuple;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testTuple)>::value, 3);

    // 2.2 获取tuple元素个数
    AscendC::Std::tuple<uint32_t, float, bool> testTupleInit{11, 2.2, true};
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testTupleInit)>::value, 3);

    // 2.3 获取tuple元素个数
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double, \
    AscendC::LocalTensor<AscendC::int4b_t>, AscendC::GlobalTensor<bfloat16_t>> testMultiTensorType;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testMultiTensorType)>::value, 15);

    // 2.4 获取tuple元素个数
    AscendC::Std::tuple<uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t,\
        uint32_t, float, bool, uint32_t> variableTuple64;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(variableTuple64)>::value, 64);

    // 2.5 获取tuple元素个数
     AscendC::Std::tuple<uint32_t, float, bool> testCopySplitTuple;
    testCopySplitTuple = testTupleInit;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testCopySplitTuple)>::value, 3);

    // 2.6 获取tuple元素个数 const
    const AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testConstTupleSize)>::value, 4);

    // 2.7 获取tuple元素个数 volatile
    volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testVolatileTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testVolatileTupleSize)>::value, 4);

    // 2.8 获取tuple元素个数 const volatile
    const volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstVolatileTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testConstVolatileTupleSize)>::value, 4);
}

TEST_F(TupleTest, TupleSizeV)
{
    // 3.1 获取tuple元素个数
    AscendC::Std::tuple<uint32_t, float, bool> testTuple;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testTuple)>, 3);

    // 3.2 获取tuple元素个数
    AscendC::Std::tuple<uint32_t, float, bool> testTupleInit{11, 2.2, true};
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testTupleInit)>, 3);

    // 3.3 获取tuple元素个数
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double, \
    AscendC::LocalTensor<AscendC::int4b_t>, AscendC::GlobalTensor<bfloat16_t>> testMultiTensorType;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testMultiTensorType)>, 15);

    // 3.4 获取tuple元素个数
    AscendC::Std::tuple<uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t,\
        uint32_t, float, bool, uint32_t> variableTuple64;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(variableTuple64)>, 64);

    // 3.5 获取tuple元素个数
     AscendC::Std::tuple<uint32_t, float, bool> testCopySplitTuple;
    testCopySplitTuple = testTupleInit;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testCopySplitTuple)>, 3);

    // 3.6 获取tuple元素个数 const
    const AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testConstTupleSize)>, 4);

    // 3.7 获取tuple元素个数 volatile
    volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testVolatileTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testVolatileTupleSize)>, 4);

    // 3.8 获取tuple元素个数 const volatile
    const volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstVolatileTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testConstVolatileTupleSize)>, 4);
}


TEST_F(TupleTest, TupleElement)
{
    // 4.1 获取第一个元素数据类型
    const AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstTuple;
    using FirstType = AscendC::Std::tuple_element<0, decltype(testConstTuple)>::type; // const uint32_t
    FirstType first = 88;
    EXPECT_EQ(first, 88);
    EXPECT_EQ(sizeof(FirstType), 4);

    // 4.2 获取第二个元素数据类型
    volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testVolatileTuple;
    using SecondType = AscendC::Std::tuple_element<1, decltype(testVolatileTuple)>::type; // volatile float
    SecondType second = 8.0;
    EXPECT_EQ(second, 8.0);
    EXPECT_EQ(sizeof(SecondType), 4);

    // 4.3 获取第三个元素数据类型
    const volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstVolatileTuple;
    using ThirdType = AscendC::Std::tuple_element<2, decltype(testConstVolatileTuple)>::type; // const volatile bool
    ThirdType third = false;
    EXPECT_EQ(third, false);
    EXPECT_EQ(sizeof(ThirdType), 1);

    // 4.4 获取第三个元素数据类型
    const AscendC::Std::tuple<const uint32_t, const volatile float, volatile uint16_t, const AscendC::LocalTensor<AscendC::int4b_t>> testConstElement;
    using ConstThirdType = AscendC::Std::tuple_element<2, decltype(testConstElement)>::type; // const volatile uint16_t
    ConstThirdType thirdConst = 22;
    EXPECT_EQ(thirdConst, 22);
    EXPECT_EQ(sizeof(ConstThirdType), 2);

    // 4.5 获取第二个元素数据类型
    using ConstSecondType = AscendC::Std::tuple_element<1, decltype(testConstElement)>::type; // const volatile float
    ConstSecondType secondConst = 8.0;
    EXPECT_EQ(secondConst, 8.0);
    EXPECT_EQ(sizeof(ConstSecondType), 4);

    // 4.6 获取第一个元素数据类型
    using ConstFirstType = AscendC::Std::tuple_element<0, decltype(testConstElement)>::type; // const uint32_t
    ConstFirstType firstConst = 88;
    EXPECT_EQ(firstConst, 88);
    EXPECT_EQ(sizeof(ConstFirstType), 4);
}

TEST_F(TupleTest, MakeTuple)
{
    // 指定数据类型聚合
    auto makeTupleEle = AscendC::Std::make_tuple(55, (float)6.6, true);

    // 64个元素聚合
    auto makeTuple64Ele = AscendC::Std::make_tuple( \
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, \
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, \
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, \
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, \
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, \
    61, 62, 63, 64);

    // Tensor类聚合
    AscendC::LocalTensor<float> xLocal;
    AscendC::LocalTensor<AscendC::int4b_t> yLocal;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    auto makeTupleTensor = AscendC::Std::make_tuple(xLocal, yLocal, xGm, yGm);

    // make_tuple作为初始化
    AscendC::Std::tuple<uint16_t, float> makeTupleInit (AscendC::Std::make_tuple((uint16_t)33, (float)4.4));

    // make_tuple作为拷贝构造初始化
    const AscendC::Std::tuple<uint32_t, float, bool> makeTupleCopyInit = AscendC::Std::make_tuple((uint32_t)55, (float)6.6, true);
}

// 测试 AscendC::Std::tuple_size 的基本功能
TEST_F(TupleTest, BasicFunctionalityTupleSize) {
    using MyTuple = AscendC::Std::tuple<int, double, char, float>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size<MyTuple>::value;
    EXPECT_EQ(tuple_size, 4);
}


// 测试 AscendC::Std::tuple_size 对空元组的处理
TEST_F(TupleTest, EmptyTupleTupleSize) {
    using EmptyTuple = AscendC::Std::tuple<>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size<EmptyTuple>::value;
    EXPECT_EQ(tuple_size, 0);
}


// 测试 AscendC::Std::tuple_size 对 const 元组的处理
TEST_F(TupleTest, ConstTupleTupleSize) {
    using ConstTuple = const AscendC::Std::tuple<int, double, char, float>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size<ConstTuple>::value;
    EXPECT_EQ(tuple_size, 4);
}


// 测试 AscendC::Std::tuple_size 对引用元组的处理
TEST_F(TupleTest, TupleWithReferencesTupleSize) {
    using TupleWithRef = AscendC::Std::tuple<int&, double&, char&, float&>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size<TupleWithRef>::value;
    EXPECT_EQ(tuple_size, 4);
}

// 测试 AscendC::Std::tuple_size 对不同元组大小的处理
TEST_F(TupleTest, DifferentTupleSizesTupleSize) {
    using Tuple1 = AscendC::Std::tuple<int>;
    using Tuple2 = AscendC::Std::tuple<int, double>;
    using Tuple3 = AscendC::Std::tuple<int, double, char>;
    using Tuple4 = AscendC::Std::tuple<int, double, char, float>;

    constexpr size_t size1 = AscendC::Std::tuple_size<Tuple1>::value;
    constexpr size_t size2 = AscendC::Std::tuple_size<Tuple2>::value;
    constexpr size_t size3 = AscendC::Std::tuple_size<Tuple3>::value;
    constexpr size_t size4 = AscendC::Std::tuple_size<Tuple4>::value;

    EXPECT_EQ(size1, 1);
    EXPECT_EQ(size2, 2);
    EXPECT_EQ(size3, 3);
    EXPECT_EQ(size4, 4);
}

// 测试 AscendC::Std::tuple_size_v 的基本功能
TEST_F(TupleTest, BasicFunctionalityTupleSizeV) {
    using MyTuple = AscendC::Std::tuple<int, double, char>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size_v<MyTuple>;
    EXPECT_EQ(tuple_size, 3);
}


// 测试 AscendC::Std::tuple_size_v 对空元组的处理
TEST_F(TupleTest, EmptyTupleTupleSizeV) {
    using EmptyTuple = AscendC::Std::tuple<>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size_v<EmptyTuple>;
    EXPECT_EQ(tuple_size, 0);
}


// 测试 AscendC::Std::tuple_size_v 对 const 元组的处理
TEST_F(TupleTest, ConstTupleTupleSizeV) {
    using ConstTuple = const AscendC::Std::tuple<int, double, char>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size_v<ConstTuple>;
    EXPECT_EQ(tuple_size, 3);
}


// 测试 AscendC::Std::tuple_size_v 对引用元组的处理
TEST_F(TupleTest, TupleWithReferencesTupleSizeV) {
    using TupleWithRef = AscendC::Std::tuple<int&, double&, char&>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size_v<TupleWithRef>;
    EXPECT_EQ(tuple_size, 3);
}

// 测试 AscendC::Std::tuple_size_v 对不同元组大小的处理
TEST_F(TupleTest, DifferentTupleSizesTupleSizeV) {
    using Tuple1 = AscendC::Std::tuple<int>;
    using Tuple2 = AscendC::Std::tuple<int, double>;
    using Tuple3 = AscendC::Std::tuple<int, double, char>;
    using Tuple4 = AscendC::Std::tuple<int, double, char, float>;

    constexpr size_t size1 = AscendC::Std::tuple_size_v<Tuple1>;
    constexpr size_t size2 = AscendC::Std::tuple_size_v<Tuple2>;
    constexpr size_t size3 = AscendC::Std::tuple_size_v<Tuple3>;
    constexpr size_t size4 = AscendC::Std::tuple_size_v<Tuple4>;

    EXPECT_EQ(size1, 1);
    EXPECT_EQ(size2, 2);
    EXPECT_EQ(size3, 3);
    EXPECT_EQ(size4, 4);
}

// 测试 make_tuple 的基本功能
TEST_F(TupleTest, BasicFunctionalityMakeTuple) {
    auto t1 = AscendC::Std::make_tuple(1, 2.0, 'h');
    EXPECT_EQ(AscendC::Std::get<0>(t1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(t1), 2.0);
    EXPECT_EQ(AscendC::Std::get<2>(t1), 'h');
}

// 测试 make_tuple 创建空元组
TEST_F(TupleTest, EmptyTupleMakeTuple) {
    auto t1 = AscendC::Std::make_tuple();
    // 检查元组的大小是否为 0
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(t1)>::value, 0);
}

// 测试 make_tuple 与不同类型的组合
TEST_F(TupleTest, DifferentTypes) {
    auto t1 = AscendC::Std::make_tuple(1, 2.0, 'a');
    EXPECT_EQ(AscendC::Std::get<0>(t1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(t1), 2.0);
    EXPECT_EQ(AscendC::Std::get<2>(t1), 'a');
}

// 测试 make_tuple 对引用的处理
TEST_F(TupleTest, TupleWithReferencesMakeTuple) {
    int a = 12;
    double b = 21.5;
    auto t1 = AscendC::Std::make_tuple(a, b);
    AscendC::Std::get<0>(t1)++;
    AscendC::Std::get<1>(t1) += 1.5;
    EXPECT_EQ(a, 12);
    EXPECT_EQ(b, 21.5);
}


// 测试 make_tuple 对 const 元素的处理
TEST_F(TupleTest, TupleWithConstElements) {
    const int a = 10;
    const double b = 20.0;
    const char s = 't';
    auto t1 = AscendC::Std::make_tuple(a, b, s);
    EXPECT_EQ(AscendC::Std::get<0>(t1), 10);
    EXPECT_EQ(AscendC::Std::get<1>(t1), 20.0);
    EXPECT_EQ(AscendC::Std::get<2>(t1), 't');
}

// 测试 AscendC::Std::tuple_element 的基本功能
TEST_F(TupleTest, BasicFunctionalityTupleElement) {
    using MyTuple = AscendC::Std::tuple<int, double>;
    using FirstType = AscendC::Std::tuple_element<0, MyTuple>::type;
    using SecondType = AscendC::Std::tuple_element<1, MyTuple>::type;

    // 检查 AscendC::Std::tuple_element 获取的元素类型是否正确
    EXPECT_TRUE((std::is_same_v<FirstType, int>));
    EXPECT_TRUE((std::is_same_v<SecondType, double>));
}

// 测试 AscendC::Std::tuple_element 对 const 元组的处理
TEST_F(TupleTest, ConstTuple) {
    using ConstTuple = const AscendC::Std::tuple<int, double>;
    using FirstType = AscendC::Std::tuple_element<0, ConstTuple>::type;
    using SecondType = AscendC::Std::tuple_element<1, ConstTuple>::type;

    EXPECT_TRUE((std::is_same_v<FirstType, const int>));
    EXPECT_TRUE((std::is_same_v<SecondType, const double>));
}


// 测试 AscendC::Std::tuple_element 对引用元组的处理
TEST_F(TupleTest, TupleWithReferencesTupleElement) {
    using TupleWithRef = AscendC::Std::tuple<int&, double&>;
    using FirstType = AscendC::Std::tuple_element<0, TupleWithRef>::type;
    using SecondType = AscendC::Std::tuple_element<1, TupleWithRef>::type;

    EXPECT_TRUE((std::is_same_v<FirstType, int&>));
    EXPECT_TRUE((std::is_same_v<SecondType, double&>));
}


// 测试 AscendC::Std::tuple_element 对不同元组大小的处理
TEST_F(TupleTest, DifferentTupleSizes) {
    using Tuple1 = AscendC::Std::tuple<int, double, char>;
    using FirstType = AscendC::Std::tuple_element<0, Tuple1>::type;
    using SecondType = AscendC::Std::tuple_element<1, Tuple1>::type;
    using ThirdType = AscendC::Std::tuple_element<2, Tuple1>::type;

    EXPECT_TRUE((std::is_same_v<FirstType, int>));
    EXPECT_TRUE((std::is_same_v<SecondType, double>));
    EXPECT_TRUE((std::is_same_v<ThirdType, char>));
}

// 测试 AscendC::Std::tie 的基本功能
TEST_F(TupleTest, BasicFunctionalityTie) {
    int a = 10;
    double b = 20.5;
    char c = 'x';
    bool flag = true;

    auto tie_result = AscendC::Std::tie(a, b, c, flag);
    EXPECT_EQ(AscendC::Std::get<0>(tie_result), a);
    EXPECT_EQ(AscendC::Std::get<1>(tie_result), b);
    EXPECT_EQ(AscendC::Std::get<2>(tie_result), c);
    EXPECT_EQ(AscendC::Std::get<3>(tie_result), flag);
}


// 测试 AscendC::Std::tie 对变量引用的修改
TEST_F(TupleTest, ModifyThroughTieTie) {
    int a = 10;
    double b = 20.5;
    char c = 'x';
    bool flag = true;

    auto tie_result = AscendC::Std::tie(a, b, c, flag);
    AscendC::Std::get<0>(tie_result) = 15;
    AscendC::Std::get<1>(tie_result) = 25.5;
    AscendC::Std::get<2>(tie_result) = 'y';
    AscendC::Std::get<3>(tie_result) = false;

    EXPECT_EQ(a, 15);
    EXPECT_EQ(b, 25.5);
    EXPECT_EQ(c, 'y');
    EXPECT_EQ(flag, false);
}


// 测试 AscendC::Std::tie 与 const 变量
TEST_F(TupleTest, TieWithConstTie) {
    const int a = 10;
    const double b = 20.5;
    const char c = 'x';
    const bool flag = true;

    auto tie_result = AscendC::Std::tie(a, b, c, flag);
    // 尝试修改 const 变量，应该会导致编译错误
    // AscendC::Std::get<0>(tie_result) = 15;  // 取消注释会导致编译错误
    // AscendC::Std::get<1>(tie_result) = 25.5; // 取消注释会导致编译错误
    // AscendC::Std::get<2>(tie_result) = 'y'; // 取消注释会导致编译错误
    // AscendC::Std::get<3>(tie_result) = false; // 取消注释会导致编译错误
    EXPECT_EQ(AscendC::Std::get<0>(tie_result), a);
    EXPECT_EQ(AscendC::Std::get<1>(tie_result), b);
    EXPECT_EQ(AscendC::Std::get<2>(tie_result), c);
    EXPECT_EQ(AscendC::Std::get<3>(tie_result), flag);
}


// 测试 AscendC::Std::tie 对不同类型的处理
TEST_F(TupleTest, DifferentTypesTie) {
    int a = 10;
    float f = 1.23f;
    char c = 'z';
    bool flag = false;

    auto tie_result = AscendC::Std::tie(a, f, c, flag);
    EXPECT_EQ(AscendC::Std::get<0>(tie_result), a);
    EXPECT_EQ(AscendC::Std::get<1>(tie_result), f);
    EXPECT_EQ(AscendC::Std::get<2>(tie_result), c);
    EXPECT_EQ(AscendC::Std::get<3>(tie_result), flag);
}


// 测试 AscendC::Std::tie 对空绑定的处理
TEST_F(TupleTest, EmptyTieTie) {
    auto tie_result = AscendC::Std::tie();
    // 检查空绑定的元组大小是否为 0
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(tie_result)>::value, 0);
}

// // 测试 forward_as_tuple 的功能
TEST_F(TupleTest, BasicUsageForwardAsTuple) {
    // 使用 AscendC::Std::forward_as_tuple 对数据进行包装
    auto t1 = AscendC::Std::forward_as_tuple(1, 2.0);

    int& aaa = AscendC::Std::get<0>(t1);
    double& bbb = AscendC::Std::get<1>(t1);

    // 验证元组的大小是否为 2
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(t1)>::value, 2);
}

// 测试 forward_as_tuple 与转发引用的结合使用
TEST_F(TupleTest, ForwardingReferencesForwardAsTuple) {
    int x = 5;
    double y = 3.14;
    char c = 'a';

    // 使用 AscendC::Std::forward_as_tuple 对转发引用进行包装
    auto t2 = AscendC::Std::forward_as_tuple(AscendC::Std::forward<int>(x), AscendC::Std::forward<double>(y), AscendC::Std::forward<char>(c));

    // 验证元组的大小是否为 3
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(t2)>::value, 3);

    // 验证元组中的元素是否正确
    EXPECT_EQ(AscendC::Std::get<0>(t2), 5);
    EXPECT_EQ(AscendC::Std::get<1>(t2), 3.14);
    EXPECT_EQ(AscendC::Std::get<2>(t2), 'a');
}

// 测试 std::forward_as_tuple 在模板函数中的使用
template <typename... Args>
void processTupleRight(AscendC::Std::tuple<Args...>&& tuple) {
    EXPECT_EQ(AscendC::Std::get<0>(tuple), 100);
    EXPECT_EQ(AscendC::Std::get<1>(tuple), 3.14159);
}

TEST_F(TupleTest, TemplateFunctionUsage1) {
    // 传递右值作为元组
    processTupleRight(AscendC::Std::forward_as_tuple(100, 3.14159));
}

// 测试 std::forward_as_tuple 在模板函数中的使用
template <typename... Args>
void processTupleLeft(AscendC::Std::tuple<Args...>&& tuple) {
    EXPECT_EQ(AscendC::Std::get<0>(tuple), 5);
    EXPECT_EQ(AscendC::Std::get<1>(tuple), 2.71);
}

TEST_F(TupleTest, TemplateFunctionUsage2) {
    int a = 5;
    double b = 2.71;

    // 传递参数作为元组
    processTupleLeft(AscendC::Std::forward_as_tuple(a, b));
}