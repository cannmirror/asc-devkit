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

class IsConvertibleTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsConvertibleTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsConvertibleTest TearDownTestCase" << std::endl;
    }
};

// 测试基本类型的可转换性
TEST_F(IsConvertibleTest, BasicTypes) {
    // int 可以隐式转换为 double
    EXPECT_TRUE((AscendC::Std::is_convertible<int, double>::value));
    // double 可以隐式转换为 int（会有精度损失，此处测试隐式转换）
    EXPECT_TRUE((AscendC::Std::is_convertible<double, int>::value));
    // char 可以隐式转换为 int
    EXPECT_TRUE((AscendC::Std::is_convertible<char, int>::value));
    // bool 可以隐式转换为 int
    EXPECT_TRUE((AscendC::Std::is_convertible<bool, int>::value));
}

// 测试自定义类的可转换性
class Base {};
class Derived : public Base {};

TEST_F(IsConvertibleTest, CustomClasses) {
    // Derived 可以隐式转换为 Base
    EXPECT_TRUE((AscendC::Std::is_convertible<Derived, Base>::value));
    // Base 不能隐式转换为 Derived
    EXPECT_FALSE((AscendC::Std::is_convertible<Base, Derived>::value));
}

// 测试带有转换运算符的类
class ConvertibleToInt {
public:
    operator int() const { return 0; }
};

TEST_F(IsConvertibleTest, ClassWithConversionOperator) {
    // ConvertibleToInt 可以隐式转换为 int
    EXPECT_TRUE((AscendC::Std::is_convertible<ConvertibleToInt, int>::value));
}

// 测试带有接受其他类型构造函数的类
class ConvertibleFromInt {
public:
    ConvertibleFromInt(int) {}
};

TEST_F(IsConvertibleTest, ClassWithConstructor) {
    // int 可以隐式转换为 ConvertibleFromInt
    EXPECT_TRUE((AscendC::Std::is_convertible<int, ConvertibleFromInt>::value));
}

// 测试数组和指针的可转换性
TEST_F(IsConvertibleTest, ArraysAndPointers) {
    // 数组可以隐式转换为指向其首元素的指针
    EXPECT_TRUE((AscendC::Std::is_convertible<int[5], int*>::value));
    // 指向基类的指针可以隐式转换为指向 void 的指针
    EXPECT_TRUE((AscendC::Std::is_convertible<Base*, void*>::value));
}

// 测试函数指针的可转换性
void func() {}
void (*funcPtr)() = func;

TEST_F(IsConvertibleTest, FunctionPointers) {
    // 函数指针类型相同，可以转换
    EXPECT_TRUE((AscendC::Std::is_convertible<decltype(funcPtr), void (*)()>::value));
}
