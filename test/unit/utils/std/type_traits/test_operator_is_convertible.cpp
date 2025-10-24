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

// testing the convertibility of basic types
TEST_F(IsConvertibleTest, BasicTypes) {
    // int can be implicitly converted to double
    EXPECT_TRUE((AscendC::Std::is_convertible<int, double>::value));
    // double can be implicitly converted to int (loss of precision)
    EXPECT_TRUE((AscendC::Std::is_convertible<double, int>::value));
    // char can be implicitly converted to int
    EXPECT_TRUE((AscendC::Std::is_convertible<char, int>::value));
    // bool can be implicitly converted to int
    EXPECT_TRUE((AscendC::Std::is_convertible<bool, int>::value));
}

// testing the convertibility of custom class
class Base {};
class Derived : public Base {};

TEST_F(IsConvertibleTest, CustomClasses) {
    // Derived can be implicitly converted to Base
    EXPECT_TRUE((AscendC::Std::is_convertible<Derived, Base>::value));
    // Base can't be implicitly converted to Derived
    EXPECT_FALSE((AscendC::Std::is_convertible<Base, Derived>::value));
}

// test a class with a conversion opertor
class ConvertibleToInt {
public:
    operator int() const { return 0; }
};

TEST_F(IsConvertibleTest, ClassWithConversionOperator) {
    // ConvertibleToInt can be implicitly converted to int
    EXPECT_TRUE((AscendC::Std::is_convertible<ConvertibleToInt, int>::value));
}

// test a class that accepts other types of constructors
class ConvertibleFromInt {
public:
    ConvertibleFromInt(int) {}
};

TEST_F(IsConvertibleTest, ClassWithConstructor) {
    // int can be implicitly converted to ConvertibleFromInt
    EXPECT_TRUE((AscendC::Std::is_convertible<int, ConvertibleFromInt>::value));
}

// testing the convertibility of arrays and pointers
TEST_F(IsConvertibleTest, ArraysAndPointers) {
    // an array can be implicitly converted to a pointer to its first element
    EXPECT_TRUE((AscendC::Std::is_convertible<int[5], int*>::value));
    // a pointer to a base class can be implicitly converted to a pointer to void
    EXPECT_TRUE((AscendC::Std::is_convertible<Base*, void*>::value));
}

// testing the convertibility of function pointers
void func() {}
void (*funcPtr)() = func;

TEST_F(IsConvertibleTest, FunctionPointers) {
    // function pointers of the same type can be converted
    EXPECT_TRUE((AscendC::Std::is_convertible<decltype(funcPtr), void (*)()>::value));
}
