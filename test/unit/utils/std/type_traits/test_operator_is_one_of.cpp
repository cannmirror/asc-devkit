/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "kernel_operator.h"

class IsOneOfTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsOneOfTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsOneOfTest TearDownTestCase" << std::endl;
    }
};


// 测试基本类型相同的情况
TEST_F(IsOneOfTest, BasicTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<int, int, double>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<int, int, double>));
    EXPECT_TRUE((AscendC::Std::is_one_of<double, double, int>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<double, double, int>));
}

// 测试基本类型不同的情况
TEST_F(IsOneOfTest, BasicTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<int, double, long>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<int, double, long>));
    EXPECT_FALSE((AscendC::Std::is_one_of<char, long, int>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<char, long, int>));
}

// 测试自定义类相同的情况
class MyClass1 {};
class MyClass3 {};
TEST_F(IsOneOfTest, CustomClassesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<MyClass1, MyClass1, MyClass3>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<MyClass1, MyClass1, MyClass3>));
}

// 测试自定义类不同的情况
class MyClass2 {};
TEST_F(IsOneOfTest, CustomClassesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<MyClass1, MyClass2, MyClass3>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<MyClass1, MyClass2, MyClass3>));
}

// 测试指针类型相同的情况
TEST_F(IsOneOfTest, PointerTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<int *, int *, double *>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<int *, int *, double *>));
    EXPECT_TRUE((AscendC::Std::is_one_of<MyClass1 *, MyClass1 *, MyClass2 *>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<MyClass1 *, MyClass1 *, MyClass2 *>));
}

// 测试指针类型不同的情况
TEST_F(IsOneOfTest, PointerTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<int *, double *, long *>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<int *, double *, long *>));
    EXPECT_FALSE((AscendC::Std::is_one_of<MyClass1 *, MyClass2 *, MyClass3 *>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<MyClass1 *, MyClass2 *, MyClass3 *>));
}

// 测试引用类型相同的情况
TEST_F(IsOneOfTest, ReferenceTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<int &, int &, double &>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<int &, int &, double &>));
    EXPECT_TRUE((AscendC::Std::is_one_of<MyClass1 &, MyClass1 &, MyClass2 &>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<MyClass1 &, MyClass1 &, MyClass2 &>));
}

// 测试引用类型不同的情况
TEST_F(IsOneOfTest, ReferenceTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<int &, double &, long &>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<int &, double &, long &>));
    EXPECT_FALSE((AscendC::Std::is_one_of<MyClass1 &, MyClass2 &, MyClass3 &>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<MyClass1 &, MyClass2 &, MyClass3 &>));
}

// 测试常量类型相同的情况
TEST_F(IsOneOfTest, ConstTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<const int, const int, const double>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<const int, const int, const double>));
    EXPECT_TRUE((AscendC::Std::is_one_of<const MyClass1, const MyClass1, const MyClass2>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<const MyClass1, const MyClass1, const MyClass2>));
}

// 测试常量类型与非常量类型不同的情况
TEST_F(IsOneOfTest, ConstAndNonConstDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<int, const int, const double>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<int, const int, const double>));
    EXPECT_FALSE((AscendC::Std::is_one_of<MyClass1, const MyClass1, const MyClass2>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<MyClass1, const MyClass1, const MyClass2>));
}

// 测试模板类型相同的情况
template <typename T> class TemplateClass {};

TEST_F(IsOneOfTest, TemplateTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<TemplateClass<int>, TemplateClass<int>, TemplateClass<double>>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<TemplateClass<int>, TemplateClass<int>, TemplateClass<double>>));
}

// 测试模板类型不同的情况
TEST_F(IsOneOfTest, TemplateTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<TemplateClass<int>, TemplateClass<double>, TemplateClass<long>>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<TemplateClass<int>, TemplateClass<double>, TemplateClass<long>>));
}
