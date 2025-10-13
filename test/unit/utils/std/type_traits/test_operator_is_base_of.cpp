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

class IsBaseOfTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsBaseOfTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsBaseOfTest TearDownTestCase" << std::endl;
    }
};

// 基础基类和派生类定义
class Base1 {};
class Derived1 : public Base1 {};

class Base2 {};
class Derived2 : public Base2 {};

// 多级继承类
class GrandBase {};
class Intermediate : public GrandBase {};
class DeepDerived : public Intermediate {};

// 虚继承相关类
class VirtualBase {};
class VirtualDerived : virtual public VirtualBase {};

// 私有继承类
class PrivateBase {};
class PrivateDerived : private PrivateBase {};

// 测试标准的基类和派生类关系
TEST_F(IsBaseOfTest, StandardBaseDerived) {
    EXPECT_TRUE((AscendC::Std::is_base_of<Base1, Derived1>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<Base1, Derived1>));
}

// 测试不同基类和派生类对
TEST_F(IsBaseOfTest, DifferentBaseDerivedPair) {
    EXPECT_TRUE((AscendC::Std::is_base_of<Base2, Derived2>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<Base2, Derived2>));
}

// 测试类和自身的基类关系
TEST_F(IsBaseOfTest, ClassIsSelfBase) {
    EXPECT_TRUE((AscendC::Std::is_base_of<Base1, Base1>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<Base1, Base1>));
}

// 测试不相关类之间的基类关系
TEST_F(IsBaseOfTest, UnrelatedClasses) {
    EXPECT_FALSE((AscendC::Std::is_base_of<Base1, Base2>::value));
    EXPECT_FALSE((AscendC::Std::is_base_of_v<Base1, Base2>));
}

// 测试派生类和基类的反向关系
TEST_F(IsBaseOfTest, DerivedToBaseReverse) {
    EXPECT_FALSE((AscendC::Std::is_base_of<Derived1, Base1>::value));
    EXPECT_FALSE((AscendC::Std::is_base_of_v<Derived1, Base1>));
}

// 测试多级继承关系
TEST_F(IsBaseOfTest, MultiLevelInheritance) {
    EXPECT_TRUE((AscendC::Std::is_base_of<GrandBase, DeepDerived>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<GrandBase, DeepDerived>));
    EXPECT_TRUE((AscendC::Std::is_base_of<Intermediate, DeepDerived>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<Intermediate, DeepDerived>));
}

// 测试多级继承反向关系
TEST_F(IsBaseOfTest, MultiLevelInheritanceReverse) {
    EXPECT_FALSE((AscendC::Std::is_base_of<DeepDerived, GrandBase>::value));
    EXPECT_FALSE((AscendC::Std::is_base_of_v<DeepDerived, GrandBase>));
}

// 测试虚继承关系
TEST_F(IsBaseOfTest, VirtualInheritance) {
    EXPECT_TRUE((AscendC::Std::is_base_of<VirtualBase, VirtualDerived>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<VirtualBase, VirtualDerived>));
}

// 测试私有继承关系（AscendC::Std::is_base_of 不考虑访问权限）
TEST_F(IsBaseOfTest, PrivateInheritance) {
    EXPECT_TRUE((AscendC::Std::is_base_of<PrivateBase, PrivateDerived>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<PrivateBase, PrivateDerived>));
}

// 测试模板类作为基类和派生类
template <typename T>
class TemplateBase {};

template <typename T>
class TemplateDerived : public TemplateBase<T> {};

TEST_F(IsBaseOfTest, TemplateClasses) {
    EXPECT_TRUE((AscendC::Std::is_base_of<TemplateBase<int>, TemplateDerived<int>>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<TemplateBase<int>, TemplateDerived<int>>));
}

// 测试模板类不同类型参数情况
TEST_F(IsBaseOfTest, TemplateClassesDifferentParams) {
    EXPECT_FALSE((AscendC::Std::is_base_of<TemplateBase<int>, TemplateDerived<double>>::value));
    EXPECT_FALSE((AscendC::Std::is_base_of_v<TemplateBase<int>, TemplateDerived<double>>));
}

