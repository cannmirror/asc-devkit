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

class EnableIfTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "EnableIfTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "EnableIfTest TearDownTestCase" << std::endl;
    }
};

template <typename T>
struct is_arithmetic {
    static constexpr bool value = AscendC::Std::is_integral<T>::value || AscendC::Std::is_floating_point<T>::value;
};

// 示例 1：根据类型是否为算术类型启用函数模板
// 仅当 T 是算术类型时，该函数模板才会被启用
template <typename T>
typename AscendC::Std::enable_if<is_arithmetic<T>::value, T>::type
multiply(T a, T b) {
    return a * b;
}

// 示例 2：根据类型大小启用不同的函数重载
// 当 T 的大小小于等于 4 字节时启用
template <typename T>
typename AscendC::Std::enable_if<(sizeof(T) <= 4), T>::type
getDefaultValue() {
    return T();
}

// 当 T 的大小大于 4 字节时启用
template <typename T>
typename AscendC::Std::enable_if<(sizeof(T) > 4), T>::type
getDefaultValue() {
    return T();
}

// 示例 3：根据模板参数是否为特定类型启用类模板
// 仅当 T 是 int 类型时，该类模板才会被启用
template <typename T, typename = typename AscendC::Std::enable_if<AscendC::Std::is_same<T, int>::value>::type>
class IntContainer {
public:
    IntContainer(T value) : data(value) {}
    T getData() const { return data; }
private:
    T data;
};

// 测试整数类型的乘法
TEST_F(EnableIfTest, ArithmeticIntegerMultiplication) {
    int result = multiply(3, 4);
    EXPECT_EQ(result, 12);
}

// 测试浮点数类型的乘法
TEST_F(EnableIfTest, ArithmeticFloatingPointMultiplication) {
    double result = multiply(2.5, 3.0);
    EXPECT_EQ(result, 7.5);
}

// 测试小尺寸类型的默认值获取
TEST_F(EnableIfTest, SizeSmallTypeDefaultValue) {
    int value = getDefaultValue<int>();
    EXPECT_EQ(value, 0);
}

// 测试大尺寸类型的默认值获取
TEST_F(EnableIfTest, SizeLargeTypeDefaultValue) {
    double value = getDefaultValue<double>();
    EXPECT_EQ(value, 0.0);
}

// 测试 IntContainer 类模板
TEST_F(EnableIfTest, SpecificTypeIntContainerFunctionality) {
    IntContainer<int> container(42);
    EXPECT_EQ(container.getData(), 42);
}
