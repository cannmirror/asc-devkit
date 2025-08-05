/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_COMMON_OPDEF_H
#define ATVC_COMMON_OPDEF_H

#include "type_list.h"
namespace ATVC {
enum class ParamType {
    INPUT,
    OUTPUT,
    TEMP,
};

template<ParamType paramType_, typename ... Ts>
struct ParamTypes {
    using types = ATVC::TypeList<Ts...>;
    static constexpr ParamType usage = paramType_;
};

template<typename ... Ts>
using OpInputs = ParamTypes<ParamType::INPUT, Ts...>;

template<typename ... Ts>
using OpOutputs = ParamTypes<ParamType::OUTPUT, Ts...>;

template<typename ... Ts>
using OpTemps = ParamTypes<ParamType::TEMP, Ts...>;


template<typename InTypeList, typename OutTypeList, typename TempTypeList=ATVC::OpTemps<>>
struct OpTraits {
    using In = InTypeList;
    using Out = OutTypeList;
    using Temp = TempTypeList;
};

template<typename TileCompute>
struct GetFunctionTraits {};

// 专门化处理函数A的模板实例
template<template<typename> class TileCompute, typename Traits>
struct GetFunctionTraits<TileCompute<Traits>> {
    using ComputeTraits = Traits;
};
}
#endif

