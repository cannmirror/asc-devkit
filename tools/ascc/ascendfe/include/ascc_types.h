/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file ascc_types.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_TYPES_H__
#define __INCLUDE_ASCC_TYPES_H__
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include "ascc_common_types.h"

namespace Ascc {
// 保存诊断信息初始buffer大小
constexpr uint32_t ASCC_CLANG_BUFFER_SIZE = 256;

enum class CompileType : uint32_t {
    OBJECT = 0,
    SHARED = 1,
    STATIC = 2,
    NONE = 3
};

class CompileArgs {
public:
    std::vector<std::string> definitions;
    std::vector<std::string> incPaths;
    std::vector<std::string> options;
    std::vector<std::string> incFiles;
    std::vector<std::string> linkFiles;
    std::vector<std::string> linkPath;
    std::string outputPath;
    std::string file;
    std::string customOption;

    std::string GetCmd(const std::string& compiler, bool isClang = true, CompileType type = CompileType::OBJECT,
        bool isAst = false) const;
    std::vector<std::string> GetCmdVector(const std::string &compiler, bool isClang = true,
        CompileType type = CompileType::OBJECT, bool isAst = false) const;
};

enum class AscCursorTypes : uint8_t {
    ASC_CURSOR_STRUCT = 0,
    ASC_CURSOR_CLASS = 1,
    ASC_CURSOR_UNION = 2,
    ASC_CURSOR_NORMAL_FUNCTION = 3,
    ASC_CURSOR_TEMPLATE_FUCNTION = 4,
    ASC_CURSOR_ATTRIBUTE = 5,
    ASC_CURSOR_CLASS_TEMPLATE = 6,
    ASC_CURSOR_FUNCTION_STRUCTS = 7,
    ASC_CURSOR_CALL_EXPR = 8,
    ASC_CURSOR_CXX_METHOD = 9,
    ASC_CURSOR_VAR_DECL = 10,
    INVALID_CURSOR_TYPE = 0xff
};

enum class KernelMode: uint32_t {
    MIX = 0,
    AIV = 1,
    AIC = 2,
    UNSUPPORTED = 3
};

enum class BuildType : uint32_t {
    DEBUG = 0,
    RELEASE = 1
};

enum class FuncMetaType : uint32_t {
    F_TYPE_KTYPE = 1,
    F_TYPE_CROSS_CORE_SYNC = 2,
    F_TYPE_MAX = 0xffffffff
};

enum class ShortSoCVersion : uint32_t {
    ASCEND910B = 0,
    ASCEND310P = 1,
    ASCEND910 = 2,
    ASCEND310B = 3,
    INVALID_TYPE = 0xffffffff
};

enum class ParamType : uint32_t {
    NORMAL_INPUT = 0,
    TEMPLATE_TYPE = 1,
    TEMPLATE_INT = 2,
    TEMPLATE_DECL = 3,
    TEMPLATE_TEMPLATE = 4,
    TEMPLATE_ENUM = 5
};

inline const std::unordered_map<Ascc::ShortSoCVersion, const char*> CCE_AICORE_MAP = {
    {Ascc::ShortSoCVersion::ASCEND910B, "__CCE_AICORE__=220"},
    {Ascc::ShortSoCVersion::ASCEND310P, "__CCE_AICORE__=200"},
    {Ascc::ShortSoCVersion::ASCEND910, "__CCE_AICORE__=100"},
    {Ascc::ShortSoCVersion::ASCEND310B, "__CCE_AICORE__=300"},
};

inline const std::unordered_map<ShortSoCVersion, const char*> NPU_ARCH_MAP = {
    {ShortSoCVersion::ASCEND910B, "__NPU_ARCH__=2201"},
    {ShortSoCVersion::ASCEND310P, "__NPU_ARCH__=2002"},
    {ShortSoCVersion::ASCEND910, "__NPU_ARCH__=1001"},
    {ShortSoCVersion::ASCEND310B, "__NPU_ARCH__=3002"},
};
} // Ascc
#endif // __INCLUDE_ASCC_TYPES_H__