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
 * \file asc_struct.h
 * \brief public struct for cooperate framework
 */

#ifndef __INCLUDE_EXTERNAL_ASC_STRUCT_H__
#define __INCLUDE_EXTERNAL_ASC_STRUCT_H__

#include <vector>
#include <string>
#include <cstdint>

namespace AscPlugin {

enum class GenMode : uint32_t {
    AICORE_ONLY = 0,
    HOST_AICORE,
    HOST_SIMT,
    SIMT_ONLY
};

enum class PluginKernelType : uint8_t {
    MIX = 0,
    AIC = 1,
    AIV = 2,
};

enum class ParamType : uint32_t {
    NORMAL_INPUT = 0, // normal input arg
    TEMPLATE_TYPE = 1, // typename or class
    TEMPLATE_INT = 2, // num or enum
    TEMPLATE_DECL = 3, // const auto& ...
    TEMPLATE_TEMPLATE = 4, // template <typename, typename> class ...
};

struct PrologueConfig {
    bool saveTemp;
    bool verbose;
    GenMode genMode;
    std::string npuSoc;            // Example: Ascend910B1
    std::string npuArch;           // Example: dav-c220-cube
    std::string tmpPath;
    std::string logPath;
    std::string source;            // user input source filename
    std::string binaryPtrName;
    std::string binaryLenName;
    std::vector<std::string> compileArgs;
};

struct PrologueResult {
    std::string originPrefix;
    std::string deviceStubPrefix;
};

struct Param {
    Param() = default;
    Param(const char *type_, const char *name_, const bool &hasDefaultValue_, const char *defaultValue_,
        const char *attribute_)
        : type(type_), name(name_), hasDefaultValue(hasDefaultValue_), defaultValue(defaultValue_),
          attribute(attribute_)
    {}
    Param(const char *type_, const char *name_, const bool &hasDefaultValue_, const char *defaultValue_,
        const char *attribute_, const ParamType& typeClass_)
        : type(type_), name(name_), hasDefaultValue(hasDefaultValue_), defaultValue(defaultValue_),
          attribute(attribute_), typeClass(typeClass_)
    {}
    std::string type;
    std::string name;
    bool hasDefaultValue = false;
    std::string defaultValue;
    std::string attribute;
    ParamType typeClass = ParamType::NORMAL_INPUT;
};

struct CoreRatio {
    bool isCoreRatio;
    uint32_t cubeNum;
    uint32_t vecNum;
};

struct TemplateInstance {
    std::vector<std::string> templateInstantiationArguments;
    std::vector<Param> instanceKernelParameters;
    std::string instanceMangledName;
    std::string instanceMangledNameConsiderPrefix;
    CoreRatio ratio;
};

struct KernelInfo {
    std::string kernelName;
    std::string kernelMangledName;
    std::string kernelMangledNameConsiderPrefix;
    std::string fileName;
    uint32_t lineNum;
    uint32_t colNum;
    std::vector<std::string> namespaces;
    std::vector<Param> kernelParameters;
    std::vector<std::string> kernelAttributes;
    CoreRatio ratio;
    bool isTemplate = false;
    std::vector<Param> templateParameters;
    std::vector<TemplateInstance> templateInstances;
};

struct GenKernelResult {
    std::string hostStub;
    std::string deviceStub;
    std::string metaInfo;
    PluginKernelType type;
};

struct PreCompileOptsResult {
    std::vector<std::string> compileOptions;
};

struct EpilogueResult {
    std::string functionRegisterCode;
    std::vector<std::string> hostExtraCompileOptions;
    std::vector<std::string> deviceCubeExtraCompileOptions;
    std::vector<std::string> deviceVecExtraCompileOptions;
};

struct FatbinLinkResult {
    std::vector<std::string> extraFatbinHostLinkOptions;
    std::string binaryRegisterCode;
};

} // namespace AscPlugin
#endif // __INCLUDE_EXTERNAL_ASC_STRUCT_H__