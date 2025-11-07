/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ascc_info_function.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_INFO_FUNCTION_H__
#define __INCLUDE_ASCC_INFO_FUNCTION_H__
#include <vector>
#include <memory>
#include <map>

#include "ascc_info_base.h"
#include "ascc_types.h"

namespace Ascc {
class AsccInfoFunction : public AsccInfoBase {
public:
    AsccInfoFunction() = default;
    ~AsccInfoFunction() override = default;

    struct ParameterInfo {
        ParameterInfo(
            const std::string &paraName_, const std::string &type_, const bool &isPointer_, ParamType paramType_)
            : paraName(paraName_), type(type_), isPointer(isPointer_), paramType(paramType_)
        {}
        std::string paraName;
        std::string type;
        bool isPointer = false;
        ParamType paramType = ParamType::NORMAL_INPUT;
    };

    struct FunctionInfo {
        std::string funcName;
        std::string definitionPos;
        uint32_t lineNo = 0; // kernel name
        uint32_t startLineNo = 0; // template
        uint32_t endLineNo = 0; // end
        std::string nameSpace;
        std::string returnType;
        std::string manglingName;
        CodeMode kernelType = CodeMode::KERNEL_TYPE_MIX_AIC_1_2;
        std::vector<ParameterInfo> params;
        bool isTemplate = false;
        bool isTempInst = false;
        bool isTempExpSpec = false;
        bool hasKfcServer = false;
        std::vector<ParameterInfo> templateParams;
        std::unordered_map<std::string, std::shared_ptr<FunctionInfo>> mangledToInstFuncInfo;
        inline bool IsTempSpec() const
        {
            return isTemplate && !isTempInst && isTempExpSpec;
        }
        inline bool IsTempDecl() const
        {
            return isTemplate && !isTempInst && !isTempExpSpec;
        }
        inline bool IsTempInst() const
        {
            return isTemplate && isTempInst && isTempExpSpec;
        }
    };

    FunctionInfo *GetFunctionInfo(const std::string &key)
    {
        auto it = functions.find(key);
        return (it != functions.end()) ? &it->second : nullptr;
    }

    void AddFunction(const std::string& key, const FunctionInfo& func)
    {
        functions[key] = func;
    }

    bool IsEmpty() const override
    {
        return functions.empty();
    }

    using FunctionIterator = std::map<std::string, FunctionInfo>::iterator;
    FunctionIterator Begin() { return functions.begin(); }
    FunctionIterator End() { return functions.end(); }

private:
    std::map<std::string, FunctionInfo> functions;
};
} // Ascc
#endif