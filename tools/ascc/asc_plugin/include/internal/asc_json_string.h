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
 * \file asc_json_string.h
 * \brief
 */

#ifndef __INCLUDE_INTERNAL_ASC_JSON_STRING_H__
#define __INCLUDE_INTERNAL_ASC_JSON_STRING_H__

#include <string>
#include <nlohmann/json.hpp>
#include "asc_struct.h"

namespace AscPlugin {

// Note: to_json refers to nlohmann::adl_serializer::to_json
// returned result to bisheng
void to_json(nlohmann::json& jsonObj, const PreCompileOptsResult& result);
void to_json(nlohmann::json& jsonObj, const PrologueResult& result);
void to_json(nlohmann::json& jsonObj, const GenKernelResult& result);
void to_json(nlohmann::json& jsonObj, const EpilogueResult& result);
void to_json(nlohmann::json& jsonObj, const FatbinLinkResult& result);

// needed to convert jsonStr information to struct
void from_json(const nlohmann::json& jsonObj, PrologueConfig& config);
void from_json(const nlohmann::json& jsonObj, Param& param);
void from_json(const nlohmann::json& jsonObj, CoreRatio& ratio);
void from_json(const nlohmann::json& jsonObj, TemplateInstance& instance);
void from_json(const nlohmann::json& jsonObj, KernelInfo& info);

// json string of struct given by bisheng. Needs to check validity of arg value
int32_t FromJson(PrologueConfig& config, const char* jsonStr);
int32_t FromJson(KernelInfo& info, const char* jsonStr);

// copy data in struct to const char**
// must template explicit instantiation due to .so
template <typename T>
int32_t DumpResultInfo(T& inputStruct, const char** result);

// write asc result struct to file
void WriteFields(std::ofstream& outFile, const PrologueResult& inputStruct);
void WriteFields(std::ofstream& outFile, const GenKernelResult& inputStruct);
void WriteFields(std::ofstream& outFile, const PreCompileOptsResult& inputStruct);
void WriteFields(std::ofstream& outFile, const EpilogueResult& inputStruct);
void WriteFields(std::ofstream& outFile, const FatbinLinkResult& inputStruct);
} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_JSON_STRING_H__