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
 * \file ascc_info_aicore_function.cpp
 * \brief
 */

#include "ascc_info_aicore_function.h"

#include "ascc_log.h"

namespace Ascc {

void AsccInfoAicoreFunc::StoreKernelDefScope(
    const std::string &key, const std::vector<std::pair<uint32_t, uint32_t>> &value)
{
    auto it = kernelDefScopeData_.find(key);
    if (it != kernelDefScopeData_.end()) {
        it->second.insert(it->second.end(), value.begin(), value.end());
    } else {
        kernelDefScopeData_[key] = value;
    }
}

void AsccInfoAicoreFunc::StoreKernelCallLineCode(
    const std::string &filePath, const std::vector<std::pair<uint32_t, std::string>> &kernelCalls)
{
    for (const auto &codeLine : kernelCalls) {
        kernelCallData_[filePath].emplace_back(codeLine);
    }
}

void AsccInfoAicoreFunc::StorekernelDefLineCode(
    const std::string &filePath, uint32_t lineNumber, const std::string &lineInfo)
{
    ASC_LOG_ASC_DEBUG(PREPROCESS, "Store Kernel function definition: [%s].", lineInfo.c_str());
    kernelDefData_[filePath].emplace_back(lineNumber, lineInfo);
}

AsccStatus AsccInfoAicoreFunc::GetKernelDefScope(
    const std::string &key, std::vector<std::pair<uint32_t, uint32_t>> &outValue) const
{
    auto it = kernelDefScopeData_.find(key);
    if (it != kernelDefScopeData_.end()) {
        outValue = it->second;
        return AsccStatus::SUCCESS;
    }
    return AsccStatus::FAILURE;
}

AsccStatus AsccInfoAicoreFunc::GetKernelCallLineCode(
    const std::string &filePath, std::vector<std::pair<uint32_t, std::string>> &outLines) const
{
    auto it = kernelCallData_.find(filePath);
    if (it != kernelCallData_.end()) {
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Get kernel function call: [%s].", it->second.front().second.c_str());
        outLines = it->second;
        return AsccStatus::SUCCESS;
    }
    ASC_LOG_ASC_DEBUG(PREPROCESS, "Get kernel function call failed.");
    return AsccStatus::FAILURE;
}

AsccStatus AsccInfoAicoreFunc::GetkernelDefLineCode(
    const std::string &filePath, std::vector<std::pair<uint32_t, std::string>> &outLines) const
{
    auto it = kernelDefData_.find(filePath);
    if (it != kernelDefData_.end()) {
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Get kernel function definition: [%s].", it->second.front().second.c_str());
        outLines = it->second;
        return AsccStatus::SUCCESS;
    }
    ASC_LOG_ASC_DEBUG(PREPROCESS, "Get kernel function definition failed.");
    return AsccStatus::FAILURE;
}

std::unordered_set<std::string> AsccInfoAicoreFunc::GetAllKeys() const
{
    std::unordered_set<std::string> keys;
    for (const auto &pair : kernelDefScopeData_) {
        keys.insert(pair.first);
    }
    for (const auto &pair : kernelCallData_) {
        keys.insert(pair.first);
    }
    for (const auto &pair : kernelDefData_) {
        keys.insert(pair.first);
    }
    return keys;
}
}  // namespace Ascc