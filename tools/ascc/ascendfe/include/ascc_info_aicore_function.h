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
 * \file ascc_info_aicore_function.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_INFO_AICORE_FUNCTION_H__
#define __INCLUDE_ASCC_INFO_AICORE_FUNCTION_H__
#include <unordered_map>
#include <vector>
#include <utility>
#include <string>
#include <unordered_set>

#include "ascc_common_types.h"

namespace Ascc {
class AsccInfoAicoreFunc {
public:
    inline static AsccInfoAicoreFunc &GetInstance()
    {
        static AsccInfoAicoreFunc instance;
        return instance;
    }

    void StoreKernelDefScope(const std::string &key, const std::vector<std::pair<uint32_t, uint32_t>> &value);

    void StoreKernelCallLineCode(
        const std::string &filePath, const std::vector<std::pair<uint32_t, std::string>> &kernelCalls);

    void StorekernelDefLineCode(const std::string &filePath, uint32_t lineNumber, const std::string &lineInfo);

    AsccStatus GetKernelDefScope(const std::string &key, std::vector<std::pair<uint32_t, uint32_t>> &outValue) const;

    AsccStatus GetKernelCallLineCode(
        const std::string &filePath, std::vector<std::pair<uint32_t, std::string>> &outLines) const;

    AsccStatus GetkernelDefLineCode(
        const std::string &filePath, std::vector<std::pair<uint32_t, std::string>> &outLines) const;

    std::unordered_set<std::string> GetAllKeys() const;

private:
    AsccInfoAicoreFunc() = default;
    ~AsccInfoAicoreFunc() = default;
    AsccInfoAicoreFunc(const AsccInfoAicoreFunc&) = delete;
    AsccInfoAicoreFunc& operator=(const AsccInfoAicoreFunc&) = delete;
    AsccInfoAicoreFunc(AsccInfoAicoreFunc&&) = delete;
    AsccInfoAicoreFunc& operator=(AsccInfoAicoreFunc&&) = delete;

    std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t>>> kernelDefScopeData_;
    std::unordered_map<std::string, std::vector<std::pair<uint32_t, std::string>>> kernelCallData_;
    std::unordered_map<std::string, std::vector<std::pair<uint32_t, std::string>>> kernelDefData_;
};
}
#endif // __INCLUDE_ASCC_INFO_AICORE_FUNCTION_H__