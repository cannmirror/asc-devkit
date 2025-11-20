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
 * \file ascc_match_global_info.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_MATCH_GLOBAL_INFO_H__
#define __INCLUDE_ASCC_MATCH_GLOBAL_INFO_H__
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "ascc_info_function.h"
#include "ascc_info_callexpr.h"
#include "ascc_log.h"

namespace Ascc {
class AsccMatchGlobalInfo {
public:
    inline static AsccMatchGlobalInfo &GetInstance()
    {
        static AsccMatchGlobalInfo instance;
        return instance;
    }

    void AddGlobalKernelCallExpr(const std::string& file, const AsccInfoCallExpr& callExpr)
    {
        globalCallExprMap_[file] = callExpr;
        fileToCallLineColumn_[callExpr.file].emplace(std::to_string(callExpr.line) + ":" +
            std::to_string(callExpr.column));
        manglingNameToCalled_.emplace(callExpr.manglingName);
    }

    std::unordered_map<std::string, AsccInfoCallExpr>& GetGlobalKernelCallExpr()
    {
        return globalCallExprMap_;
    }

    std::unordered_map<std::string, std::unordered_set<std::string>>& GetGlobalKernelCallLineColumn()
    {
        return fileToCallLineColumn_;
    }

    bool IsCalled(const std::string& manglingName) const
    {
        return manglingNameToCalled_.count(manglingName) > 0;
    }

    bool HasKernelCall() const
    {
        return manglingNameToCalled_.size() > 0;
    }

private:
    AsccMatchGlobalInfo() = default;
    ~AsccMatchGlobalInfo() = default;
    AsccMatchGlobalInfo(const AsccMatchGlobalInfo&) = delete;
    AsccMatchGlobalInfo& operator=(const AsccMatchGlobalInfo&) = delete;
    std::unordered_map<std::string, AsccInfoCallExpr> globalCallExprMap_; // key: <file>:<line>:<col>, value: callExpr
    std::unordered_map<std::string, std::unordered_set<std::string>> fileToCallLineColumn_;
    std::unordered_set<std::string> manglingNameToCalled_;
};
} // Ascc
#endif