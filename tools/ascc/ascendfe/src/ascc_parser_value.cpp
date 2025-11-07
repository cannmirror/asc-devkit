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
 * \file ascc_parser_value.cpp
 * \brief
 */

#include "ascc_parser_value.h"

#include <csignal>
#include <unordered_set>

#include "ascc_types.h"
#include "ascc_log.h"
#include "ascc_tmp_file_manager.h"
#include "ascc_global_env_manager.h"
#include "ascc_utils.h"

namespace Ascc {
void AsccParserValue::InitParser(const std::string &name, const std::string &description,
    const bool &isContinue, const bool &isSupportEqual)
{
    name_ = name;
    description_ = description;
    isContinue_ = isContinue;
    isSupportEqual_ = isSupportEqual;
}

AsccStatus AsccParserValue::Parse(const std::vector<std::string>& args, std::unordered_set<uint32_t>& parsedIndex)
{
    if (isContinue_) {
        return ParseContiOption(args, parsedIndex);
    }
    return ParseOption(args, parsedIndex);
}

std::vector<std::string> AsccParserValue::SplitByComma(const std::string& str) const
{
    const char gapChar = ',';
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = str.find(gapChar);
    std::string content = "";
    while (end != std::string::npos) {
        content = str.substr(start, end - start);
        result.emplace_back(content);
        start = end + 1;
        end = str.find(gapChar, start);
        ASC_LOG_ASC_INFO(OPTION, "Parse value option [%s] : [%s].", name_.c_str(), content.c_str());
    }
    content = str.substr(start);
    result.emplace_back(content);
    ASC_LOG_ASC_INFO(OPTION, "Parse value option [%s] : [%s].", name_.c_str(), content.c_str());
    return result;
}

AsccStatus AsccParserValue::ParseOption(
    const std::vector<std::string> &args, std::unordered_set<uint32_t> &parsedIndex)
{
    for (size_t i = 0; i < args.size(); ++i) {
        if (parsedIndex.find(i) != parsedIndex.end() || args[i].size() < name_.size()) {
            continue;
        }
        if (args[i] == name_) {
            if (name_ == "--save-temps") {
                parsedIndex.emplace(i);
                valid_ = true;
                ASC_LOG_ASC_INFO(OPTION, "Parse value option [%s].", name_.c_str());
                continue;
            }
            if (i + 1 < args.size() && parsedIndex.find(i + 1) == parsedIndex.end() && (*args[i + 1].begin()) != '-') {
                value_.emplace_back(args[i + 1]);
                parsedIndex.emplace(i);
                parsedIndex.emplace(i + 1);
                valid_ = true;
                ASC_LOG_ASC_INFO(OPTION, "Parse value option [%s] : [%s].", name_.c_str(), args[i + 1].c_str());
            } else {
                Ascc::HandleError(std::string("Set [" + name_ + "] error. Please check --help!"));
                return AsccStatus::FAILURE;
            }
            continue;
        }
        size_t equalIndex = args[i].find('=');
        if (isSupportEqual_ && equalIndex != std::string::npos &&
            args[i].substr(0, equalIndex) == name_) {
            if (equalIndex >= args[i].size() - 1) {
                Ascc::HandleError(std::string("[" + name_ + "] cannot be empty. Please check --help!"));
                return AsccStatus::FAILURE;
            }
            const std::string& content = args[i].substr(equalIndex + 1);
            size_t commaIndex = content.find(',');
            if (commaIndex == std::string::npos) {
                value_.emplace_back(content);
                parsedIndex.emplace(i);
                valid_ = true;
                ASC_LOG_ASC_INFO(OPTION, "Parse value option [%s] : [%s].", name_.c_str(), content.c_str());
            } else {
                const auto& inputList = SplitByComma(content);
                value_.insert(value_.end(), inputList.begin(), inputList.end());
                parsedIndex.emplace(i);
                valid_ = true;
            }
        }
    }
    return AsccStatus::SUCCESS;
}
AsccStatus AsccParserValue::ParseContiOption(
    const std::vector<std::string> &args, std::unordered_set<uint32_t> &parsedIndex)
{
    for (size_t i = 0; i < args.size(); ++i) {
        if (parsedIndex.find(i) != parsedIndex.end() || args[i].size() < name_.size()) {
            continue;
        }
        const std::string &opt = args[i].substr(0, name_.size());
        const std::string &content = args[i].substr(name_.size());
        if (opt == name_) {
            if (opt.size() == args[i].size()) {
                Ascc::HandleError(
                    std::string("Set [" + name_ + "] error. Please check --help!"));
                return AsccStatus::FAILURE;
            }
            value_.emplace_back(content);
            parsedIndex.emplace(i);
            valid_ = true;
            ASC_LOG_ASC_INFO(OPTION, "Parse value option [%s] : [%s].", name_.c_str(), content.c_str());
        }
    }
    return AsccStatus::SUCCESS;
}

}  // namespace Ascc