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
 * \file ascc_parser_bool.cpp
 * \brief
 */

#include "ascc_parser_bool.h"

#include <csignal>
#include <unordered_set>

#include "ascc_types.h"
#include "ascc_log.h"

namespace Ascc {
void AsccParserBool::InitParser(const std::string &name, const std::string &description,
    const bool &isContinue, const bool &isSupportEqual)
{
    (void)isContinue;
    (void)isSupportEqual;
    name_ = name;
    description_ = description;
}

AsccStatus AsccParserBool::Parse(const std::vector<std::string>& args, std::unordered_set<uint32_t>& parsedIndex)
{
    for (size_t i = 0; i < args.size(); ++i) {
        if (parsedIndex.find(i) != parsedIndex.end()) {
            continue;
        }
        if (args[i] == name_) {
            ASC_LOG_ASC_INFO(OPTION, "Parse bool option [%s]", name_.c_str());
            parsedIndex.emplace(i);
            valid_ = true;
        }
    }
    return AsccStatus::SUCCESS;
}

} // Ascc