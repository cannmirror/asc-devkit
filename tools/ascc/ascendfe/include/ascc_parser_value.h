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
 * \file ascc_parser_value.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_PARSER_VALUE_H__
#define __INCLUDE_ASCC_PARSER_VALUE_H__

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>

#include "ascc_parser_base.h"

namespace Ascc {
class AsccParserValue : public AsccParserBase {
public:
    AsccParserValue() = default;

    void InitParser(const std::string &name, const std::string &description, const bool &isContinue,
        const bool &isSupportEqual) override;

    AsccStatus Parse(const std::vector<std::string>& args, std::unordered_set<uint32_t>& parsedIndex) override;

    std::vector<std::string> GetValue() const override { return value_; }

    bool IsValid() const override { return valid_; }

    bool IsContinue() const override {return isContinue_;}

    bool IsSupportEqual() const override { return isSupportEqual_; }

    const std::string& GetName() const override { return name_; }

    const std::string& GetDescription() const override { return description_; }

private:
    AsccStatus ParseOption(const std::vector<std::string>& args, std::unordered_set<uint32_t> &parsedIndex);
    AsccStatus ParseContiOption(const std::vector<std::string>& args, std::unordered_set<uint32_t> &parsedIndex);
    std::vector<std::string> SplitByComma(const std::string& str) const;
private:
    std::string name_;
    std::string description_;
    std::vector<std::string> value_;
    bool valid_ = false;
    bool isContinue_ = false;
    bool isSupportEqual_ = false;
};
} // namespace Ascc
#endif // __INCLUDE_ASCC_PARSER_VALUE_H__