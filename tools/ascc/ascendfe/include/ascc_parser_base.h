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
 * \file ascc_parser_base.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_PARSER_BASE_H__
#define __INCLUDE_ASCC_PARSER_BASE_H__
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>

#include "ascc_types.h"
namespace Ascc {
class AsccParserBase {
public:
    virtual ~AsccParserBase() = default;

    virtual void InitParser(const std::string &name, const std::string &description, const bool &isContinue,
        const bool &isSupportEqual) = 0;

    virtual AsccStatus Parse(const std::vector<std::string>& args, std::unordered_set<uint32_t>& parsedIndex) = 0;

    virtual std::vector<std::string> GetValue() const = 0;

    virtual bool IsValid() const = 0;

    virtual bool IsContinue() const = 0;

    virtual bool IsSupportEqual() const = 0;

    virtual const std::string& GetName() const = 0;

    virtual const std::string& GetDescription() const = 0;
};
} // namespace Ascc
#endif // __INCLUDE_ASCC_PARSER_BASE_H__