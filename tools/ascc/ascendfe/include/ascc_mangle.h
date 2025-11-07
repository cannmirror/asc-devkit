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
 * \file ascc_mangle.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_MANGLE_H__
#define __INCLUDE_ASCC_MANGLE_H__
#include <string>
#include <unordered_map>

namespace Ascc {
class AsccMangle {
public:
    inline static AsccMangle &GetInstance()
    {
        static AsccMangle instance;
        return instance;
    }
    std::unordered_map<std::string, std::string>& GetOriginToFixedMangledNames()
    {
        return originToFixedMangledNames;
    }

private:
    AsccMangle() = default;
    ~AsccMangle() = default;
    std::unordered_map<std::string, std::string> originToFixedMangledNames;
};
}
#endif