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
 * \file ascc_info_storage.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_INFO_STORAGE_H__
#define __INCLUDE_ASCC_INFO_STORAGE_H__

#include <memory>
#include <unordered_map>
#include "ascc_types.h"
#include "ascc_info_base.h"

namespace Ascc {
class AsccInfoStorage {
public:
    inline static AsccInfoStorage &GetInstance()
    {
        static AsccInfoStorage instance;
        return instance;
    }
    using FileInfos = std::unordered_map<AscCursorTypes, std::shared_ptr<AsccInfoBase>>;

public:

    void AddInfo(const std::string& fileKey, AscCursorTypes cursorType, std::shared_ptr<AsccInfoBase> infoPtr)
    {
        allInfos[fileKey][cursorType] = infoPtr;
    }

    const std::unordered_map<std::string, FileInfos>& GetAllInfos() const
    {
        return this->allInfos;
    }

    std::shared_ptr<AsccInfoBase> GetInfo(const std::string& fileKey, AscCursorTypes cursorType) const
    {
        auto itFile = allInfos.find(fileKey);
        if (itFile != allInfos.end()) {
            auto itFunc = itFile->second.find(cursorType);
            if (itFunc != itFile->second.end()) {
                return itFunc->second;
            }
        }
        return nullptr;
    }

private:
    AsccInfoStorage() = default;
    ~AsccInfoStorage() = default;
    AsccInfoStorage(const AsccInfoStorage &) = delete;
    AsccInfoStorage &operator=(const AsccInfoStorage &) = delete;
    AsccInfoStorage(const AsccInfoStorage &&) = delete;
    AsccInfoStorage &operator=(const AsccInfoStorage &&) = delete;
    std::unordered_map<std::string, FileInfos> allInfos;
};
} // Ascc
#endif