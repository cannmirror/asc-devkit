/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "JSON.h"

namespace llvm {
namespace json {
Value *Object::get(StringRef K) {
    auto I = Data.find(K);
    if (I == end())
        return nullptr;
    return &I->second;
}

const Value *Object::get(StringRef K) const {
    auto I = Data.find(K);
    if (I == end())
        return nullptr;
    return &I->second;
}

llvm::Optional<int64_t> Object::getInteger(StringRef K) const {
    if (auto *V = get(K))
        return V->getAsInteger();
    return llvm::None;
}

llvm::Optional<llvm::StringRef> Object::getString(llvm::StringRef K) const {
    if (auto *V = get(K))
        return V->getAsString();
    return llvm::None;
}
} // namespace json
} // namespace llvm
