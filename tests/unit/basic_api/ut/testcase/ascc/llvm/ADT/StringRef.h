/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_ADT_STRING_ERF_H
#define LLVM_ADT_STRING_ERF_H
#include <cstring>
#include <string>

namespace llvm {
class StringRef {
private:
    const char *Data = "";
    size_t Length = 0;
public:
    static constexpr size_t strLen(const char *Str) {
        return Str != nullptr ? strlen(Str) : 0;
    }
    StringRef() = default;
    StringRef(std::nullptr_t) = delete;
    constexpr StringRef(const char *data, size_t length)
        : Data(data), Length(length) {}
    constexpr StringRef(const char *Str)
        : Data(Str), Length(Str ? strLen(Str) : 0) {}
    StringRef(const std::string &Str)
        : Data(Str.data()), Length(Str.length()) {}

    using iterator = const char *;
    iterator begin() const { return Data; }
    iterator end() const { return Data + Length; }
    bool empty() const { return Length == 0; }
    std::string str() const {
        if (!Data) return std::string();
        return std::string(Data, Length);
    }
    std::string getAsString() const;
    const char *data() const { return Data; }
    size_t size() const { return Length; }
    friend bool operator==(const StringRef& LHS, const StringRef& RHS) {
        return strcmp(LHS.data(), RHS.data()) == 0;
    }
    friend bool operator!=(const StringRef& LHS, const StringRef& RHS) {
        return strcmp(LHS.data(), RHS.data()) != 0;
    }
    friend bool operator<(StringRef LHS, StringRef RHS) {
        return strcmp(LHS.data(), RHS.data()) < 0;
    }
};
}
#endif
