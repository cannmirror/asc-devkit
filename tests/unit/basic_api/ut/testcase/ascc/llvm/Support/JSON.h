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
#ifndef LLVM_SUPPORT_JSON_H
#define LLVM_SUPPORT_JSON_H

#include <variant>
#include <string>
#include <map>
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Process.h"

namespace llvm {
namespace json {
class Value;
class Object {
private:
    std::map<llvm::StringRef, Value> Data;

public:
    Object() = default;
    Value &operator[](llvm::StringRef Key) {
        return Data[Key];
    }
    using iterator = typename std::map<llvm::StringRef, Value>::iterator;
    iterator begin() { return Data.begin(); }
    iterator end() { return Data.end(); }

    using const_iterator = typename std::map<llvm::StringRef, Value>::const_iterator;
    const_iterator begin() const { return Data.begin(); }
    const_iterator end() const { return Data.end(); }

    Value *get(StringRef K);
    const Value *get(StringRef K) const;

    llvm::Optional<int64_t> getInteger(StringRef K) const;
    llvm::Optional<llvm::StringRef> getString(llvm::StringRef K) const;
};

// simple Value
class Value {
public:
    enum class Type { Null, Bool, Int, Uint, Double, String, Object };

private:
    std::variant<std::monostate, bool, int64_t, uint32_t, double, std::string, Object> data;

public:
    Value() : data(std::monostate{}) {}

    Value(bool b) : data(b) {}
    Value(int i) : data(static_cast<int64_t>(i)) {}
    Value(int64_t i) : data(i) {}
    Value(uint32_t u) : data(u) {}
    Value(double d) : data(d) {}
    Value(const std::string& s) : data(s) {}
    Value(const char* s) : data(std::string(s)) {}
    Value(const Object& obj) : data(obj) {}
    Value(Object&& obj) : data(std::move(obj)) {}

    bool isNull() const   { return data.index() == 0; }
    bool isBool() const   { return data.index() == 1; }
    bool isInt() const    { return data.index() == 2; }
    bool isUint() const   { return data.index() == 3; }
    bool isDouble() const { return data.index() == 4; }
    bool isString() const { return data.index() == 5; }
    bool isObject() const { return data.index() == 6; }

    bool getAsBoolean() const {
        if (isBool()) return std::get<bool>(data);
        throw std::runtime_error("Not a boolean");
    }

    llvm::Optional<int64_t> getAsInteger() const {
        return llvm::None;
    }

    llvm::Optional<llvm::StringRef> getAsString() const {
        return llvm::None;
    }

    const Object *getAsObject() const {
        if (isObject()) return &std::get<Object>(data);
        return nullptr;
    }

    Value& operator=(const Object& obj) {
        data = obj;
        return *this;
    }

    Value& operator=(Object&& obj) {
        data = std::move(obj);
        return *this;
    }
};

inline Expected<Value> parse(StringRef JSON) {
    llvm::json::Object obj;
    obj["status"] = true;
    obj["code"] = 200;
    return llvm::json::Value(std::move(obj));
}
} // namespace json
} // namespace llvm
#endif