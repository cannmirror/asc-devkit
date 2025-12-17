/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_SUPPORT_ERROR_H
#define LLVM_SUPPORT_ERROR_H

#include <string>
#include <variant>

namespace llvm {
class Error {
public:
    Error() = default;
    ~Error() = default;
};

template <class T>
class Expected {
public:
    Expected(const T& value) : data_(value) {}
    Expected(T&& value) : data_(std::move(value)) {}
    Expected(const std::string& error) : data_(std::make_exception_ptr(std::runtime_error(error))) {}
    bool operator!() const {
        return true;
    }
    Error takeError() {
        return Error();
    }
    T* operator->()  {
        return &Value;
    }

private:
    std::variant<T, std::exception_ptr> data_;
    T Value;
};

inline std::string toString(Error Err) {
    return "Error";
}
} // namespace llvm
#endif