/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_SUPPORT_COMMAND_LINE_H
#define LLVM_SUPPORT_COMMAND_LINE_H

namespace llvm {
namespace cl {
class OptionCategory {
private:
    StringRef const Name;
    StringRef const Description;

    void registerCategory() {}

public:
    OptionCategory(std::string Name) : Name(Name) {
        registerCategory();
    }
};
}
}
#endif