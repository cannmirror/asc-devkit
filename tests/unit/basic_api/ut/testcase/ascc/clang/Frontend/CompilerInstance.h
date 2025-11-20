/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CLANG_FRONTEND_COMPILER_INSTANCE_H
#define CLANG_FRONTEND_COMPILER_INSTANCE_H
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Basic/Diagnostic.h"

namespace clang {
class CompilerInstance {
    std::shared_ptr<CompilerInvocation> Invocation;
    std::shared_ptr<DiagnosticsEngine> Diagnostics;
public:
    HeaderSearchOptions &getHeaderSearchOpts() {
        return Invocation->getHeaderSearchOpts();
    }

    ASTContext &getASTContext() const {}

    DiagnosticsEngine &getDiagnostics()
    {
        Diagnostics = std::make_shared<DiagnosticsEngine>();
        return *Diagnostics;
    }
};
}
#endif