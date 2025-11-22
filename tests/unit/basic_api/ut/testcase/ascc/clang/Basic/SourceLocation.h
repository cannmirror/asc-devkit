/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CLANG_BASIC_SOURCELOCATION_H
#define CLANG_BASIC_SOURCELOCATION_H

#include <cstdint>
#include "clang/Basic/FileManager.h"

namespace clang {
constexpr uint32_t LINE = 2;
constexpr uint32_t COL = 2;
class SourceManager;
class FileID {
    int ID = 0;
public:
    bool isValid() const { return ID != 0; }
    bool isInvalid() const { return ID == 0; }
};

class SourceLocation {
public:
    SourceLocation() = default;
    using UIntTy = uint32_t;
    using IntTy = int32_t;
    bool isValid() const { return ID != 0; }
    bool isMacroID() const { return ID != 0; }
private:
    UIntTy ID = 0;
};
class SourceRange {
  SourceLocation B;
  SourceLocation E;

public:
  SourceRange() = default;
  SourceRange(SourceLocation loc) : B(loc), E(loc) {}
  SourceRange(SourceLocation begin, SourceLocation end) : B(begin), E(end) {}

  SourceLocation getBegin() const { return B; }
  SourceLocation getEnd() const { return E; }

};
class FullSourceLoc : public SourceLocation {
public:
    FullSourceLoc() = default;
    explicit FullSourceLoc(SourceLocation Loc, const SourceManager &SM) : SrcMgr(&SM) {}
    FileEntry *getFileEntry() {
        return &fe;
    }
    unsigned getSpellingLineNumber() {
        return 0;
    }
    unsigned getSpellingColumnNumber() {
        return 0;
    }
    const SourceManager *SrcMgr = nullptr;
    FileEntry fe;
};

class PresumedLoc {
public:
    PresumedLoc () = default;
    bool isInvalid() const { return Filename == nullptr; }
    bool isValid() const { return Filename != nullptr; }
    const char *getFilename() const {
        return Filename;
    }
    unsigned getLine() const {
        return Line;
    }
    unsigned getColumn() const {
        return Col;
    }
private:
    const char *Filename = nullptr;
    unsigned Line = LINE;
    unsigned Col = COL;
};
}
#endif