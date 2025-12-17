/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef CLANG_BASIC_SOURCEMANAGER_H
#define CLANG_BASIC_SOURCEMANAGER_H
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/FileManager.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class SLocEntry {
public:
  SLocEntry() = default;
  bool isFile() const { return false; }
};
class SourceManager {
public:
    SourceManager() = default;
    FileID getFileID(SourceLocation SpellingLoc) const { return FileID(); };
    SourceLocation getSpellingLoc(SourceLocation Loc) const { return Loc;};
    const FileEntry *getFileEntryForID(FileID FID) {};
    PresumedLoc getPresumedLoc(SourceLocation Loc, bool UseLineDirectives = true) const { return pLoc; }
    SourceLocation getExpansionLoc(SourceLocation Loc) const { return sLoc; }
    unsigned getExpansionLineNumber(SourceLocation Loc, bool *Invalid = nullptr) const { return 0u; }
    unsigned getSpellingLineNumber(SourceLocation Loc, bool *Invalid = nullptr) const { return 0u; }
    llvm::StringRef getFilename(SourceLocation SpellingLoc) const {
        return llvm::StringRef("");
    }
    SourceLocation getImmediateMacroCallerLoc(SourceLocation Loc) { return sLoc; };
    const SLocEntry &getSLocEntry(FileID FID, bool *Invalid = nullptr) const { return se; }
    llvm::StringRef getBufferData(FileID FID, bool *Invalid = nullptr) const { return llvm::StringRef(""); }
    std::pair<FileID, unsigned int> getDecomposedLoc(SourceLocation Loc) const { return std::make_pair(FileID(), 0u); }
    unsigned int getSpellingColumnNumber(SourceLocation Loc, bool *Invalid = nullptr) const { return 0u; }
private:
    SLocEntry se;
    PresumedLoc pLoc;
    SourceLocation sLoc;
};
}  // namespace clang
#endif