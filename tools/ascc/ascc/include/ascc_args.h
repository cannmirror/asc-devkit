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
 * \file ascc_args.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_ARGS_H__
#define __INCLUDE_ASCC_ARGS_H__
#include "ascc_option.h"
namespace Ascc {

// input + output files
inline Opt<std::string> outputOpt("output-file", ShortDesc("o"), HasArgFlag::REQUIRED, Init("a.out"),
    ValueDesc("<file>"), HelpDesc("Specify name and location of the output file.\n"));

inline OptList<std::string> inputOpt(FormatFlag::POSITIONAL, ValueDesc("input files"));

// compile .o / .so
inline Opt<bool> compileOpt("compile", ShortDesc("c"), HasArgFlag::NONE,
    HelpDesc("Compile each .C/.c/.cc/.cxx/.cpp/.asc input file into an object file.\n"));

inline Opt<std::string> sharedOpt("shared", ShortDesc("shared"), HasArgFlag::NONE,
    HelpDesc("Generate a shared library.\n"));

// dependency files
inline Opt<bool> mdOpt("generate-dependencies-with-compile", ShortDesc("MD"), HasArgFlag::NONE,
    HelpDesc("Generate a dependency file and compile the input file.\n"));

inline Opt<bool> mmdOpt("generate-nonsystem-dependencies-with-compile", ShortDesc("MMD"), HasArgFlag::NONE,
    HelpDesc("Same as --generate-dependencies-with-compile, but skip header files found in system directories.\n"));

inline Opt<bool> mpOpt("generate-dependency-targets", ShortDesc("MP"), HasArgFlag::NONE,
    HelpDesc("Add an empty target for each dependency.\n"));

inline Opt<std::string> mfOpt("dependency-output", ShortDesc("MF"), HasArgFlag::REQUIRED, ValueDesc("<file>"),
    HelpDesc("Specify the output file for the dependency file generated with -M/-MM/-MD/-MMD.\n"));

inline Opt<std::string> mtOpt("dependency-target-name", ShortDesc("MT"), HasArgFlag::REQUIRED,
    ValueDesc("<target_name>"),
    HelpDesc("Specify the target name of the generated rule when generating a dependency file.\n"));

// include
inline OptList<std::string> incOpt("include-path", ShortDesc("I"), HasArgFlag::REQUIRED, FormatFlag::PREFIX,
    ValueDesc("<dir>"), HelpDesc("Specify the list of include search paths.\n"), MiscFlags::COMMAS);

// library
inline OptList<std::string> libDirOpt("library-path", ShortDesc("L"), HasArgFlag::REQUIRED, FormatFlag::PREFIX,
    ValueDesc("<dir>"), HelpDesc("Specify the list of library search paths.\n"), MiscFlags::COMMAS);

inline OptList<std::string> libOpt("library", ShortDesc("l"), HasArgFlag::REQUIRED, FormatFlag::PREFIX,
    ValueDesc("<library>"),
    HelpDesc("Specify libraries to be used in the linking stage without the library file extension.\n"),
    MiscFlags::COMMAS);

// macro
inline OptList<std::string> defOpt("define-macro", ShortDesc("D"), HasArgFlag::REQUIRED, FormatFlag::PREFIX,
    ValueDesc("<def>"), HelpDesc("Specify macro definitions to define for use during preprocessing or compilation.\n"),
    MiscFlags::COMMAS);

// other compile options
inline Opt<bool> debugOpt("debug", ShortDesc("g"), HasArgFlag::NONE,
    HelpDesc("Generate debug information for code.\n"));

inline Opt<bool> sanitizerOpt("sanitizer", ShortDesc("sanitizer"), HasArgFlag::NONE,
    HelpDesc("Generate saitizer information for code.\n"));

inline Opt<std::string> optimizeOpt("optimize", ShortDesc("O"), HasArgFlag::REQUIRED, FormatFlag::PREFIX,
    ValueDesc("<level>"), HelpDesc("Provide optimization reports for the specified kind of optimization.\n"));

inline Opt<std::string> saveOpt("save-temps", HasArgFlag::OPTIONAL, ValueDesc("(dir)"),
    HelpDesc("Save intermediate compilation results.\n"));

inline Opt<std::string> archOpt("npu-architecture", ShortDesc("arch"), HasArgFlag::REQUIRED,
    ArgOccNumFlag::REQUIRED, ValueDesc("<arch>"),
    HelpDesc("Specify the name of the NPU architecture.\n"));

// frontend options
inline Opt<std::string> timeOpt("time", ShortDesc("time"), HasArgFlag::NONE,
    HelpDesc("Collect the statistics of compilation time.\n"));

inline Opt<std::string> printCmdOpt("verbose", ShortDesc("v"), HasArgFlag::NONE,
    HelpDesc("Collect the commands of compilation process.\n"));
}
#endif
