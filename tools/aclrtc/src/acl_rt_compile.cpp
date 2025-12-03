/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <stdlib.h>
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#include <unordered_map>
#include <thread>
#include <vector>
#include <string>
#include <string_view>
#include <unistd.h>

#include "acl_base.h"
#include "acl_rt_compile.h"

#if !defined(UT_TEST) && !defined(ST_TEST)
namespace {
#endif
const int ACL_ERROR_RTC_INVALID_PROG = 176000;                          // 无效的aclrtcProg (hanle)
const int ACL_ERROR_RTC_INVALID_INPUT = 176001;                         // 除prog以外的入参错误
const int ACL_ERROR_RTC_INVALID_OPTION = 176002;                        // 编译选项错误
const int ACL_ERROR_RTC_COMPILATION = 176003;                           // 编译报错
const int ACL_ERROR_RTC_LINKING = 176004;                               // 链接报错
const int ACL_ERROR_RTC_NO_NAME_EXPR_AFTER_COMPILATION = 176005;        // 编译后没有函数名
const int ACL_ERROR_RTC_NO_LOWERED_NAMES_BEFORE_COMPILATION = 176006;   // 编译后核函数名无法转换成Mangling名称
const int ACL_ERROR_RTC_NAME_EXPR_NOT_VALID = 176007;                   // 传入无效的核函数名
const int ACL_ERROR_RTC_PROG_CREATION_FAILURE = 276000;                 // 创建aclrtcProg (hanle) 失败
const int ACL_ERROR_RTC_OUT_OF_MEMORY = 276001;                         // 内存不足
const int ACL_ERROR_RTC_FAILURE = 576000;                               // ACLRTC内部错误
// ACL_SUCCESS  0

enum class AclrtcType {
    ACL_RTC_TYPE_AICORE = 0,
    ACL_RTC_TYPE_AICPU
};

enum class CompileType : uint32_t {
    ASC = 0,
    AICPU,
    CCE,
};

class AclrtcProgram {
public:
    AclrtcProgram() = default;
    AclrtcProgram(void *program) : program_(program) {}
    AclrtcProgram(void *program, AclrtcType type) : program_(program), type_(type) {}
    void* GetProgram() const { return program_; }
    void SetProgram(void *program) { program_ = program; }
    AclrtcType GetType() const { return type_; }
    void SetType(AclrtcType type) { type_ = type; }
private:
    void *program_ = nullptr;
    AclrtcType type_ = AclrtcType::ACL_RTC_TYPE_AICORE;
};

// bisheng rtc
using asrtcProgram = void*;
typedef enum {
    ASRTC_SUCCESS = 0,
    ASRTC_ERROR_OUT_OF_MEMORY,
    ASRTC_ERROR_PROGRAM_CREATION_FAILURE,
    ASRTC_ERROR_INVALID_PROGRAM,
    ASRTC_ERROR_INVALID_INPUT,
    ASRTC_ERROR_INVALID_OPTION,
    ASRTC_ERROR_COMPILE,
    ASRTC_ERROR_LINK,
    ASRTC_ERROR_NOT_IMPLEMENTED,
    ASRTC_ERROR_INTERNAL_ERROR,
    ASRTC_ERROR_IO
} asrtcResult;

// utils function
inline bool EndsWith(std::string_view str, std::string_view suffix) noexcept
{
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.substr(str.length() - suffix.length()) == suffix;
}

std::string ExtractCannPath(const std::string& pluginPath) {
    const std::vector<std::string> potentialPath = {
        "/x86_64-linux/lib64/libacl_rtc.so",
        "/aarch64-linux/lib64/libacl_rtc.so",
    };
    for (const std::string& expectedPath : potentialPath) {
        if (EndsWith(pluginPath, expectedPath)) {
            return pluginPath.substr(0, pluginPath.size() - expectedPath.size());
        }
    }
    return std::string();
}

std::string GetCannPath() {
    Dl_info info;
    const void* symbolAddr = reinterpret_cast<const void*>(&aclrtcCreateProg);
    if (dladdr(symbolAddr, &info) != 0) {
        char absPath[PATH_MAX] = {0};
        if (realpath(info.dli_fname, absPath) != nullptr) {
            std::string cannPath = ExtractCannPath(std::string(absPath));
            return cannPath;
        }
    }
    return "";
}

bool PathCheck(const char* path) {
    return (access(path, W_OK) == 0 || access(path, R_OK) == 0 || access(path, F_OK) == 0);
}

// set compile type
inline CompileType& GetCompileType()
{
    static thread_local CompileType type = CompileType::ASC;
    return type;
}

inline void SetCompileTypeWithSuffix(const char* name)
{
    auto checkAndSetType = [name](CompileType type) {
        constexpr const char* compile_suffix_list[] = {
            ".asc",
            ".aicpu",
            ".cce"
        };
        if (EndsWith(name, compile_suffix_list[static_cast<uint32_t>(type)])) {
            GetCompileType() = type;
        }
    };
    // check .cce suffix
    checkAndSetType(CompileType::CCE);
}

// 1. define function pointer
using asrtcCreateProgramFuncPtr = 
    asrtcResult (*)(asrtcProgram, const char*, const char*, int, const char* const*, const char* const*);
using asrtcDestroyProgramFuncPtr = asrtcResult (*)(asrtcProgram);
using asrtcCompileProgramFuncPtr = asrtcResult (*)(asrtcProgram, int, const char* const*);
using asrtcGetDeviceELFSizeFuncPtr = asrtcResult (*)(asrtcProgram, size_t*);
using asrtcGetDeviceELFFuncPtr = asrtcResult (*)(asrtcProgram, char*);
using asrtcAddNameExpressionFuncPtr = asrtcResult (*)(asrtcProgram, const char* const);
using asrtcGetLoweredNameFuncPtr = asrtcResult (*)(asrtcProgram, const char*, const char**);
using asrtcGetProgramLogSizeFuncPtr = asrtcResult (*)(asrtcProgram, size_t*);
using asrtcGetProgramLogFuncPtr = asrtcResult (*)(asrtcProgram, char*);

// 2. global function pointer
void* handle = nullptr;
asrtcCreateProgramFuncPtr asrtcCreateProgramPtr = nullptr;
asrtcDestroyProgramFuncPtr asrtcDestroyProgramPtr = nullptr;
asrtcCompileProgramFuncPtr asrtcCompileProgramPtr = nullptr;
asrtcGetDeviceELFSizeFuncPtr asrtcGetDeviceELFSizePtr = nullptr;
asrtcGetDeviceELFFuncPtr asrtcGetDeviceELFPtr = nullptr;
asrtcAddNameExpressionFuncPtr asrtcAddNameExpressionPtr = nullptr;
asrtcGetLoweredNameFuncPtr asrtcGetLoweredNamePtr = nullptr;
asrtcGetProgramLogSizeFuncPtr asrtcGetProgramLogSizePtr = nullptr;
asrtcGetProgramLogFuncPtr asrtcGetProgramLogPtr = nullptr;

void __attribute__((constructor)) LoadExtraLib() {
    std::string cannPath = GetCannPath();
    std::string libPath = cannPath + "/x86_64-linux/ccec_compiler/lib/libasrtc.so";
    // 3. dlopen
    if (!PathCheck(libPath.c_str())) {
        return;
    }
    handle = dlopen(libPath.c_str(), RTLD_GLOBAL | RTLD_NOW);
    // 4. dlsym
    asrtcCreateProgramPtr = (asrtcCreateProgramFuncPtr)dlsym(handle, "asrtcCreateProgram");
    asrtcDestroyProgramPtr = (asrtcDestroyProgramFuncPtr)dlsym(handle, "asrtcDestroyProgram");
    asrtcCompileProgramPtr = (asrtcCompileProgramFuncPtr)dlsym(handle, "asrtcCompileProgram");
    asrtcGetDeviceELFSizePtr = (asrtcGetDeviceELFSizeFuncPtr)dlsym(handle, "asrtcGetDeviceELFSize");
    asrtcGetDeviceELFPtr = (asrtcGetDeviceELFFuncPtr)dlsym(handle, "asrtcGetDeviceELF");
    asrtcAddNameExpressionPtr = (asrtcAddNameExpressionFuncPtr)dlsym(handle, "asrtcAddNameExpression");
    asrtcGetLoweredNamePtr = (asrtcGetLoweredNameFuncPtr)dlsym(handle, "asrtcGetLoweredName");
    asrtcGetProgramLogSizePtr = (asrtcGetProgramLogSizeFuncPtr)dlsym(handle, "asrtcGetProgramLogSize");
    asrtcGetProgramLogPtr = (asrtcGetProgramLogFuncPtr)dlsym(handle, "asrtcGetProgramLog");
}

void __attribute__((destructor)) UnloadExtraLib() {
    if (handle != nullptr) {
        dlclose(handle);
        handle = nullptr;
    }
}

const std::unordered_map<asrtcResult, aclError> ccecRet2AclrtcRet = {
    {asrtcResult::ASRTC_SUCCESS, ACL_SUCCESS},
    {asrtcResult::ASRTC_ERROR_OUT_OF_MEMORY, ACL_ERROR_RTC_OUT_OF_MEMORY},
    {asrtcResult::ASRTC_ERROR_PROGRAM_CREATION_FAILURE, ACL_ERROR_RTC_PROG_CREATION_FAILURE},
    {asrtcResult::ASRTC_ERROR_INVALID_PROGRAM, ACL_ERROR_RTC_INVALID_PROG},
    {asrtcResult::ASRTC_ERROR_INVALID_INPUT, ACL_ERROR_RTC_INVALID_INPUT},
    {asrtcResult::ASRTC_ERROR_INVALID_OPTION, ACL_ERROR_RTC_INVALID_OPTION},
    {asrtcResult::ASRTC_ERROR_COMPILE, ACL_ERROR_RTC_COMPILATION},
    {asrtcResult::ASRTC_ERROR_LINK, ACL_ERROR_RTC_LINKING},
    {asrtcResult::ASRTC_ERROR_NOT_IMPLEMENTED, ACL_ERROR_RTC_FAILURE},
    {asrtcResult::ASRTC_ERROR_INTERNAL_ERROR, ACL_ERROR_RTC_FAILURE},
    {asrtcResult::ASRTC_ERROR_IO, ACL_ERROR_RTC_FAILURE},
};

aclError ErrorCodeProcess(asrtcResult result) {
    auto it = ccecRet2AclrtcRet.find(result);
    if (it != ccecRet2AclrtcRet.end()) {
        return it->second;
    }
    return ACL_ERROR_RTC_FAILURE;
}

__attribute__((unused))
aclError aclrtcAddNameExpr(aclrtcProg prog, const char *nameExpr) {
    AclrtcProgram ascProg(prog);
    return ErrorCodeProcess(asrtcAddNameExpressionPtr(prog, nameExpr));
}

__attribute__((unused))
aclError aclrtcGetLoweredName(aclrtcProg prog, const char *nameExpr, const char **manglingName) {
    AclrtcProgram ascProg(prog);
    return ErrorCodeProcess(asrtcGetLoweredNamePtr(prog, nameExpr, manglingName));
}
#if !defined(UT_TEST) && !defined(ST_TEST)
}
#endif

aclError aclrtcCreateProg(aclrtcProg *prog, const char *src, const char *name, int numHeaders, const char **headers,
    const char **includeNames) {
    SetCompileTypeWithSuffix(name);
    return ErrorCodeProcess(asrtcCreateProgramPtr(prog, src, name, numHeaders, headers, includeNames));
}

aclError aclrtcCompileProg(aclrtcProg prog, int numOptions, const char **options) {
    std::vector<const char*> optionsPlugin;
    if (GetCompileType() == CompileType::ASC) {
        optionsPlugin.emplace_back("-xasc");
    }
    optionsPlugin.emplace_back("-std=c++17");
    std::string cannPath = GetCannPath();
    std::string includePath = cannPath + "/include";
    std::string tikcfwPath = cannPath + "/compiler/tikcpp/tikcfw";
    std::string interfacePath = cannPath + "/compiler/tikcpp/tikcfw/interface";
    std::string implPath = cannPath + "/compiler/tikcpp/tikcfw/impl";
    if (!PathCheck(cannPath.c_str()) || !PathCheck(includePath.c_str()) || !PathCheck(tikcfwPath.c_str()) ||
        !PathCheck(interfacePath.c_str()) || !PathCheck(implPath.c_str())) {
        return ACL_ERROR_RTC_FAILURE;
    }
    includePath = "-I" + includePath;
    tikcfwPath = "-I" + tikcfwPath;
    interfacePath = "-I" + interfacePath;
    implPath = "-I" + implPath;
    optionsPlugin.emplace_back(includePath.c_str());
    optionsPlugin.emplace_back(tikcfwPath.c_str());
    optionsPlugin.emplace_back(interfacePath.c_str());
    optionsPlugin.emplace_back(implPath.c_str());
    for (int i = 0; i < numOptions; ++i) {
        optionsPlugin.emplace_back(options[i]);
    }
    AclrtcProgram ascProg(prog);
    return ErrorCodeProcess(asrtcCompileProgramPtr(prog, optionsPlugin.size(), optionsPlugin.data()));
}

aclError aclrtcDestroyProg(aclrtcProg *prog) {
    return ErrorCodeProcess(asrtcDestroyProgramPtr(prog));
}

aclError aclrtcGetBinData(aclrtcProg prog, char *binData) {
    AclrtcProgram ascProg(prog);
    return ErrorCodeProcess(asrtcGetDeviceELFPtr(prog, binData));
}

aclError aclrtcGetBinDataSize(aclrtcProg prog, size_t *binDataSizeRet) {
    AclrtcProgram ascProg(prog);
    return ErrorCodeProcess(asrtcGetDeviceELFSizePtr(prog, binDataSizeRet));
}

aclError aclrtcGetCompileLogSize(aclrtcProg prog, size_t *logSizeRet) {
    AclrtcProgram ascProg(prog);
    return ErrorCodeProcess(asrtcGetProgramLogSizePtr(prog, logSizeRet));
}

aclError aclrtcGetCompileLog(aclrtcProg prog, char *log) {
    AclrtcProgram ascProg(prog);
    return ErrorCodeProcess(asrtcGetProgramLogPtr(prog, log));
}