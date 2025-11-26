/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file asc_utils.h
 * \brief
 */

#ifndef __INCLUDE_INTERNAL_ASC_UTILS_H__
#define __INCLUDE_INTERNAL_ASC_UTILS_H__

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <utility>
#include <sys/stat.h>

#include "asc_struct.h"

namespace AscPlugin {

struct KernelFuncInfo {
    std::string mangledName;
    std::string fileName;
    uint32_t lineNum;
    uint32_t colNum;
    bool operator==(const KernelFuncInfo &other) const {
        return mangledName == other.mangledName &&
               fileName == other.fileName &&
               lineNum == other.lineNum &&
               colNum == other.colNum;
    }
};

struct KernelTypeResult {
    bool hasMixOneToOne = false;
    bool hasMixOneToTwo = false;
    bool hasMixOneToOneWithKfc = false;
    bool hasMixOneToTwoWithKfc = false;
};

enum class KernelMetaType : uint32_t {
    KERNEL_TYPE_AIV_ONLY = 0,
    KERNEL_TYPE_AIC_ONLY = 1,
    KERNEL_TYPE_MIX_AIV_1_0 = 2,
    KERNEL_TYPE_MIX_AIC_1_0 = 3,
    KERNEL_TYPE_MIX_AIC_1_1 = 4,
    KERNEL_TYPE_MIX_AIC_1_2 = 5,
    KERNEL_TYPE_AICORE = 6,
    KERNEL_TYPE_VECTOR_CORE = 7,
    KERNEL_TYPE_MIX_AICORE = 8,
    KERNEL_TYPE_MIX_VECTOR_CORE = 9,
    KERNEL_TYPE_MAX = 0xff
};

enum class KfcScene : bool {
    Open = true,
    Close = false
};

enum class PathStatus : uint32_t {
    EXIST = 0,
    READ,
    WRITE,
    NOT_EXIST
};

enum class ShortSocVersion : uint32_t {
    ASCEND910B = 0,
    ASCEND310P = 1,
    ASCEND910 = 2,
    ASCEND310B = 3,
    ASCEND910_95 = 4,
    INVALID_TYPE = 0xffffffff
};

// for split architecture, cube / vec; otherwise means aicore(use cube) + vec
enum class CoreType: uint32_t {
    CUBE = 0,
    VEC
};

inline const std::unordered_map<KernelMetaType, std::string> KTYPE_STR_MAP = {
    {KernelMetaType::KERNEL_TYPE_AIV_ONLY, "KERNEL_TYPE_AIV_ONLY"},
    {KernelMetaType::KERNEL_TYPE_AIC_ONLY, "KERNEL_TYPE_AIC_ONLY"},
    {KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0, "KERNEL_TYPE_MIX_AIV_1_0"},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0, "KERNEL_TYPE_MIX_AIC_1_0"},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1, "KERNEL_TYPE_MIX_AIC_1_1"},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "KERNEL_TYPE_MIX_AIC_1_2"}
};

// related to CodeMode in host_stub_util.py
inline const std::unordered_map<KernelMetaType, PluginKernelType> META_KTYPE_TO_KTYPE = {
    {KernelMetaType::KERNEL_TYPE_AIV_ONLY, PluginKernelType::AIV},
    {KernelMetaType::KERNEL_TYPE_AIC_ONLY, PluginKernelType::AIC},
    {KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0, PluginKernelType::MIX},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0, PluginKernelType::MIX},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1, PluginKernelType::MIX},
    {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, PluginKernelType::MIX},
    // 310P
    {KernelMetaType::KERNEL_TYPE_AICORE, PluginKernelType::AIC},
    {KernelMetaType::KERNEL_TYPE_MIX_VECTOR_CORE, PluginKernelType::AIC}
};

const std::unordered_map<std::string, AscPlugin::ShortSocVersion> SOC_VERSION_MAP = {
    {"Ascend910B1", AscPlugin::ShortSocVersion::ASCEND910B},        // ascend910b_list
    {"Ascend910B2", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910B2C", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910B3", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910B4", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910B4-1", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910_9391", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910_9381", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910_9372", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910_9392", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910_9382", AscPlugin::ShortSocVersion::ASCEND910B},
    {"Ascend910_9362", AscPlugin::ShortSocVersion::ASCEND910B},

    {"Ascend910A", AscPlugin::ShortSocVersion::ASCEND910},          // ascend910_list
    {"Ascend910ProA", AscPlugin::ShortSocVersion::ASCEND910},
    {"Ascend910B", AscPlugin::ShortSocVersion::ASCEND910},
    {"Ascend910ProB", AscPlugin::ShortSocVersion::ASCEND910},
    {"Ascend910PremiumA", AscPlugin::ShortSocVersion::ASCEND910},

    {"Ascend310P1", AscPlugin::ShortSocVersion::ASCEND310P},        // ascend310p_list
    {"Ascend310P3", AscPlugin::ShortSocVersion::ASCEND310P},
    {"Ascend310P5", AscPlugin::ShortSocVersion::ASCEND310P},
    {"Ascend310P7", AscPlugin::ShortSocVersion::ASCEND310P},
    {"Ascend310P3Vir01", AscPlugin::ShortSocVersion::ASCEND310P},
    {"Ascend310P3Vir02", AscPlugin::ShortSocVersion::ASCEND310P},
    {"Ascend310P3Vir04", AscPlugin::ShortSocVersion::ASCEND310P},
    {"Ascend310P3Vir08", AscPlugin::ShortSocVersion::ASCEND310P},

    {"Ascend310B1", AscPlugin::ShortSocVersion::ASCEND310B},        // ascend310b_list
    {"Ascend310B2", AscPlugin::ShortSocVersion::ASCEND310B},
    {"Ascend310B3", AscPlugin::ShortSocVersion::ASCEND310B},
    {"Ascend310B4", AscPlugin::ShortSocVersion::ASCEND310B},

    {"Ascend910_957b", AscPlugin::ShortSocVersion::ASCEND910_95},   // ascend910_95_list
    {"Ascend910_950z", AscPlugin::ShortSocVersion::ASCEND910_95},
    {"Ascend910_958b", AscPlugin::ShortSocVersion::ASCEND910_95},
    {"Ascend910_958a", AscPlugin::ShortSocVersion::ASCEND910_95},
    {"Ascend910_9599", AscPlugin::ShortSocVersion::ASCEND910_95},
    {"Ascend910_957d", AscPlugin::ShortSocVersion::ASCEND910_95},
    {"Ascend910_9581", AscPlugin::ShortSocVersion::ASCEND910_95},
    {"Ascend910_9589", AscPlugin::ShortSocVersion::ASCEND910_95},
    {"Ascend910_957c", AscPlugin::ShortSocVersion::ASCEND910_95},
};

const std::map<std::pair<AscPlugin::ShortSocVersion, AscPlugin::CoreType>, std::string> CCE_AICORE_MAP = {
    {{AscPlugin::ShortSocVersion::ASCEND910, AscPlugin::CoreType::CUBE}, "dav-c100"},
    {{AscPlugin::ShortSocVersion::ASCEND910, AscPlugin::CoreType::VEC}, "dav-c100"},
    {{AscPlugin::ShortSocVersion::ASCEND310P, AscPlugin::CoreType::CUBE}, "dav-m200"},
    {{AscPlugin::ShortSocVersion::ASCEND310P, AscPlugin::CoreType::VEC}, "dav-m200-vec"},
    {{AscPlugin::ShortSocVersion::ASCEND910B, AscPlugin::CoreType::CUBE}, "dav-c220-cube"},
    {{AscPlugin::ShortSocVersion::ASCEND910B, AscPlugin::CoreType::VEC}, "dav-c220-vec"},
    {{AscPlugin::ShortSocVersion::ASCEND310B, AscPlugin::CoreType::CUBE}, "dav-m300"},
    {{AscPlugin::ShortSocVersion::ASCEND310B, AscPlugin::CoreType::VEC}, "dav-m300"},
    {{AscPlugin::ShortSocVersion::ASCEND910_95, AscPlugin::CoreType::CUBE}, "dav-c310-cube"},
    {{AscPlugin::ShortSocVersion::ASCEND910_95, AscPlugin::CoreType::VEC}, "dav-c310-vec"},
};

const std::unordered_map<std::string, AscPlugin::ShortSocVersion> CCE_AICORE_ARCH_MAP = {
    {"dav-c220-cube", AscPlugin::ShortSocVersion::ASCEND910B},      // ascend910b_list
    {"dav-c220-vec", AscPlugin::ShortSocVersion::ASCEND910B},
    {"dav-c220", AscPlugin::ShortSocVersion::ASCEND910B},
    {"dav-m200", AscPlugin::ShortSocVersion::ASCEND310P},           // ascend310p_list
    {"dav-m200-vec", AscPlugin::ShortSocVersion::ASCEND310P},
    {"dav-c310-cube", AscPlugin::ShortSocVersion::ASCEND910_95},    // ascend910_95_list
    {"dav-c310-vec", AscPlugin::ShortSocVersion::ASCEND910_95},
    {"dav-c310", AscPlugin::ShortSocVersion::ASCEND910_95},
};

const std::unordered_map<AscPlugin::KernelMetaType, std::string> KERNEL_TYPE_STR_MAP = {
    {AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY, "KERNEL_TYPE_AIV_ONLY"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY, "KERNEL_TYPE_AIC_ONLY"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0, "KERNEL_TYPE_MIX_AIV_1_0"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0, "KERNEL_TYPE_MIX_AIC_1_0"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1, "KERNEL_TYPE_MIX_AIC_1_1"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "KERNEL_TYPE_MIX_AIC_1_2"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_AICORE, "KERNEL_TYPE_AICORE"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_VECTOR_CORE, "KERNEL_TYPE_VECTOR_CORE"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AICORE, "KERNEL_TYPE_MIX_AICORE"},
    {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_VECTOR_CORE, "KERNEL_TYPE_MIX_VECTOR_CORE"}
};

const std::unordered_map<AscPlugin::ShortSocVersion, AscPlugin::KernelMetaType> DEFAULT_KERNEL_TYPE_MAP = {
    {AscPlugin::ShortSocVersion::ASCEND310P, AscPlugin::KernelMetaType::KERNEL_TYPE_AICORE},
    {AscPlugin::ShortSocVersion::ASCEND910B, AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2},
    {AscPlugin::ShortSocVersion::ASCEND910_95, AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2},
};

const std::unordered_map<std::string, AscPlugin::KernelMetaType> KERNEL_TYPE_MAP_V220 = {
    {"KERNEL_TYPE_AIV_ONLY", AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY},
    {"KERNEL_TYPE_AIC_ONLY", AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY},
    {"KERNEL_TYPE_MIX_AIV_1_0", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0},
    {"KERNEL_TYPE_MIX_AIC_1_0", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0},
    {"KERNEL_TYPE_MIX_AIC_1_1", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1},
    {"KERNEL_TYPE_MIX_AIC_1_2", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2}
};

const std::unordered_map<std::string, AscPlugin::KernelMetaType> KERNEL_TYPE_MAP_V200 = {
    {"KERNEL_TYPE_AICORE", AscPlugin::KernelMetaType::KERNEL_TYPE_AICORE},
    {"KERNEL_TYPE_MIX_VECTOR_CORE", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_VECTOR_CORE}
};

const std::unordered_map<std::string, AscPlugin::KernelMetaType> KERNEL_TYPE_MAP_C310 = {
    {"KERNEL_TYPE_AIV_ONLY", AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY},
    {"KERNEL_TYPE_AIC_ONLY", AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY},
    {"KERNEL_TYPE_MIX_AIV_1_0", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0},
    {"KERNEL_TYPE_MIX_AIC_1_0", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0},
    {"KERNEL_TYPE_MIX_AIC_1_1", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1},
    {"KERNEL_TYPE_MIX_AIC_1_2", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2}
};

constexpr int32_t ASC_SUCCESS = 0;
constexpr int32_t ASC_FAILURE = 1;
constexpr int32_t ASC_NULLPTR = 2;
constexpr int32_t ASC_CANNPATH_NOT_FOUND = 3;
constexpr int32_t ASC_SOC_NOT_SUPPORT = 4;
constexpr int32_t ASC_JSONSTR_ARG_MISSING = 5;

constexpr uint32_t ASC_PRINT_MASK = 0x01;
constexpr uint32_t ASC_FFTS_MASK = 0x02;
constexpr uint32_t ASC_L2CACHE_HINT_MASK = 0x04;

inline constexpr char ORIGIN_KERNEL_PREFIX[] = "__origin__";
inline constexpr char DEVICE_STUB_PREFIX[] = "__device_stub__";
inline constexpr size_t DEVICE_STUB_PREFIX_LEN = 15;
inline constexpr char ANONYMOUS_NAME[] = "(Anonymous Namespace)";

// key: kernel manglename
// value: {kernel original name, variable1 type, variable1 name, variable2 type, variable2 name ....}
inline std::unordered_map<std::string, std::vector<std::string>> g_kernelVarMap;

/**
 * @brief Validates path and accessibility
 * @param path Null-terminated C-style path string
 * @param needCheck true means will show error message, false means only check realpath, no error message
 * @return PathStatus enum (EXIST/READ/WRITE/NOT_EXIST)
 */
PathStatus PathCheck(const char* path, bool needCheck);

/**
 * @brief Resolves relative paths and returns absolute path
 * @param path Input path
 * @return Canonical absolute path
 */
std::string CheckAndGetFullPath(const char* path);
std::string CheckAndGetFullPath(const std::string& path);

/**
 * @brief Extracts directory path from full path
 * @param filePath Input path (e.g., "/home/user/doc.txt")
 * @return Directory component (e.g., "/home/user/")
 */
std::string GetFilePath(const std::string& filePath);

/**
 * @brief check whether srcStr starts with specific prefix
 * @param srcStr input string
 * @param prefix contents to be searched in srcStr
 * @return true means starts with prefix
 */
bool StartsWith(const std::string& srcStr, const std::string& prefix);

/**
 * @brief check whether srcStr ends with specific suffix
 * @param srcStr input string
 * @param suffix contents to be searched in srcStr
 * @return true means ends with suffix
 */
bool EndsWith(const std::string& srcStr, const std::string& suffix);

/**
 * @brief get time stamp
 * @return time stamp (e.g., "20250807113201")
 */
std::string GenerateTimestamp();

/**
 * @brief Generate tmpPath / logPath based on users input
 * @param userFilePath path passed by user to store logs (e.g., "/tmp")
 * @param srcFilePath path for source file (e.g., "/home/user/add_custom.cpp")
 * @param timeTag time stamp (e.g., "20250807113201")
 * @param folderTypeName folder type name  ("log" / "temp")
 * @return folder path (e.g., "/tmp/asc_plugin/add_custom_[timestamp]/log")
 */
std::string GetTempFolder(const std::string& userFilePath, const std::string& srcFilePath, const std::string& timeTag,
    const std::string& folderTypeName);

/**
 * @brief Create directory by mkdir(c++)
 * @param fileStatus status of file
 * @param dirPath directory path (e.g., "/tmp/asc_plugin/add_custom/20250807113201_3770950/log")
 * @return return SUCCESS if execute mkdir successfully
 */
int32_t ExecMkdir(struct stat& fileStatus, const char* dirPath);

/**
 * @brief Create directory recursively
 * @param dirPath directory path (e.g., "/tmp/asc_plugin/add_custom/20250807113201_3770950/log")
 * @return return SUCCESS if create directory successfully
 */
int32_t CreateDirectory(const std::string& dirPath);

/**
 * @brief Executes system command and captures output
 * @param command Shell command to execute
 * @return Pair containing <command_output, exit_status>
 */
std::pair<std::string, int32_t> ExecuteCommand(const char *command);

/**
 * @brief Executes compile command
 * @param command Shell command to execute
 * @return return SUCCESS if compile successfully
 */
int32_t ExecuteCompile(const std::string &cmd);

/**
 * @brief Split a string into multiple lines of text based on newline characters
 * @param str the input string
 * @return return the vector<std::string> of the input string
 */
std::vector<std::string> SplitLines(const std::string& str);

/**
 * @brief Get kernel type based on core ratio given by bisheng. If invalid, return defaultKtype.
 * @param ratio core ratio given by bisheng
 * @param defaultKType kernel type to be returned when core ratio is invalid
 * @return return kernel type
 */
KernelMetaType GetBishengKTypeByCoreRatio(const CoreRatio& ratio,
    const KernelMetaType& defaultKType =  KernelMetaType::KERNEL_TYPE_MAX);

/**
 * @brief Get kernel type from the kernel type set
 * @param kTypeSet kernel type to be returned when core ratio is invalid
 * @return return kernel type
 */
KernelMetaType ExtractKernelType(const std::unordered_set<KernelMetaType> kTypeSet);

/**
 * @brief Convert string to upper case
 * @param str input string
 * @return return string with all upper case
 */
std::string ToUpper(const std::string& str);

} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_UTILS_H__