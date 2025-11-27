cmake_minimum_required(VERSION 3.16)

# 从 version.info 读取版本号
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.info" VERSION_LINE REGEX "^Version=")
string(REGEX REPLACE "Version=(.*)" "\\1" PROJECT_VERSION "${VERSION_LINE}")
if(NOT PROJECT_VERSION)
    message(FATAL_ERROR "Failed to read version from version.info")
endif()

function(parse_semantic_version version_str out_major out_minor out_patch out_prerelease out_version_num)
    # 解析主/次/补丁版本（格式：MAJOR.MINOR.PATCH）
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)(-.*)?" VERSION_MATCH "${version_str}")
    if(NOT VERSION_MATCH)
        message(FATAL_ERROR "Invalid version format: ${version_str}. Expected x.y.z[-prerelease]")
    endif()

    set(major ${CMAKE_MATCH_1})
    set(minor ${CMAKE_MATCH_2})
    set(patch ${CMAKE_MATCH_3})
    set(prerelease ${CMAKE_MATCH_4})

    # 移除预发布版本前的 '-'
    string(REGEX REPLACE "^-" "" prerelease "${prerelease}")

    # 计算 base_value = major*10^7 + minor*10^5 + patch*10^3
    math(EXPR base_value "${major} * 10000000 + ${minor} * 100000 + ${patch} * 1000")
    set(version_num ${base_value})

    # 处理预发布版本（alpha/beta/rc）
    if(prerelease)
        string(REGEX MATCH "^(alpha|beta|rc)\\.([0-9]+)" PR_MATCH "${prerelease}")
        if(NOT PR_MATCH)
            message(FATAL_ERROR "Invalid prerelease format: ${prerelease}. Use 'alpha.N' / 'beta.N' / 'rc.N'")
        endif()

        set(pr_type ${CMAKE_MATCH_1})
        set(pr_num ${CMAKE_MATCH_2})

        # 预发布权重（alpha 最低，rc 最高）
        if(pr_type STREQUAL "alpha")
            set(weight 300)
        elseif(pr_type STREQUAL "beta")
            set(weight 200)
        elseif(pr_type STREQUAL "rc")
            set(weight 100)
        endif()

        # 计算带预发布的版本数值
        math(EXPR version_num "${base_value} - ${weight} + ${pr_num}")
    endif()

    # 输出变量（传递到函数外部）
    set(${out_major} ${major} PARENT_SCOPE)
    set(${out_minor} ${minor} PARENT_SCOPE)
    set(${out_patch} ${patch} PARENT_SCOPE)
    set(${out_prerelease} "${prerelease}" PARENT_SCOPE)
    set(${out_version_num} ${version_num} PARENT_SCOPE)
endfunction()

# --------------------------
# 3. 调用函数解析版本号
# --------------------------
parse_semantic_version(
    ${PROJECT_VERSION}
    MAJOR
    MINOR
    PATCH
    PRERELEASE
    VERSION_NUM
)

# -------------------------
# 4. 生成 asc_devkit_version.h
# -------------------------
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/asc-devkit/asc_devkit_version.h
    @ONLY
)

set(VERSION_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/asc-devkit/asc_devkit_version.h
)

install(FILES ${VERSION_FILES}
    DESTINATION asc-devkit
)
