# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

#### CPACK to package run #####
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Detected architecture: x86_64")
    set(ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    message(STATUS "Detected architecture: ARM64")
    set(ARCH aarch64)
else ()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()
# 打印路径
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/makeself-fetch.cmake)

set(script_prefix ${CMAKE_SOURCE_DIR}/scripts/package/scripts/)
install(DIRECTORY ${script_prefix}/
    DESTINATION share/info/asc-devkit/script
    COMPONENT asc-devkit
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)
set(SCRIPTS_FILES
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.bash
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.csh
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.fish
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/py/merge_binary_info_config.py
)

install(FILES ${SCRIPTS_FILES}
    DESTINATION share/info/asc-devkit/script COMPONENT asc-devkit
)

set(COMMON_FILES
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/install_common_parser.sh
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func_v2.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func_v3.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_installer.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/script_operator.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_cfg.inc
)

set(PACKAGE_FILES
    ${COMMON_FILES}
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/multi_version.inc
)
set(LATEST_MANGER_FILES
    ${COMMON_FILES}
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
    ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
)
set(CONF_FILES 
    ${CMAKE_SOURCE_DIR}/scripts/package/common/cfg/path.cfg
)
install(FILES ${CONF_FILES}
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf COMPONENT asc-devkit
)
install(FILES ${PACKAGE_FILES}
   DESTINATION share/info/asc-devkit/script COMPONENT asc-devkit
)
install(FILES ${CMAKE_BINARY_DIR}/version.asc-devkit.info
    DESTINATION share/info/asc-devkit
    RENAME version.info
    COMPONENT asc-devkit
)

set(HCCL_CC_BUILD_DIR ${CMAKE_BINARY_DIR}/impl/adv_api/detail/hccl/cc)
install(FILES "${HCCL_CC_BUILD_DIR}/src/libmc2_client.so" "${HCCL_CC_BUILD_DIR}/src/common/hcomm_dlsym/libmc2_compat.so"
    DESTINATION "hccl/lib64" COMPONENT asc-devkit OPTIONAL)
install(FILES "${HCCL_CC_BUILD_DIR}/src/libmc2_server.json"
    DESTINATION "hccl/built-in/data/op/aicpu" COMPONENT asc-devkit OPTIONAL)

function(install_mc2_runtime_staging_link src_rel_path dst_rel_path required)
    install(CODE "
set(_src_rel_path \"${src_rel_path}\")
set(_dst_rel_path \"${dst_rel_path}\")
set(_required \"${required}\")
set(_install_prefix \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\")
file(TO_CMAKE_PATH \"\${_install_prefix}\" _install_prefix)
set(_src_path \"\${_install_prefix}/\${_src_rel_path}\")
set(_dst_path \"\${_install_prefix}/\${_dst_rel_path}\")
if(EXISTS \"\${_dst_path}\" AND IS_DIRECTORY \"\${_dst_path}\" AND NOT IS_SYMLINK \"\${_dst_path}\")
    message(FATAL_ERROR \"MC2 runtime staging path \${_dst_rel_path} is a directory, cannot create file softlink.\")
elseif(EXISTS \"\${_dst_path}\")
    message(STATUS \"MC2 runtime staging path \${_dst_rel_path} already exists, skip.\")
else()
    if(IS_SYMLINK \"\${_dst_path}\")
        file(REMOVE \"\${_dst_path}\")
    endif()
    if(NOT EXISTS \"\${_src_path}\")
        if(_required)
            message(FATAL_ERROR \"MC2 runtime staging source \${_src_rel_path} does not exist.\")
        else()
            message(WARNING \"MC2 runtime staging source \${_src_rel_path} does not exist, skip creating \${_dst_rel_path}.\")
        endif()
    else()
        get_filename_component(_dst_dir \"\${_dst_path}\" DIRECTORY)
        file(MAKE_DIRECTORY \"\${_dst_dir}\")
        file(RELATIVE_PATH _link_target \"\${_dst_dir}\" \"\${_src_path}\")
        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" -E create_symlink \"\${_link_target}\" \"\${_dst_path}\"
            RESULT_VARIABLE _link_ret
        )
        if(NOT _link_ret EQUAL 0)
            message(FATAL_ERROR \"failed to create MC2 runtime staging link \${_dst_rel_path} -> \${_src_rel_path}.\")
        endif()
        message(STATUS \"Created MC2 runtime staging link \${_dst_rel_path} -> \${_src_rel_path}.\")
    endif()
endif()
" COMPONENT asc-devkit)
endfunction()

install_mc2_runtime_staging_link("hccl/include/hccl/hccl.h" "${CMAKE_SYSTEM_PROCESSOR}-linux/include/hccl/hccl.h" TRUE)
install_mc2_runtime_staging_link("hccl/include/hccl/hccl_mc2.h" "${CMAKE_SYSTEM_PROCESSOR}-linux/include/hccl/hccl_mc2.h" TRUE)
install_mc2_runtime_staging_link("hccl/lib64/libmc2_client.so" "${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/libmc2_client.so" TRUE)
install_mc2_runtime_staging_link("hccl/lib64/libmc2_compat.so" "${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/libmc2_compat.so" TRUE)
install_mc2_runtime_staging_link("hccl/built-in/data/op/aicpu/libmc2_server.json" "opp/built-in/op_impl/aicpu/config/libmc2_server.json" TRUE)
install_mc2_runtime_staging_link("hccl/Ascend/aicpu/mc2_server.tar.gz" "opp/built-in/op_impl/aicpu/kernel/mc2_server.tar.gz" FALSE)
# ============= CPack =============
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")

set(CPACK_INSTALL_PREFIX "/")

set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
set(CPACK_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CPACK_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CPACK_MAKESELF_PATH "${MAKESELF_PATH}")
set(CPACK_ARCH "${ARCH}")
set(CPACK_SET_DESTDIR ON)
set(CPACK_GENERATOR External)
set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself.cmake")
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CPACK_VERSION "${VERSION}")

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
include(CPack)
