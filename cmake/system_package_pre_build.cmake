# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(_ASC_DEVKIT_PACKAGE_NAME "asc-devkit")
set(_ASC_DEVKIT_REQUIRED_VARIABLES
    CPACK_ASC_DEVKIT_ARCH
    CPACK_ASC_DEVKIT_BINARY_DIR
    CPACK_ASC_DEVKIT_SOURCE_DIR
    CPACK_ASC_DEVKIT_CANN_CMAKE_DIR
    CPACK_ASC_DEVKIT_INSTALL_PREFIX
)
foreach(_ASC_DEVKIT_REQUIRED_VARIABLE IN LISTS _ASC_DEVKIT_REQUIRED_VARIABLES)
    if(NOT DEFINED ${_ASC_DEVKIT_REQUIRED_VARIABLE} OR "${${_ASC_DEVKIT_REQUIRED_VARIABLE}}" STREQUAL "")
        message(FATAL_ERROR "Missing required CPack variable: ${_ASC_DEVKIT_REQUIRED_VARIABLE}")
    endif()
endforeach()

set(_ASC_DEVKIT_ARCH "${CPACK_ASC_DEVKIT_ARCH}")
set(_ASC_DEVKIT_BINARY_DIR "${CPACK_ASC_DEVKIT_BINARY_DIR}")
set(_ASC_DEVKIT_SOURCE_DIR "${CPACK_ASC_DEVKIT_SOURCE_DIR}")
set(_ASC_DEVKIT_CANN_CMAKE_DIR "${CPACK_ASC_DEVKIT_CANN_CMAKE_DIR}")
set(_ASC_DEVKIT_INSTALL_PREFIX "${CPACK_ASC_DEVKIT_INSTALL_PREFIX}")
set(_ASC_DEVKIT_DEVICE_STAGING "${_ASC_DEVKIT_BINARY_DIR}/_CPack_Packages/makeself_staging")
set(_ASC_DEVKIT_SYSTEM_PACKAGE_WORK "${_ASC_DEVKIT_BINARY_DIR}/system_package_work")

set(_ASC_DEVKIT_PACKAGE_ROOT "")
set(_ASC_DEVKIT_PACKAGE_ROOT_CANDIDATES "")
if(DEFINED CPACK_TEMPORARY_DIRECTORY)
    string(REGEX REPLACE "^/" "" _ASC_DEVKIT_REL_INSTALL_PREFIX "${_ASC_DEVKIT_INSTALL_PREFIX}")
    list(APPEND _ASC_DEVKIT_PACKAGE_ROOT_CANDIDATES
        "${CPACK_TEMPORARY_DIRECTORY}/${_ASC_DEVKIT_PACKAGE_NAME}/${_ASC_DEVKIT_REL_INSTALL_PREFIX}"
        "${CPACK_TEMPORARY_DIRECTORY}/${_ASC_DEVKIT_PACKAGE_NAME}${_ASC_DEVKIT_INSTALL_PREFIX}"
        "${CPACK_TEMPORARY_DIRECTORY}/${_ASC_DEVKIT_REL_INSTALL_PREFIX}"
        "${CPACK_TEMPORARY_DIRECTORY}${_ASC_DEVKIT_INSTALL_PREFIX}"
        "${CPACK_TEMPORARY_DIRECTORY}"
    )
    if(DEFINED CPACK_INSTALL_PREFIX)
        string(REGEX REPLACE "^/" "" _ASC_DEVKIT_REL_CPACK_INSTALL_PREFIX "${CPACK_INSTALL_PREFIX}")
        list(APPEND _ASC_DEVKIT_PACKAGE_ROOT_CANDIDATES
            "${CPACK_TEMPORARY_DIRECTORY}/${_ASC_DEVKIT_PACKAGE_NAME}/${_ASC_DEVKIT_REL_CPACK_INSTALL_PREFIX}"
            "${CPACK_TEMPORARY_DIRECTORY}/${_ASC_DEVKIT_PACKAGE_NAME}${CPACK_INSTALL_PREFIX}"
            "${CPACK_TEMPORARY_DIRECTORY}/${_ASC_DEVKIT_REL_CPACK_INSTALL_PREFIX}"
            "${CPACK_TEMPORARY_DIRECTORY}${CPACK_INSTALL_PREFIX}"
        )
    endif()
endif()

if(DEFINED CMAKE_INSTALL_PREFIX)
    list(APPEND _ASC_DEVKIT_PACKAGE_ROOT_CANDIDATES "${CMAKE_INSTALL_PREFIX}")
endif()

foreach(_ASC_DEVKIT_PACKAGE_ROOT_CANDIDATE IN LISTS _ASC_DEVKIT_PACKAGE_ROOT_CANDIDATES)
    if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT_CANDIDATE}/share/info/${_ASC_DEVKIT_PACKAGE_NAME}/version.info")
        set(_ASC_DEVKIT_PACKAGE_ROOT "${_ASC_DEVKIT_PACKAGE_ROOT_CANDIDATE}")
        break()
    endif()
endforeach()

if(NOT _ASC_DEVKIT_PACKAGE_ROOT)
    foreach(_ASC_DEVKIT_PACKAGE_ROOT_CANDIDATE IN LISTS _ASC_DEVKIT_PACKAGE_ROOT_CANDIDATES)
        if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT_CANDIDATE}")
            set(_ASC_DEVKIT_PACKAGE_ROOT "${_ASC_DEVKIT_PACKAGE_ROOT_CANDIDATE}")
            break()
        endif()
    endforeach()
endif()

if(NOT _ASC_DEVKIT_PACKAGE_ROOT)
    message(FATAL_ERROR "Failed to locate asc-devkit system package staging root.")
endif()

if(EXISTS "${_ASC_DEVKIT_DEVICE_STAGING}")
    file(COPY "${_ASC_DEVKIT_DEVICE_STAGING}/" DESTINATION "${_ASC_DEVKIT_PACKAGE_ROOT}")
endif()

file(REMOVE_RECURSE "${_ASC_DEVKIT_SYSTEM_PACKAGE_WORK}/_CPack_Packages")
file(MAKE_DIRECTORY "${_ASC_DEVKIT_SYSTEM_PACKAGE_WORK}/_CPack_Packages")
file(CREATE_LINK
    "${_ASC_DEVKIT_PACKAGE_ROOT}"
    "${_ASC_DEVKIT_SYSTEM_PACKAGE_WORK}/_CPack_Packages/makeself_staging"
    SYMBOLIC
)

execute_process(
    COMMAND python3
        "${_ASC_DEVKIT_CANN_CMAKE_DIR}/scripts/package/package.py"
        --pkg_name "${_ASC_DEVKIT_PACKAGE_NAME}"
        --chip_name ""
        --os_arch "linux-${_ASC_DEVKIT_ARCH}"
        --version_dir ""
        --delivery_dir "${_ASC_DEVKIT_SYSTEM_PACKAGE_WORK}"
        --source_dir "${_ASC_DEVKIT_SOURCE_DIR}"
    WORKING_DIRECTORY "${_ASC_DEVKIT_BINARY_DIR}"
    RESULT_VARIABLE _ASC_DEVKIT_PACKAGE_RESULT
    OUTPUT_VARIABLE _ASC_DEVKIT_PACKAGE_OUTPUT
    ERROR_VARIABLE _ASC_DEVKIT_PACKAGE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _ASC_DEVKIT_PACKAGE_RESULT EQUAL 0)
    message(FATAL_ERROR "Generate asc-devkit system package metadata failed: ${_ASC_DEVKIT_PACKAGE_OUTPUT} ${_ASC_DEVKIT_PACKAGE_ERROR}")
endif()

if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/conf/path.cfg")
    execute_process(COMMAND chmod 440 "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/conf/path.cfg")
endif()
if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/bin")
    execute_process(COMMAND find "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/bin" -type f -exec chmod 550 {} +)
endif()
if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/lib64")
    execute_process(COMMAND find "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/lib64" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/devlib")
    execute_process(COMMAND find "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/devlib" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/include")
    execute_process(COMMAND find "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/include" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/lib64/device/lib64")
    execute_process(COMMAND find "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/lib64/device/lib64" -type f -exec chmod 550 {} +)
endif()
if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/pkg_inc")
    execute_process(COMMAND find "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/pkg_inc" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/include/version")
    execute_process(COMMAND find "${_ASC_DEVKIT_PACKAGE_ROOT}/${_ASC_DEVKIT_ARCH}-linux/include/version" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${_ASC_DEVKIT_PACKAGE_ROOT}/opp/built-in/op_impl")
    execute_process(COMMAND find "${_ASC_DEVKIT_PACKAGE_ROOT}/opp/built-in/op_impl" -type f -exec chmod 440 {} +)
endif()
