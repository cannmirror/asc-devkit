# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../common/src/ SRC_COMMON)

add_executable(ascendfe
  ${SRC}
  ${SRC_COMMON}
)

target_include_directories(ascendfe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/
)

target_link_libraries(ascendfe PRIVATE
    $<BUILD_INTERFACE:intf_pub_cxx17>
    libclangTooling_static
    libclangFrontend_static
    libclangParse_static
    libclangSerialization_static
    libclangASTMatchers_static
    libclangSema_static
    libclangSupport_static
    libclangAST_static
    libclangAnalysis_static
    libclangDriver_static
    libclangBasic_static
    libclangEdit_static
    libclangLex_static
    libLLVMWindowsDriver_static
    libLLVMFrontendOpenMP_static
    libLLVMAnalysis_static
    libLLVMProfileData_static
    libLLVMCore_static
    libLLVMBitstreamReader_static
    libLLVMMCParser_static
    libLLVMMC_static
    libLLVMBinaryFormat_static
    libLLVMOption_static
    libLLVMSupport_static
    libLLVMDemangle_static
    c_sec_static
    json
    pthread
    dl
    z
)

target_compile_options(ascendfe PRIVATE
    -fPIE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -fno-strict-aliasing
    -fno-common
    -Wall
    -Wextra
    -Wfloat-equal
    -D_FORTIFY_SOURCE=2
    -Os
    -DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_PROGRAM}
    -DCMAKE_Cxx_COMPILER_LAUNCHER=${CCACHE_PROGRAM}
)

target_link_options(ascendfe PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -Wl,-Bsymbolic
    -Wl,--exclude-libs=ALL
    -Wl,--gc-sections
)

install(TARGETS ascendfe
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)