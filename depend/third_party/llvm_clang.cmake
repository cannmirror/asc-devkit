# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

unset(LLVM_CLANG_FOUND CACHE)
unset(LLVM_CLANG_INCLUDE CACHE)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(LLVM_CLANG_NAME "llvm-15.0.4")
set(LLVM_CLANG_DOWNLOAD_PATH ${CANN_3RD_LIB_PATH}/pkg)
set(LLVM_CLANG_INSTALL_PATH ${CANN_3RD_LIB_PATH}/llvm_clang)
set(LLVM_CLANG_SOURCE_PATH ${CANN_3RD_LIB_PATH}/${LLVM_CLANG_NAME}/llvm-project-llvmorg-15.0.4)

find_path(LLVM_CLANG_INCLUDE
        NAMES clang-c/Index.h
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/include)

find_library(LIBCLANGTOOLING_STATIC_LIBRARY
        NAMES libclangTooling.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGASTMATCHERS_STATIC_LIBRARY
        NAMES libclangASTMatchers.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGFRONTEND_STATIC_LIBRARY
        NAMES libclangFrontend.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGDRIVER_STATIC_LIBRARY
        NAMES libclangDriver.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMWINDOWSDRIVER_STATIC_LIBRARY
        NAMES libLLVMWindowsDriver.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMOPTION_STATIC_LIBRARY
        NAMES libLLVMOption.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGPARSE_STATIC_LIBRARY
        NAMES libclangParse.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGSERIALIZATION_STATIC_LIBRARY
        NAMES libclangSerialization.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGSEMA_STATIC_LIBRARY
        NAMES libclangSema.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGSUPPORT_STATIC_LIBRARY
        NAMES libclangSupport.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGEDIT_STATIC_LIBRARY
        NAMES libclangEdit.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGANALYSIS_STATIC_LIBRARY
        NAMES libclangAnalysis.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGAST_STATIC_LIBRARY
        NAMES libclangAST.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGLEX_STATIC_LIBRARY
        NAMES libclangLex.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBCLANGBASIC_STATIC_LIBRARY
        NAMES libclangBasic.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMFRONTENDOPENMP_STATIC_LIBRARY
        NAMES libLLVMFrontendOpenMP.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMANALYSIS_STATIC_LIBRARY
        NAMES libLLVMAnalysis.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMPROFILEDATA_STATIC_LIBRARY
        NAMES libLLVMProfileData.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMCORE_STATIC_LIBRARY
        NAMES libLLVMCore.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMBITSTREAMREADER_STATIC_LIBRARY
        NAMES libLLVMBitstreamReader.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMMCPARSER_STATIC_LIBRARY
        NAMES libLLVMMCParser.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMMC_STATIC_LIBRARY
        NAMES libLLVMMC.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMBINARYFORMAT_STATIC_LIBRARY
        NAMES libLLVMBinaryFormat.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMSUPPORT_STATIC_LIBRARY
        NAMES libLLVMSupport.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

find_library(LIBLLVMDEMANGLE_STATIC_LIBRARY
        NAMES libLLVMDemangle.a
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${LLVM_CLANG_INSTALL_PATH}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(llvm_clang
        FOUND_VAR
        LLVM_CLANG_FOUND
        REQUIRED_VARS
        LLVM_CLANG_INCLUDE
        LIBCLANGFRONTEND_STATIC_LIBRARY
        LIBCLANGTOOLING_STATIC_LIBRARY
        LIBCLANGASTMATCHERS_STATIC_LIBRARY
        LIBCLANGDRIVER_STATIC_LIBRARY
        LIBLLVMWINDOWSDRIVER_STATIC_LIBRARY
        LIBLLVMOPTION_STATIC_LIBRARY
        LIBCLANGPARSE_STATIC_LIBRARY
        LIBCLANGSERIALIZATION_STATIC_LIBRARY
        LIBCLANGSEMA_STATIC_LIBRARY
        LIBCLANGSUPPORT_STATIC_LIBRARY
        LIBCLANGEDIT_STATIC_LIBRARY
        LIBCLANGANALYSIS_STATIC_LIBRARY
        LIBCLANGAST_STATIC_LIBRARY
        LIBCLANGLEX_STATIC_LIBRARY
        LIBCLANGBASIC_STATIC_LIBRARY
        LIBLLVMFRONTENDOPENMP_STATIC_LIBRARY
        LIBLLVMANALYSIS_STATIC_LIBRARY
        LIBLLVMPROFILEDATA_STATIC_LIBRARY
        LIBLLVMCORE_STATIC_LIBRARY
        LIBLLVMBITSTREAMREADER_STATIC_LIBRARY
        LIBLLVMMCPARSER_STATIC_LIBRARY
        LIBLLVMMC_STATIC_LIBRARY
        LIBLLVMBINARYFORMAT_STATIC_LIBRARY
        LIBLLVMSUPPORT_STATIC_LIBRARY
        LIBLLVMDEMANGLE_STATIC_LIBRARY
        )

if(LLVM_CLANG_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message("llvm_clang found in ${LLVM_CLANG_INSTALL_PATH}, and not force rebuild cann third_party")
    add_library(llvm_clang INTERFACE)
else()
    if (NOT EXISTS "${LLVM_CLANG_SOURCE_PATH}")
        message(info, "download llvm code, don't use llvm cache.")
        set(LLVM_FILE llvm-project-llvmorg-15.0.4.tar.gz)
        set(LLVM_CLANG_URL "https://gitcode.com/cann-src-third-party/llvm/releases/download/15.0.4/${LLVM_FILE}")
        message(STATUS "Downloading ${LLVM_CLANG_NAME} from ${LLVM_CLANG_URL}")
        include(FetchContent)
        FetchContent_Declare(
            ${LLVM_CLANG_NAME}
            URL ${LLVM_CLANG_URL}
            TLS_VERIFY FALSE
            URL_HASH SHA256=e24b4d3bf7821dcb1c901d1e09096c1f88fb00095c5a6ef893baab4836975e52
            DOWNLOAD_DIR ${LLVM_CLANG_DOWNLOAD_PATH}
            SOURCE_DIR ${LLVM_CLANG_SOURCE_PATH}
        )
        FetchContent_MakeAvailable(${LLVM_CLANG_NAME})


        # file(DOWNLOAD
        #     ${LLVM_CLANG_URL}
        #     ${LLVM_CLANG_SOURCE_PATH}/${LLVM_FILE}
        # #     URL_HASH SHA256=e24b4d3bf7821dcb1c901d1e09096c1f88fb00095c5a6ef893baab4836975e52
        #     SHOW_PROGRESS
        # )
        # execute_process(
        #     COMMAND mkdir -p ${LLVM_CLANG_SOURCE_PATH}
        #     COMMAND chmod 755 -R ${LLVM_CLANG_SOURCE_PATH}
        #     COMMAND tar -xf ${LLVM_CLANG_SOURCE_PATH}/${LLVM_FILE} --overwrite --strip-components=1 -C ${LLVM_CLANG_SOURCE_PATH}
        # )
    else ()
        message(info, "use llvm cache,do not need download llvm code.")
    endif()

    set(LLVM_C_COMPILE_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -D_FORTIFY_SOURCE=2 -fvisibility=hidden -fstack-protector-all -fPIE")
    set(LLVM_CXX_COMPILE_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -D_FORTIFY_SOURCE=2 -fvisibility=hidden -fstack-protector-all -fPIE")
    set(LLVM_EXE_SAFE_LINK_FLAGS "-Wl,-z,relro,-z,now,-z,noexecstack -pie")
    set(LLVM_SHARED_SAFE_LINK_FLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")

    set(_LLVM_C_COMPILER gcc)
    set(_LLVM_CXX_COMPILER g++)
    set(CMAKE_GENERATOR "Unix Makefiles")

    include(ExternalProject)
    ExternalProject_Add(llvm_clang
                        SOURCE_DIR ${LLVM_CLANG_SOURCE_PATH}
                        CONFIGURE_COMMAND ${CMAKE_COMMAND}
                            -G ${CMAKE_GENERATOR}
                            -DLLVM_CCACHE_BUILD=ON
                            -DLLVM_ENABLE_PROJECTS=clang
                            -DLIBCLANG_BUILD_STATIC=ON
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DCMAKE_BUILD_TYPE=Release
                            -DLLVM_ENABLE_ASSERTIONS=OFF
                            -DCLANG_DEFAULT_LINKER=lld
                            -DBUILD_SHARED_LIBS=OFF
                            -DLLVM_ENABLE_PIC=ON
                            -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=ON
                            -DLLVM_TARGETS_TO_BUILD=AArch64::X86::ARM
                            -DCMAKE_C_COMPILER=${_LLVM_C_COMPILER}
                            -DCMAKE_CXX_COMPILER=${_LLVM_CXX_COMPILER}
                            -DLLVM_ENABLE_LIBPFM=OFF
                            -DLLVM_ENABLE_RTTI=ON
                            -DCMAKE_C_FLAGS=${LLVM_C_COMPILE_FLAGS}
                            -DCMAKE_CXX_FLAGS=${LLVM_CXX_COMPILE_FLAGS}
                            -DCMAKE_EXE_LINKER_FLAGS=${LLVM_EXE_SAFE_LINK_FLAGS}
                            -DCMAKE_SHARED_LINKER_FLAGS=${LLVM_SHARED_SAFE_LINK_FLAGS}
                            -DCMAKE_INSTALL_PREFIX=${LLVM_CLANG_INSTALL_PATH}
                            -DLLVM_ENABLE_BINDINGS=OFF
                            -DLLVM_ENABLE_ZSTD=OFF
                            -DLLVM_ENABLE_TERMINFO=OFF
                            <SOURCE_DIR>/llvm
                        BUILD_COMMAND cmake --build . --target install-clangFrontend -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangTooling -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangDriver -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMWindowsDriver -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMOption -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangParse -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangSerialization -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangSema -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangSupport -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangEdit -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangAnalysis -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangAST -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangASTMatchers -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangLex -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clangBasic -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMFrontendOpenMP -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMAnalysis -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMProfileData -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMCore -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMBitstreamReader -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMMCParser -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMMC -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMBinaryFormat -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMSupport -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-LLVMDemangle -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-llvm-headers -- -Orecurse $(MAKE)
                            COMMAND cmake --build . --target install-clang-headers -- -Orecurse $(MAKE)
                        INSTALL_COMMAND cmake --build . --target install-libclang-headers -- -Orecurse $(MAKE)
                        LIST_SEPARATOR ::
                        )
endif()

if (NOT EXISTS ${LLVM_CLANG_INSTALL_PATH}/include)
  file(MAKE_DIRECTORY "${LLVM_CLANG_INSTALL_PATH}/include")
endif ()

if(NOT TARGET libclangTooling_static)
    add_library(libclangTooling_static STATIC IMPORTED)
    add_dependencies(libclangTooling_static llvm_clang)
    set_target_properties(libclangTooling_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangTooling.a
        )
endif()

if(NOT TARGET libclangASTMatchers_static)
    add_library(libclangASTMatchers_static STATIC IMPORTED)
    add_dependencies(libclangASTMatchers_static llvm_clang)
    set_target_properties(libclangASTMatchers_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangASTMatchers.a
        )
endif()

if(NOT TARGET libclangFrontend_static)
    add_library(libclangFrontend_static STATIC IMPORTED)
    add_dependencies(libclangFrontend_static llvm_clang)
    set_target_properties(libclangFrontend_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangFrontend.a
        )
endif()

if(NOT TARGET libclangDriver_static)
    add_library(libclangDriver_static STATIC IMPORTED)
    add_dependencies(libclangDriver_static llvm_clang)
    set_target_properties(libclangDriver_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangDriver.a
        )
endif()

if(NOT TARGET libLLVMWindowsDriver_static)
    add_library(libLLVMWindowsDriver_static STATIC IMPORTED)
    add_dependencies(libLLVMWindowsDriver_static llvm_clang)
    set_target_properties(libLLVMWindowsDriver_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMWindowsDriver.a
        )
endif()

if(NOT TARGET libLLVMOption_static)
    add_library(libLLVMOption_static STATIC IMPORTED)
    add_dependencies(libLLVMOption_static llvm_clang)
    set_target_properties(libLLVMOption_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMOption.a
        )
endif()

if(NOT TARGET libclangParse_static)
    add_library(libclangParse_static STATIC IMPORTED)
    add_dependencies(libclangParse_static llvm_clang)
    set_target_properties(libclangParse_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangParse.a
        )
endif()

if(NOT TARGET libclangSerialization_static)
    add_library(libclangSerialization_static STATIC IMPORTED)
    add_dependencies(libclangSerialization_static llvm_clang)
    set_target_properties(libclangSerialization_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangSerialization.a
        )
endif()

if(NOT TARGET libclangSema_static)
    add_library(libclangSema_static STATIC IMPORTED)
    add_dependencies(libclangSema_static llvm_clang)
    set_target_properties(libclangSema_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangSema.a
        )
endif()

if(NOT TARGET libclangSupport_static)
    add_library(libclangSupport_static STATIC IMPORTED)
    add_dependencies(libclangSupport_static llvm_clang)
    set_target_properties(libclangSupport_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangSupport.a
        )
endif()

if(NOT TARGET libclangEdit_static)
    add_library(libclangEdit_static STATIC IMPORTED)
    add_dependencies(libclangEdit_static llvm_clang)
    set_target_properties(libclangEdit_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangEdit.a
        )
endif()

if(NOT TARGET libclangAnalysis_static)
    add_library(libclangAnalysis_static STATIC IMPORTED)
    add_dependencies(libclangAnalysis_static llvm_clang)
    set_target_properties(libclangAnalysis_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangAnalysis.a
        )
endif()

if(NOT TARGET libclangAST_static)
    add_library(libclangAST_static STATIC IMPORTED)
    add_dependencies(libclangAST_static llvm_clang)
    set_target_properties(libclangAST_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangAST.a
        )
endif()

if(NOT TARGET libclangLex_static)
    add_library(libclangLex_static STATIC IMPORTED)
    add_dependencies(libclangLex_static llvm_clang)
    set_target_properties(libclangLex_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangLex.a
        )
endif()

if(NOT TARGET libclangBasic_static)
    add_library(libclangBasic_static STATIC IMPORTED)
    add_dependencies(libclangBasic_static llvm_clang)
    set_target_properties(libclangBasic_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libclangBasic.a
        )
endif()

if(NOT TARGET libLLVMFrontendOpenMP_static)
    add_library(libLLVMFrontendOpenMP_static STATIC IMPORTED)
    add_dependencies(libLLVMFrontendOpenMP_static llvm_clang)
    set_target_properties(libLLVMFrontendOpenMP_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMFrontendOpenMP.a
        )
endif()

if(NOT TARGET libLLVMAnalysis_static)
    add_library(libLLVMAnalysis_static STATIC IMPORTED)
    add_dependencies(libLLVMAnalysis_static llvm_clang)
    set_target_properties(libLLVMAnalysis_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMAnalysis.a
        )
endif()

if(NOT TARGET libLLVMProfileData_static)
    add_library(libLLVMProfileData_static STATIC IMPORTED)
    add_dependencies(libLLVMProfileData_static llvm_clang)
    set_target_properties(libLLVMProfileData_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMProfileData.a
        )
endif()

if(NOT TARGET libLLVMCore_static)
    add_library(libLLVMCore_static STATIC IMPORTED)
    add_dependencies(libLLVMCore_static llvm_clang)
    set_target_properties(libLLVMCore_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMCore.a
        )
endif()

if(NOT TARGET libLLVMBitstreamReader_static)
    add_library(libLLVMBitstreamReader_static STATIC IMPORTED)
    add_dependencies(libLLVMBitstreamReader_static llvm_clang)
    set_target_properties(libLLVMBitstreamReader_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMBitstreamReader.a
        )
endif()

if(NOT TARGET libLLVMMCParser_static)
    add_library(libLLVMMCParser_static STATIC IMPORTED)
    add_dependencies(libLLVMMCParser_static llvm_clang)
    set_target_properties(libLLVMMCParser_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMMCParser.a
        )
endif()

if(NOT TARGET libLLVMMC_static)
    add_library(libLLVMMC_static STATIC IMPORTED)
    add_dependencies(libLLVMMC_static llvm_clang)
    set_target_properties(libLLVMMC_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMMC.a
        )
endif()

if(NOT TARGET libLLVMBinaryFormat_static)
    add_library(libLLVMBinaryFormat_static STATIC IMPORTED)
    add_dependencies(libLLVMBinaryFormat_static llvm_clang)
    set_target_properties(libLLVMBinaryFormat_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMBinaryFormat.a
        )
endif()

if(NOT TARGET libLLVMSupport_static)
    add_library(libLLVMSupport_static STATIC IMPORTED)
    add_dependencies(libLLVMSupport_static llvm_clang)
    set_target_properties(libLLVMSupport_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMSupport.a
        )
endif()

if(NOT TARGET libLLVMDemangle_static)
    add_library(libLLVMDemangle_static STATIC IMPORTED)
    add_dependencies(libLLVMDemangle_static llvm_clang)
    set_target_properties(libLLVMDemangle_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${LLVM_CLANG_INSTALL_PATH}/include
        IMPORTED_LOCATION             ${LLVM_CLANG_INSTALL_PATH}/lib/libLLVMDemangle.a
        )
endif()
