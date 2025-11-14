aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC)

add_library(asc_plugin SHARED
  ${SRC}
)

if(BUILD_OPEN_PROJECT)
    include(${CMAKE_SOURCE_DIR}/depend/third_party/llvm_clang.cmake)
    include(${CMAKE_SOURCE_DIR}/depend/third_party/json.cmake)
    add_dependencies(asc_plugin llvm_clang)
    target_include_directories(asc_plugin PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include/external
            ${CMAKE_CURRENT_SOURCE_DIR}/include/internal
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
            ${LLVM_CLANG_INSTALL_PATH}/include/
            ${JSON_INCLUDE_DIR}
        )
else()
    target_include_directories(asc_plugin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/external
        ${CMAKE_CURRENT_SOURCE_DIR}/include/internal
        ${TOP_DIR}/abl/mmpa/inc
    )
endif()

target_link_libraries(asc_plugin PRIVATE
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
    mmpa
    c_sec_static
    json
    pthread
    dl
    z
)

target_compile_options(asc_plugin PRIVATE
    -fPIE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -fno-strict-aliasing
    -fno-common
    -Wall
    -Wextra
    -Wfloat-equal
    -Werror
    -Wformat
    -D_FORTIFY_SOURCE=2
    -Os
    -DCMAKE_C_COMPILER_LAUNCHER=${CCACHE_PROGRAM}
    -DCMAKE_Cxx_COMPILER_LAUNCHER=${CCACHE_PROGRAM}
)

target_link_options(asc_plugin PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -Wl,-Bsymbolic
    -Wl,--exclude-libs=ALL
    -Wl,--gc-sections
)

install(TARGETS asc_plugin
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)