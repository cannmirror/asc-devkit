aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../common/src/ SRC_COMMON)

add_executable(bishengcc
  ${SRC}
  ${SRC_COMMON}
)

target_include_directories(bishengcc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/
)

target_link_libraries(bishengcc PRIVATE
    $<BUILD_INTERFACE:intf_pub_cxx17>
    libLLVMSupport_static
)

target_compile_options(bishengcc PRIVATE
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
)

target_link_options(bishengcc PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -Wl,-Bsymbolic
    -Wl,--exclude-libs=ALL
    -Wl,--gc-sections
)

install(TARGETS bishengcc
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)