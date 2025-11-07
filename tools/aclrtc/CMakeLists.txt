aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ SRC)

add_library(acl_rtc SHARED
    ${SRC}
)

if(BUILD_OPEN_PROJECT)
    target_include_directories(acl_rtc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/acl
        ${CCEC_PATH}/../include
    )
else()
    target_include_directories(acl_rtc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl/
        ${CCEC_PATH}/../include
    )
endif()

target_link_directories(acl_rtc PRIVATE
    ${CCEC_PATH}/../lib
)

target_link_libraries(acl_rtc PRIVATE
    $<BUILD_INTERFACE:intf_pub_cxx17>
    dl
)

target_compile_options(acl_rtc PRIVATE
    -fPIE
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

target_link_options(acl_rtc PRIVATE
    -Wl,-z,relro,-z,now,-z,noexecstack
    -Wl,-Bsymbolic
    -Wl,--gc-sections
)

install(TARGETS acl_rtc
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)