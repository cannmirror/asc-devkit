add_library(aicpu_api STATIC aicpu_dump.cpp)

target_compile_options(aicpu_api PRIVATE
    -O2
    -fPIC
    -std=c++17
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -fstack-protector-strong
    -ftrapv
)

if(BUILD_OPEN_PROJECT)
    target_include_directories(aicpu_api PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
else()
    target_include_directories(aicpu_api PRIVATE
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/asc/asc-devkit/include
    )
endif()

target_compile_definitions(aicpu_api PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},device>:__AICPU_DEVICE__>
    _GLIBCXX_USE_CXX11_ABI=0>
    _FORTIFY_SOURCE=2>
)

install(TARGETS aicpu_api
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL
)

install(FILES
    ${ASCENDC_DIR}/include/aicpu_api/aicpu_api.h
    DESTINATION ${INSTALL_LIBRARY_DIR}/asc/include/aicpu_api OPTIONAL
)