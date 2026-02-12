cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(simt_sample LANGUAGES ASC CXX)

add_executable(gather
    $<$<STREQUAL:${GATHER_VERSION},v1>:gather_v1.asc>
    $<$<STREQUAL:${GATHER_VERSION},v2>:gather_v2.asc>
)

target_compile_options(gather PRIVATE
    # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:-g -O0>>
)

target_link_directories(gather PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:$ENV{ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib>>
)

target_link_libraries(gather PRIVATE
    m
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:npu_drv>>
)