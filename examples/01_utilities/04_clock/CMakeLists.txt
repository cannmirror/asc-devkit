cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(demo
    gather.asc
)

target_compile_options(demo PRIVATE
    # 该NPU架构版本号暂未正式发布，后续可能会有变更，请关注正式版本发布信息
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:-g -O0>>
)

target_link_directories(demo PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:$ENV{ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib>>
)

target_link_libraries(demo PRIVATE 
    m
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:npu_drv>>
)