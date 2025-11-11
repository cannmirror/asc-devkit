cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

add_executable(add_tque
    basic_api_tque_add.asc
)
add_executable(add
    add.asc
)
add_executable(add_basic
    basic_api_memory_allocator_add.asc
)

target_compile_options(add_tque PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
target_compile_options(add PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
target_compile_options(add_basic PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)