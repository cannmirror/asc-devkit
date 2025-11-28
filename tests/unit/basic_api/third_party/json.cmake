set(JSON_NAME "json")

unset(json_FOUND CACHE)
unset(JSON_INCLUDE CACHE)

set(JSON_DOWNLOAD_PATH ${CANN_3RD_LIB_PATH}/pkg)
set(JSON_PATH ${CANN_3RD_LIB_PATH}/json)

find_path(JSON_INCLUDE
        NAMES nlohmann/json.hpp
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${JSON_PATH}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json
        FOUND_VAR
        json_FOUND
        REQUIRED_VARS
        JSON_INCLUDE
        )

if(json_FOUND)
    message("json found in ${JSON_PATH}")
    set(JSON_INCLUDE_DIR ${JSON_PATH}/include)
    add_library(json INTERFACE)

else ()
    # 默认配置的json不存在则下载
    set(JSON_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz")
    message(STATUS "Downloading ${JSON_NAME} from ${JSON_URL}")

    include(FetchContent)
    FetchContent_Declare(
        ${JSON_NAME}
        URL ${JSON_URL}
        URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
        DOWNLOAD_DIR ${third_party_TEM_DIR}
        SOURCE_DIR "${JSON_PATH}"  # 直接解压到此目录
        TLS_VERIFY OFF
    )
    FetchContent_MakeAvailable(${JSON_NAME})
endif()