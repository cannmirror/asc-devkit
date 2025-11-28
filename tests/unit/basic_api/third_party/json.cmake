set(JSON_NAME "json")
set(JSON_PATH ${ASCENDC_TOOLS_ROOT_DIR}/third_party/json_src)

# 默认配置的json不存在则下载
if (NOT EXISTS "${JSON_PATH}/include/nlohmann/json.hpp")
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