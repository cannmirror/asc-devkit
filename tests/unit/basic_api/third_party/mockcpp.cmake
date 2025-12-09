# function(create_opensource target_name suffix_name product_side install_prefix toolchain_file)

set(open_source_target_name mockcpp)

set(mockcpp_SRC_DIR ${ASCENDC_TOOLS_ROOT_DIR}/third_party/mockcpp_src)

if (CMAKE_HOST_SYSTEM_PROCESSOR  STREQUAL "aarch64")
    set(mockcpp_CXXFLAGS "-fPIC")
else()
    set(mockcpp_CXXFLAGS "-fPIC -std=c++11")
endif()
set(mockcpp_FLAGS "-fPIC")
set(mockcpp_LINKER_FLAGS "")

if ((NOT DEFINED ABI_ZERO) OR (ABI_ZERO STREQUAL ""))
    set(ABI_ZERO "true")
endif()


if (ABI_ZERO STREQUAL true)
    set(mockcpp_CXXFLAGS "${mockcpp_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(mockcpp_FLAGS "${mockcpp_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(BOOST_INCLUDE_DIRS ${mockcpp_SRC_DIR}/../boost_src)
if (NOT EXISTS "${CMAKE_INSTALL_PREFIX}/mockcpp/lib/libmockcpp.a" OR TRUE)
    set(PATCH_FILE ${third_party_TEM_DIR}/mockcpp-2.7_py3.patch)
    if (NOT EXISTS ${PATCH_FILE})
        file(DOWNLOAD
            "https://raw.gitcode.com/cann-src-third-party/mockcpp/blobs/7207d936a909ab59b7748da8c63c2419ae37297f/mockcpp-2.7_py3.patch"
            ${PATCH_FILE}
            TIMEOUT 60
            EXPECTED_HASH SHA256=f1e9091992bf5c340af7d8c2f800b8d43d198fe4a8130f7bcd3f7cba1b0a324b
        )
    endif()
    include(ExternalProject)
    ExternalProject_Add(mockcpp
        URL "https://raw.gitcode.com/cann-src-third-party/mockcpp/blobs/868e1f78ddb352f201145283e0b8761a6245bde1/mockcpp-2.7.tar.gz"
        URL_HASH SHA256=73ab0a8b6d1052361c2cebd85e022c0396f928d2e077bf132790ae3be766f603
        DOWNLOAD_DIR ${third_party_TEM_DIR}
        SOURCE_DIR ${mockcpp_SRC_DIR}
        TLS_VERIFY OFF
        PATCH_COMMAND git init && git apply ${PATCH_FILE} && sed -i
        "1icmake_minimum_required(VERSION 3.16.0)" CMakeLists.txt && rm -rf .git

        CONFIGURE_COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR}
            -DCMAKE_CXX_FLAGS=${mockcpp_CXXFLAGS}
            -DCMAKE_C_FLAGS=${mockcpp_FLAGS}
            -DBOOST_INCLUDE_DIRS=${BOOST_INCLUDE_DIRS}
            -DCMAKE_SHARED_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
            -DCMAKE_EXE_LINKER_FLAGS=${mockcpp_LINKER_FLAGS}
            -DBUILD_32_BIT_TARGET_BY_64_BIT_COMPILER=OFF
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/mockcpp
            <SOURCE_DIR>
        BUILD_COMMAND make install -j 16
    )
endif()

set(MOCKCPP_DIR ${CMAKE_INSTALL_PREFIX}/mockcpp)

set(MOCKCPP_INCLUDE_ONE ${MOCKCPP_DIR}/include)

set(MOCKCPP_INCLUDE_TWO ${BOOST_INCLUDE_DIRS})

set(MOCKCPP_STATIC_LIBRARY ${MOCKCPP_DIR}/lib/libmockcpp.a)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mockcpp
    REQUIRED_VARS MOCKCPP_INCLUDE_ONE MOCKCPP_INCLUDE_TWO MOCKCPP_STATIC_LIBRARY
)

message("mockcpp_FOUND is ${mockcpp_FOUND}")

if(mockcpp_FOUND)
    set(MOCKCPP_INCLUDE_DIR ${MOCKCPP_INCLUDE_ONE} ${MOCKCPP_INCLUDE_TWO})
    get_filename_component(MOCKCPP_LIBRARY_DIR ${MOCKCPP_STATIC_LIBRARY} DIRECTORY)

    if(NOT TARGET mockcpp_static)
        add_library(mockcpp_static STATIC IMPORTED)
        set_target_properties(mockcpp_static PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MOCKCPP_INCLUDE_DIR}"
            IMPORTED_LOCATION "${MOCKCPP_STATIC_LIBRARY}"
            )
        add_dependencies(mockcpp_static mockcpp)
    endif()
endif()