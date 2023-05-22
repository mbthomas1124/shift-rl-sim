### CMake Version ##############################################################

cmake_minimum_required(VERSION 3.10)

### Project Configuration ######################################################

get_filename_component(PROJECT_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" PROJECT_DIR_NAME ${PROJECT_DIR_NAME})

project(${PROJECT_DIR_NAME}
        VERSION 1.0.0.0 # <major>.<minor>.<patch>.<tweak>
        LANGUAGES CXX)

### List of Files ##############################################################

set(INCLUDE
    ${PROJECT_SOURCE_DIR}/include/ZITrader.h
)

set(SRC
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/ZITrader.cpp
)

set(OTHER
    ${PROJECT_SOURCE_DIR}/batch.sh
    ${PROJECT_SOURCE_DIR}/batch2.sh
    ${PROJECT_SOURCE_DIR}/initiator.cfg
)

### Compiler Flags #############################################################

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_definitions("-DHAVE_STD_UNIQUE_PTR=1") # QuickFIX C++17 compatibility req.

# Common Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexceptions -pedantic-errors")

# Debug Flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

# Release Flags
# -O2 instead of -O3
# -ftlo stands for Link Time Optimization (LTO)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -finline-functions -flto")

# GCC (Ubuntu 20.04 LTS Focal Fossa)
if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif(UNIX AND NOT APPLE)

# GCC (Raspbian Buster for Raspberry Pi 4)
# Ubuntu rules apply here as well
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7l")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7l")

# Clang (macOS Catalina 10.15)
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overloaded-virtual")
    include_directories(/usr/local/include)

    # set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=-*,cppcoreguidelines-*,modernize-*,readability-*")
    # set(CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY},-cppcoreguidelines-avoid-magic-numbers")
    # set(CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY},-cppcoreguidelines-macro-usage")
    # set(CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY},-cppcoreguidelines-pro-bounds-array-to-pointer-decay")
    # set(CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY},-cppcoreguidelines-pro-bounds-pointer-arithmetic")
    # set(CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY},-readability-magic-numbers")
endif(APPLE)

### Build Types ################################################################

# if no build type is set, the default is Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE})

### Build Configuration ########################################################

# Always enforce search for SHIFT LibCoreClient
# (necessary when using the same name for Debug and Release)
set (LIBCORECLIENT "LIBCORECLIENT-NOTFOUND")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    find_library(LIBCORECLIENT shift_coreclient-d)
endif(CMAKE_BUILD_TYPE MATCHES Debug)
if(CMAKE_BUILD_TYPE MATCHES Release)
    find_library(LIBCORECLIENT shift_coreclient)
endif(CMAKE_BUILD_TYPE MATCHES Release)
if(NOT LIBCORECLIENT)
    message(FATAL_ERROR "SHIFT LibCoreClient was not found.")
endif(NOT LIBCORECLIENT)

add_executable(${PROJECT_NAME}
               ${INCLUDE} ${SRC} ${OTHER})

target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_PREFIX_PATH}/include
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}
                      ${LIBCORECLIENT})

# initiator.cfg
configure_file(${PROJECT_SOURCE_DIR}/initiator.cfg
               ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/initiator.cfg
               COPYONLY)

################################################################################
