### CMake Version ##############################################################

cmake_minimum_required(VERSION 3.10)

### Project Configuration ######################################################

get_filename_component(PROJECT_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
string(REPLACE " " "_" PROJECT_DIR_NAME ${PROJECT_DIR_NAME})

project(${PROJECT_DIR_NAME}
        VERSION 1.0.0.0 # <major>.<minor>.<patch>.<tweak>
        LANGUAGES CXX)

### List of Files ##############################################################

set(CONFIG
    ${PROJECT_SOURCE_DIR}/config/dbLogin.txt
)

set(INCLUDE
    ${PROJECT_SOURCE_DIR}/include/DBConnector.h
)

set(SRC
    ${PROJECT_SOURCE_DIR}/src/DBConnector.cpp
    ${PROJECT_SOURCE_DIR}/src/main.cpp
)

### Compiler Flags #############################################################

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
    include_directories(/usr/include/postgresql)
endif(UNIX AND NOT APPLE)

# GCC (Raspbian Buster for Raspberry Pi 4)
# Ubuntu rules apply here as well
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7l")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7l")

# Clang (macOS Catalina 10.15)
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
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

find_package(Boost REQUIRED
             COMPONENTS program_options)

find_package(PostgreSQL REQUIRED)

# Always enforce search for SHIFT LibMiscUtils
# (necessary when using the same name for Debug and Release)
set (LIBMISCUTILS "LIBMISCUTILS-NOTFOUND")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    find_library(LIBMISCUTILS shift_miscutils-d)
endif(CMAKE_BUILD_TYPE MATCHES Debug)
if(CMAKE_BUILD_TYPE MATCHES Release)
    find_library(LIBMISCUTILS shift_miscutils)
endif(CMAKE_BUILD_TYPE MATCHES Release)
if(NOT LIBMISCUTILS)
    message(FATAL_ERROR "SHIFT LibMiscUtils was not found.")
endif(NOT LIBMISCUTILS)

add_executable(${PROJECT_NAME}
               ${INCLUDE} ${SRC})

target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_PREFIX_PATH}/include
                           PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}
                      ${Boost_LIBRARIES}
                      ${PostgreSQL_LIBRARIES}
                      ${LIBMISCUTILS})

# Required when linking <experimental/filesystem> using GCC
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif(UNIX AND NOT APPLE)

### Install Configuration ######################################################

# If no installation path is set, the default is /usr/local
if(NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX /usr/local)
endif(NOT CMAKE_INSTALL_PREFIX)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        CONFIGURATIONS Release)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/config/
        DESTINATION share/shift-research/${PROJECT_NAME}
        CONFIGURATIONS Release)

################################################################################
