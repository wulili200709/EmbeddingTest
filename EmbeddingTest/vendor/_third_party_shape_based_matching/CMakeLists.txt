cmake_minimum_required(VERSION 3.10)
project(shape_based_matching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators (Ninja/Unix Makefiles)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Common compile flags
if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wno-sign-compare)

    # arm or x86
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        add_compile_options(-mfpu=neon)
    else()
        add_compile_options(-march=native)

        # some places of the algorithm are designed for 128 SIMD
        # so 128 SSE may slightly faster than 256 AVX, you may want this
        # add_compile_options(-msse -msse2 -msse3 -msse4 -mssse3)  # SSE only
    endif()
endif()

# OpenMP (MSVC is disabled by default due unsupported custom reduction pragma)
if(MSVC)
    option(SHAPE_USE_OPENMP "Enable OpenMP acceleration" OFF)
else()
    option(SHAPE_USE_OPENMP "Enable OpenMP acceleration" ON)
endif()

# OpenCV
if(DEFINED ENV{OpenCV_DIR} AND NOT OpenCV_DIR)
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
endif()
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS 3.0)
    message(FATAL_ERROR "OpenCV >= 3.0 is required, found ${OpenCV_VERSION}")
endif()

# test exe
add_executable(${PROJECT_NAME}_test line2Dup.cpp test.cpp)
target_include_directories(${PROJECT_NAME}_test PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/MIPP
)
target_compile_definitions(${PROJECT_NAME}_test PRIVATE
    SHAPE_TEST_PREFIX_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/test"
)
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${OpenCV_LIBS})

if(SHAPE_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${PROJECT_NAME}_test PRIVATE OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP requested but not found, building without OpenMP.")
    endif()
endif()

