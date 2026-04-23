cmake_minimum_required(VERSION 3.14)
project(bq_cpp_impl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The SIMD kernels use `__attribute__((target(...)))` so they compile
# regardless of the global -march. We still turn on aggressive opts +
# native tuning when available so the hot scalar / encode paths vectorize.
add_compile_options(-O3 -Wall -Wextra)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
endif()

find_package(OpenMP)

add_library(bq STATIC BinaryQuantizer.cpp)
target_include_directories(bq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(bq PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(bq_demo main.cpp)
target_link_libraries(bq_demo PRIVATE bq)
