cmake_minimum_required(VERSION 3.14)
project(bq_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -Wall -Wextra)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Core library (reused by both bq_demo and the python extension)
add_library(bq STATIC BinaryQuantizer.cpp)
set_property(TARGET bq PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(bq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bq PUBLIC OpenMP::OpenMP_CXX)

# Standalone CLI demo (kept for parity with the original tree).
add_executable(bq_demo main.cpp)
target_link_libraries(bq_demo PRIVATE bq)

# Python module
pybind11_add_module(bq_cpp bq_binding.cpp)
target_link_libraries(bq_cpp PRIVATE bq OpenMP::OpenMP_CXX)

install(TARGETS bq_cpp DESTINATION .)
