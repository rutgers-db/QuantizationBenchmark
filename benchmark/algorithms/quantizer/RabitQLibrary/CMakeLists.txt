cmake_minimum_required(VERSION 3.12)
project(rabitqlib_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include directories: rabitqlib headers live under include/
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/usr/include/eigen3)
include_directories(/usr/local/include/eigen3)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE -DFAST_SCAN -DUSE_AVX2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Create Python module
pybind11_add_module(rabitqlib_cpp rabitqlib_binding.cpp)

# Link against required libraries
target_link_libraries(rabitqlib_cpp PRIVATE OpenMP::OpenMP_CXX)

# Installation
install(TARGETS rabitqlib_cpp DESTINATION .)
