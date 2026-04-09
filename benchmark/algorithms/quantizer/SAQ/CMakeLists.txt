cmake_minimum_required(VERSION 3.12)
project(saq_cpp)

set(CMAKE_CXX_STANDARD 20)
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

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/opt/SAQ/saqlib)
include_directories(/opt/SAQ/saqlib/third)

# Compiler flags: C++20, AVX-512, optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Create Python module
pybind11_add_module(saq_cpp saq_binding.cpp)

# Link against required libraries
target_link_libraries(saq_cpp PRIVATE
    OpenMP::OpenMP_CXX
    fmt
    glog
    gflags
)

# Installation
install(TARGETS saq_cpp DESTINATION .)
