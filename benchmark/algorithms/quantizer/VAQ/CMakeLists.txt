cmake_minimum_required(VERSION 3.13)
project(vaq_cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Use pybind11 from Python package
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Compiler flags for optimizations
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-psabi -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=haswell -mavx -mavx2 -mfma -ffast-math")
# Remove EIGEN_USE_BLAS to avoid sgemm_ conflict in Python bindings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVAQ_OPTIMIZE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# Suppress all warnings to reduce noise
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fpermissive")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/opt/VAQ/bitvecengine)
include_directories(/opt/VAQ/external/eigen)

# add_definitions(-DEIGEN_DONT_PARALLELIZE )

# Create Python module
pybind11_add_module(vaq_cpp vaq_binding.cpp)

# Link against required libraries
target_link_libraries(vaq_cpp PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(vaq_cpp PRIVATE /opt/VAQ/build/bitvecengine/libVAQ.so)
target_link_libraries(vaq_cpp PRIVATE /opt/VAQ/build/bitvecengine/libFPGrowth.so)
target_link_libraries(vaq_cpp PRIVATE glpk)
target_link_libraries(vaq_cpp PRIVATE blas)
target_link_libraries(vaq_cpp PRIVATE lapack)

# Installation
install(TARGETS vaq_cpp DESTINATION .)
