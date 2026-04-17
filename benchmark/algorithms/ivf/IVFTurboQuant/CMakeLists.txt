cmake_minimum_required(VERSION 3.12)
project(turbo_quant_cpp)

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

# OpenMP for multi-threaded train/add/query
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# AVX-512 + AVX2 + FMA — required by turbo_quant_index_unified.hpp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx512f -mavx2 -mfma")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

pybind11_add_module(turbo_quant_cpp turbo_quant_binding.cpp)

target_link_libraries(turbo_quant_cpp PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS turbo_quant_cpp DESTINATION .)
