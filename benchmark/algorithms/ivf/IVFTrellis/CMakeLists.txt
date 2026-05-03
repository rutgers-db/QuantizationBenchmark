cmake_minimum_required(VERSION 3.12)
project(trellis_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/usr/include/eigen3)
include_directories(/usr/local/include/eigen3)

# AVX-512 VBMI is required for _mm512_permutex2var_epi8 (in-register table).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vbmi -mfma -ftree-vectorize -fexceptions -fpic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

pybind11_add_module(trellis_cpp trellis_binding.cpp)

target_link_libraries(trellis_cpp PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS trellis_cpp DESTINATION .)
