cmake_minimum_required(VERSION 3.12)
project(e8pq_cpp)

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

# faiss is built and installed from source by the Dockerfile; we link against
# the installed static/shared libfaiss and use its public headers.
find_package(faiss REQUIRED)

# rabitqlib headers (FHT-Kac rotator, l2norm_sqr, dot_product) are shared
# with IVFE8 — the Dockerfile copies that include tree into this algorithm's
# working directory before cmake runs.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(/usr/include/eigen3)
include_directories(/usr/local/include/eigen3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -ftree-vectorize -fexceptions -fpic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

pybind11_add_module(e8pq_cpp e8pq_binding.cpp)

target_link_libraries(e8pq_cpp PRIVATE OpenMP::OpenMP_CXX faiss)

install(TARGETS e8pq_cpp DESTINATION .)
