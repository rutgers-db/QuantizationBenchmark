cmake_minimum_required(VERSION 3.16)
project(osq_cpp)

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

pybind11_add_module(osq_cpp
    osq_binding.cpp
    OptimizedScalarQuantizer.cpp
    OSQIndex.cpp
)

target_link_libraries(osq_cpp PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS osq_cpp DESTINATION .)
