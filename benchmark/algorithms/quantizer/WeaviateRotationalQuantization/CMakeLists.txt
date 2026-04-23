cmake_minimum_required(VERSION 3.15)
project(rq_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -ffast-math -fno-math-errno -Wall -Wextra)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_compile_options(-mavx2 -mfma -mpopcnt -mbmi2)
endif()

find_package(OpenMP REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})

add_library(rq STATIC rq.cpp)
set_property(TARGET rq PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(rq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rq PUBLIC OpenMP::OpenMP_CXX)

add_executable(rq_example example.cpp)
target_link_libraries(rq_example PRIVATE rq)

pybind11_add_module(rq_cpp rq_binding.cpp)
target_link_libraries(rq_cpp PRIVATE rq OpenMP::OpenMP_CXX)

install(TARGETS rq_cpp DESTINATION .)
