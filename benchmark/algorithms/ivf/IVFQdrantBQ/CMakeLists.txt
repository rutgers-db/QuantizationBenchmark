cmake_minimum_required(VERSION 3.16)
project(ivf_bq_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The dockerfile copies the QdrantBinaryQuantization sources next to this
# CMakeLists. For local builds, the same files can be found via a relative
# path into the sibling quantizer directory. Try both.
set(BQ_SRC_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/BinaryQuantizer.cpp")
set(BQ_SRC_SIBLING
    "${CMAKE_CURRENT_SOURCE_DIR}/../../quantizer/QdrantBinaryQuantization/BinaryQuantizer.cpp")

if(EXISTS ${BQ_SRC_LOCAL})
    set(BQ_SRC ${BQ_SRC_LOCAL})
    set(BQ_INC ${CMAKE_CURRENT_SOURCE_DIR})
elseif(EXISTS ${BQ_SRC_SIBLING})
    set(BQ_SRC ${BQ_SRC_SIBLING})
    get_filename_component(BQ_INC ${BQ_SRC} DIRECTORY)
else()
    message(FATAL_ERROR "Could not find BinaryQuantizer.cpp; expected at "
        "${BQ_SRC_LOCAL} or ${BQ_SRC_SIBLING}")
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${BQ_INC})

pybind11_add_module(ivf_bq_cpp
    ivf_bq_binding.cpp
    ${BQ_SRC}
)

target_link_libraries(ivf_bq_cpp PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS ivf_bq_cpp DESTINATION .)
