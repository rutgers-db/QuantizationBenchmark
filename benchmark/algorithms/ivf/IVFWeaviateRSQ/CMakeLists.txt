cmake_minimum_required(VERSION 3.16)
project(ivf_rq_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Pull rq.{h,cpp} either from a copy placed next to this CMakeLists (docker
# build) or from the sibling quantizer directory (local build).
set(RQ_SRC_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/rq.cpp")
set(RQ_SRC_SIBLING
    "${CMAKE_CURRENT_SOURCE_DIR}/../../quantizer/WeaviateRotationalQuantization/rq.cpp")

if(EXISTS ${RQ_SRC_LOCAL})
    set(RQ_SRC ${RQ_SRC_LOCAL})
    set(RQ_INC ${CMAKE_CURRENT_SOURCE_DIR})
elseif(EXISTS ${RQ_SRC_SIBLING})
    set(RQ_SRC ${RQ_SRC_SIBLING})
    get_filename_component(RQ_INC ${RQ_SRC} DIRECTORY)
else()
    message(FATAL_ERROR "Could not find rq.cpp; expected at "
        "${RQ_SRC_LOCAL} or ${RQ_SRC_SIBLING}")
endif()

add_compile_options(-O3 -ffast-math -fno-math-errno -Wall -Wextra)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options(-mavx2 -mfma -mpopcnt -mbmi2)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${RQ_INC})

pybind11_add_module(ivf_rq_cpp
    ivf_rq_binding.cpp
    ${RQ_SRC}
)

target_link_libraries(ivf_rq_cpp PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS ivf_rq_cpp DESTINATION .)
