cmake_minimum_required(VERSION 3.15)
project(diskann_quantizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# DiskANN source path
# In Docker, DiskANN is at /opt/DiskANN
# For local builds, use environment variable or fallback to local path
if(DEFINED ENV{DISKANN_PATH})
    set(DISKANN_ROOT "$ENV{DISKANN_PATH}")
elseif(EXISTS "/opt/DiskANN")
    # Docker environment
    set(DISKANN_ROOT "/opt/DiskANN")
else()
    # Local development
    set(DISKANN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../tmp/DiskANN")
endif()

message(STATUS "Using DiskANN from: ${DISKANN_ROOT}")

# Include directories
include_directories(
    ${DISKANN_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# DiskANN source files - include all source files
file(GLOB DISKANN_SOURCES "${DISKANN_ROOT}/src/*.cpp")

# Create Python module
pybind11_add_module(diskann_cpp
    diskann_pybind.cpp
    ${DISKANN_SOURCES}
)

# Link libraries
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(diskann_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link libaio for async I/O
target_link_libraries(diskann_cpp PRIVATE aio)

# MKL (Intel Math Kernel Library)
if(DEFINED MKL_INCLUDE_PATH AND DEFINED MKL_PATH)
    message(STATUS "Using MKL from: ${MKL_PATH}")
    target_include_directories(diskann_cpp PRIVATE ${MKL_INCLUDE_PATH})
    target_link_directories(diskann_cpp PRIVATE ${MKL_PATH})
    target_link_libraries(diskann_cpp PRIVATE mkl_intel_lp64 mkl_gnu_thread mkl_core pthread m dl)
    target_compile_definitions(diskann_cpp PRIVATE MKL_ILP64)
else()
    find_package(MKL QUIET)
    if(MKL_FOUND)
        target_link_libraries(diskann_cpp PRIVATE ${MKL_LIBRARIES})
        target_include_directories(diskann_cpp PRIVATE ${MKL_INCLUDE_DIRS})
        target_compile_definitions(diskann_cpp PRIVATE USE_MKL)
    else()
        # Try to find MKL manually (for Intel oneAPI installation)
        if(EXISTS "/opt/intel/oneapi/mkl/latest/include")
            message(STATUS "Found Intel oneAPI MKL")
            target_include_directories(diskann_cpp PRIVATE /opt/intel/oneapi/mkl/latest/include)
            target_link_directories(diskann_cpp PRIVATE /opt/intel/oneapi/mkl/latest/lib/intel64)
            target_link_libraries(diskann_cpp PRIVATE mkl_intel_lp64 mkl_gnu_thread mkl_core pthread m dl)
            target_compile_definitions(diskann_cpp PRIVATE USE_MKL)
        else()
            message(STATUS "MKL not found, using OpenBLAS and LAPACKE")
            # Use OpenBLAS and LAPACKE
            target_link_libraries(diskann_cpp PRIVATE openblas lapacke)
        endif()
    endif()
endif()

# Boost (for program_options, not strictly needed for library)
find_package(Boost QUIET COMPONENTS program_options)
if(Boost_FOUND)
    target_include_directories(diskann_cpp PRIVATE ${Boost_INCLUDE_DIRS})
endif()

# Compiler flags (matching DiskANN's build settings)
target_compile_options(diskann_cpp PRIVATE
    -march=native
    -mtune=native
    -mavx2
    -mfma
    -msse2
    -ftree-vectorize
    -fopenmp
    -fopenmp-simd
    -funroll-loops
    -Wall
    -O3
    -DNDEBUG
    -DUSE_AVX2
)

# Install
install(TARGETS diskann_cpp LIBRARY DESTINATION .)
