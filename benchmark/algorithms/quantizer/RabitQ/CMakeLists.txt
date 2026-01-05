cmake_minimum_required(VERSION 3.12)
project(rabitq_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Disable stripping of debug symbols
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_SKIP_BUILD_RPATH FALSE)
# set(CMAKE_SKIP_INSTALL_RPATH FALSE)

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

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/opt/RaBitQ/src)
include_directories(/usr/include/eigen3)
include_directories(/usr/local/include/eigen3)

# Compiler flags for AVX2 and optimizations
# Add -g for debug symbols, -O0 for no optimization (easier debugging)
# Switch to -O3 for production builds
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -march=native -mavx2 -mfma")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE -DFAST_SCAN -DUSE_AVX2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Define constants based on dimensions
# You can add more as needed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DnumC=1 -DDIM=128 -DBB=128 -DB_QUERY=4")

# Create Python module
pybind11_add_module(rabitq_cpp rabitq_binding.cpp)

# # Force debug symbols and disable stripping
# target_compile_options(rabitq_cpp PRIVATE -g -O0)
# target_link_options(rabitq_cpp PRIVATE -g)
# set_target_properties(rabitq_cpp PROPERTIES
#     LINK_FLAGS "-g"
#     STRIP FALSE
# )

# Link against required libraries
target_link_libraries(rabitq_cpp PRIVATE OpenMP::OpenMP_CXX)

# Installation
install(TARGETS rabitq_cpp DESTINATION .)
