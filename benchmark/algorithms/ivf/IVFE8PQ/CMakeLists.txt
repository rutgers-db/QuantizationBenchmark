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

# Enable AVX512VBMI (Ice Lake / Sapphire Rapids / Zen 4+) for the FastScan
# register-resident LUT path in IVFE8PQFastScan. If the compiler/CPU does not
# support VBMI, the fastscan code automatically falls back to a 32-wide gather
# from the float LUT.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx512vbmi")
check_cxx_source_compiles("
    #include <immintrin.h>
    int main() {
        __m512i a = _mm512_set1_epi8(0);
        __m512i b = _mm512_set1_epi8(0);
        __m512i c = _mm512_permutex2var_epi8(a, b, a);
        return _mm_cvtsi128_si32(_mm512_castsi512_si128(c));
    }" HAVE_AVX512VBMI)
unset(CMAKE_REQUIRED_FLAGS)
if(HAVE_AVX512VBMI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vbmi")
    message(STATUS "IVFE8PQ: AVX512VBMI enabled (register-resident LUT path)")
else()
    message(STATUS "IVFE8PQ: AVX512VBMI not available, falling back to 32-wide gather")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

pybind11_add_module(e8pq_cpp e8pq_binding.cpp)

target_link_libraries(e8pq_cpp PRIVATE OpenMP::OpenMP_CXX faiss)

install(TARGETS e8pq_cpp DESTINATION .)
