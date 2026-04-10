cmake_minimum_required(VERSION 3.16)
project(residual_opq_ivfpq_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
find_package(OpenMP REQUIRED)
find_package(faiss CONFIG REQUIRED)

set(FAISS_LINK_TARGET faiss)
if(TARGET faiss_avx512_spr)
    set(FAISS_LINK_TARGET faiss_avx512_spr)
elseif(TARGET faiss_avx512)
    set(FAISS_LINK_TARGET faiss_avx512)
elseif(TARGET faiss_avx2)
    set(FAISS_LINK_TARGET faiss_avx2)
endif()

pybind11_add_module(residual_opq_ivfpq_cpp
    residual_opq_ivfpq_binding.cpp
)

target_link_libraries(residual_opq_ivfpq_cpp
    PRIVATE
        faiss
        OpenMP::OpenMP_CXX
)

if(NOT FAISS_LINK_TARGET STREQUAL "faiss")
    target_link_libraries(residual_opq_ivfpq_cpp PRIVATE ${FAISS_LINK_TARGET})
endif()

target_compile_definitions(residual_opq_ivfpq_cpp PRIVATE NDEBUG)
if(NOT CMAKE_BUILD_TYPE)
    set_target_properties(residual_opq_ivfpq_cpp PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

install(TARGETS residual_opq_ivfpq_cpp DESTINATION .)
