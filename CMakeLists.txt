cmake_minimum_required(VERSION 3.22)

set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)

project(Axium LANGUAGES C CXX CUDA)

# set(CMAKE_CUDA_HOST_COMPILER "cl.exe")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(CUDA_INCLUDE_DIR "${CUDA_PATH}/include")
set(CUDA_LIB_DIR "${CUDA_PATH}/lib/x64")

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
)

add_library(Axium SHARED ${SRC_FILES})

if(MSVC)
    target_compile_options(Axium PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/O1 /Ob0 /Z7 /Oi /RTC1 /arch:AVX2>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C>>:/O1 /Ob0 /Z7 /Oi /RTC1 /arch:AVX2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /Ob2 /Oi /MDd /DNDEBUG /fp:fast /arch:AVX2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:/O2 /Ob2 /Oi /MDd /DNDEBUG /fp:fast /arch:AVX2>
    )
else()
    target_compile_options(Axium PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O1 -g>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C>>:-O1 -g>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -Ofast -march=native -ffast-math -funroll-loops>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:-O3 -Ofast -march=native -ffast-math -funroll-loops>
    )
endif()

target_compile_options(Axium PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17>
    $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>
)

target_include_directories(Axium PUBLIC
    ${CUDA_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(Axium PRIVATE AXIUM_EXPORTS)
target_compile_definitions(Axium
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
)

target_link_directories(Axium PRIVATE ${CUDA_LIB_DIR})
target_link_libraries(Axium PRIVATE cudart_static)

set_target_properties(Axium PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ---- Build the test executable ----
file(GLOB TEST_SOURCES ${CMAKE_SOURCE_DIR}/test/*.cpp)
add_executable(test ${TEST_SOURCES})
target_link_libraries(test PRIVATE Axium)