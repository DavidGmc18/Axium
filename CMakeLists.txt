cmake_minimum_required(VERSION 3.22)

project(Axium LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(CUDA_INCLUDE_DIR "${CUDA_PATH}/include")
set(CUDA_LIB_DIR "${CUDA_PATH}/lib/x64")

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cu
)

add_library(Axium SHARED ${SRC_FILES})

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








# cmake_minimum_required(VERSION 3.22)

# project(Axium LANGUAGES C CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_ARCHITECTURES 86)

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # ---- CUDA setup ----
# set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
# set(CUDA_INCLUDE_DIR "${CUDA_PATH}/include")
# set(CUDA_LIB_DIR "${CUDA_PATH}/lib/x64")

# include_directories(${CUDA_INCLUDE_DIR})
# link_directories(${CUDA_LIB_DIR})

# # ---- Project includes ----
# include_directories(${CMAKE_SOURCE_DIR}/include)

# # ---- Source files ----
# file(GLOB_RECURSE SRC_FILES
#     ${CMAKE_SOURCE_DIR}/src/*.cpp
#     ${CMAKE_SOURCE_DIR}/src/*.c
#     ${CMAKE_SOURCE_DIR}/src/*.cu
# )

# # ---- Library target ----
# add_library(Axium SHARED ${SRC_FILES})

# # ---- Compiler/link flags ----
# if (WIN32)
#     target_compile_definitions(Axium PRIVATE AXIUM_EXPORTS)
# endif()

# target_link_libraries(Axium PRIVATE cudart_static)

# set_target_properties(Axium PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )

# # ---- Build the test executable ----
# file(GLOB TEST_SOURCES ${CMAKE_SOURCE_DIR}/test/*.cpp)
# add_executable(test ${TEST_SOURCES})
# target_link_libraries(test PRIVATE Axium)