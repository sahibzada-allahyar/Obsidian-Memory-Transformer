cmake_minimum_required(VERSION 3.18)
project(ltm LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA configuration
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)  # Support Volta, Turing, Ampere
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra")

# Dependencies
find_package(MPI REQUIRED)
find_package(NCCL REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUTLASS REQUIRED)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
    ${NCCL_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${CUTLASS_INCLUDE_DIRS}
)

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Enable fast math for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Add subdirectories
add_subdirectory(src)
add_subdirectory(python_bindings)
add_subdirectory(tests)

# Main library target
add_library(ltm STATIC
    $<TARGET_OBJECTS:ltm_core>
    $<TARGET_OBJECTS:ltm_attention>
    $<TARGET_OBJECTS:ltm_memory>
    $<TARGET_OBJECTS:ltm_ops>
    $<TARGET_OBJECTS:ltm_parallel>
    $<TARGET_OBJECTS:ltm_quantization>
    $<TARGET_OBJECTS:ltm_transformer>
)

target_link_libraries(ltm PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CUSPARSE_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${CUTLASS_LIBRARIES}
)

# Installation
install(TARGETS ltm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h*"
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ltmConfigVersion.cmake"
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ltmConfigVersion.cmake"
    DESTINATION lib/cmake/ltm
)

# Export targets
export(TARGETS ltm FILE "${CMAKE_CURRENT_BINARY_DIR}/ltmTargets.cmake")
install(EXPORT ltmTargets
    FILE ltmTargets.cmake
    NAMESPACE ltm::
    DESTINATION lib/cmake/ltm
)
