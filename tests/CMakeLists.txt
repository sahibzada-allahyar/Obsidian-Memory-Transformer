# Tests configuration
enable_testing()

# Find GTest package
find_package(GTest REQUIRED)

# Test executables
add_executable(ltm_tests
    attention/flash_attention_test.cu
    attention/memory_attention_test.cu
    ltm/compression_gate_test.cu
    ltm/memory_bank_test.cu
    ops/fused_ops_test.cu
    ops/mma_ops_test.cu
    parallel/mpi_utils_test.cpp
    parallel/pipeline_test.cpp
    parallel/tensor_parallel_test.cpp
    quantization/calibrator_test.cu
    quantization/quantizer_test.cu
    transformer/titan_inspired_block_test.cu
    main_test.cpp
)

target_compile_features(ltm_tests PRIVATE cxx_std_17)

target_include_directories(ltm_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(ltm_tests PRIVATE
    ltm
    GTest::GTest
    GTest::Main
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${CUTLASS_LIBRARIES}
)

# Set test properties
set_target_properties(ltm_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(ltm_tests)

# Performance tests
add_executable(ltm_perf_tests
    performance/attention_perf_test.cu
    performance/memory_perf_test.cu
    performance/transformer_perf_test.cu
    performance/main_perf_test.cpp
)

target_compile_features(ltm_perf_tests PRIVATE cxx_std_17)

target_include_directories(ltm_perf_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(ltm_perf_tests PRIVATE
    ltm
    GTest::GTest
    GTest::Main
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${CUTLASS_LIBRARIES}
)

set_target_properties(ltm_perf_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Custom targets for running specific test groups
add_custom_target(run_unit_tests
    COMMAND ltm_tests
    DEPENDS ltm_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(run_perf_tests
    COMMAND ltm_perf_tests
    DEPENDS ltm_perf_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Test data directory
set(TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${TEST_DATA_DIR})

# Copy test data to build directory
add_custom_command(TARGET ltm_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEST_DATA_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/data
)

# Test configuration
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/test_config.h
)

target_include_directories(ltm_tests PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(ltm_perf_tests PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Coverage target (if enabled)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_COMPILER_IS_GNUCXX)
        include(CodeCoverage)
        append_coverage_compiler_flags()
        setup_target_for_coverage_gcovr_html(
            NAME coverage
            EXECUTABLE ltm_tests
            DEPENDENCIES ltm_tests
        )
    endif()
endif()

# Memory check target
find_program(VALGRIND "valgrind")
if(VALGRIND)
    add_custom_target(memcheck
        COMMAND ${VALGRIND} --tool=memcheck --leak-check=full --show-reachable=yes
                --num-callers=20 --track-origins=yes
                $<TARGET_FILE:ltm_tests>
        DEPENDS ltm_tests
    )
endif()

# Sanitizer builds
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Address sanitizer
    add_executable(ltm_tests_asan ${TEST_SOURCES})
    target_compile_options(ltm_tests_asan PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(ltm_tests_asan PRIVATE -fsanitize=address)
    target_link_libraries(ltm_tests_asan PRIVATE ltm GTest::GTest GTest::Main)
    
    # Thread sanitizer
    add_executable(ltm_tests_tsan ${TEST_SOURCES})
    target_compile_options(ltm_tests_tsan PRIVATE -fsanitize=thread)
    target_link_options(ltm_tests_tsan PRIVATE -fsanitize=thread)
    target_link_libraries(ltm_tests_tsan PRIVATE ltm GTest::GTest GTest::Main)
    
    # UB sanitizer
    add_executable(ltm_tests_ubsan ${TEST_SOURCES})
    target_compile_options(ltm_tests_ubsan PRIVATE -fsanitize=undefined)
    target_link_options(ltm_tests_ubsan PRIVATE -fsanitize=undefined)
    target_link_libraries(ltm_tests_ubsan PRIVATE ltm GTest::GTest GTest::Main)
endif()
