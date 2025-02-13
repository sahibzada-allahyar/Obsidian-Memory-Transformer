# Trainer library configuration
add_library(ltm_trainer OBJECT
    trainer.cpp
    optimizer.cpp
    scheduler.cpp
    data_loader.cpp
    checkpoint.cpp
    metrics.cpp
    distributed.cpp
    profiler.cpp
)

target_compile_features(ltm_trainer PUBLIC cxx_std_17)

set_target_properties(ltm_trainer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ltm_trainer PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(ltm_trainer PUBLIC
    ltm_core
    ltm_attention
    ltm_memory
    ltm_ops
    ltm_parallel
    ltm_quantization
    ltm_transformer
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${CUTLASS_LIBRARIES}
)

# Trainer executable
add_executable(ltm_train
    main.cpp
)

target_link_libraries(ltm_train PRIVATE
    ltm_trainer
    ltm
)

# Install targets
install(TARGETS ltm_trainer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS ltm_train
    RUNTIME DESTINATION bin
)

# Configuration files
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/config/default_config.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/config/distributed_config.yaml
    DESTINATION etc/ltm/trainer
)
