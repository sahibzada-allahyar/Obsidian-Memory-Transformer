# Models library configuration
add_library(ltm_models OBJECT
    config/model_config.cpp
    config/training_config.cpp
    config/inference_config.cpp
    config/distributed_config.cpp
    config/yaml_parser.cpp
    model_factory.cpp
    model_registry.cpp
    checkpointing.cpp
    base_model.cpp
    titan_model.cpp
    titan_config.cpp
    model_utils.cpp
)

target_compile_features(ltm_models PUBLIC cxx_std_17)

set_target_properties(ltm_models PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ltm_models PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Find YAML library for config parsing
find_package(yaml-cpp REQUIRED)

target_link_libraries(ltm_models PUBLIC
    ltm_core
    ltm_attention
    ltm_memory
    ltm_ops
    ltm_parallel
    ltm_quantization
    ltm_transformer
    yaml-cpp
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${CUTLASS_LIBRARIES}
)

# Model configuration files
set(MODEL_CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/config)
set(MODEL_CONFIG_FILES
    ${MODEL_CONFIG_DIR}/titan_base.yaml
    ${MODEL_CONFIG_DIR}/titan_small.yaml
    ${MODEL_CONFIG_DIR}/titan_medium.yaml
    ${MODEL_CONFIG_DIR}/titan_large.yaml
    ${MODEL_CONFIG_DIR}/titan_xl.yaml
)

# Install model library
install(TARGETS ltm_models
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install model configuration files
install(FILES ${MODEL_CONFIG_FILES}
    DESTINATION etc/ltm/models
)

# Install model schema files
install(FILES
    ${MODEL_CONFIG_DIR}/schema/model_schema.json
    ${MODEL_CONFIG_DIR}/schema/training_schema.json
    ${MODEL_CONFIG_DIR}/schema/inference_schema.json
    DESTINATION etc/ltm/models/schema
)

# Generate model configuration header
set(MODEL_CONFIG_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/config/model_config.h.in)
set(MODEL_CONFIG_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/model_config.h)

configure_file(${MODEL_CONFIG_TEMPLATE} ${MODEL_CONFIG_OUTPUT})

target_include_directories(ltm_models PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated config header
)

# Model registry configuration
set(MODEL_REGISTRY_FILE ${CMAKE_CURRENT_BINARY_DIR}/model_registry.cpp)
add_custom_command(
    OUTPUT ${MODEL_REGISTRY_FILE}
    COMMAND ${CMAKE_COMMAND}
        -DCONFIG_DIR=${MODEL_CONFIG_DIR}
        -DOUTPUT_FILE=${MODEL_REGISTRY_FILE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_registry.cmake
    DEPENDS ${MODEL_CONFIG_FILES}
    COMMENT "Generating model registry"
)

# Add model registry to sources
target_sources(ltm_models PRIVATE
    ${MODEL_REGISTRY_FILE}
)

# Model documentation
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/models.md
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/configurations.md
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/checkpointing.md
    DESTINATION share/doc/ltm/models
)

# Model utilities
install(PROGRAMS
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/convert_checkpoint.py
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/analyze_model.py
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/validate_config.py
    DESTINATION bin
)

# Testing support
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Python bindings support
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# Version information
set(LTM_MODELS_VERSION_MAJOR 0)
set(LTM_MODELS_VERSION_MINOR 1)
set(LTM_MODELS_VERSION_PATCH 0)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.h
)

# Export version information
set(LTM_MODELS_VERSION 
    "${LTM_MODELS_VERSION_MAJOR}.${LTM_MODELS_VERSION_MINOR}.${LTM_MODELS_VERSION_PATCH}"
    PARENT_SCOPE
)
