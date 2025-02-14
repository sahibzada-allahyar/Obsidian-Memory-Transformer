# Inference library configuration
add_library(ltm_inference OBJECT
    inference.cpp
    model_loader.cpp
    tokenizer.cpp
    cache_manager.cpp
    batch_processor.cpp
    pipeline.cpp
    quantized_inference.cpp
    tensor_parallel_inference.cpp
)

target_compile_features(ltm_inference PUBLIC cxx_std_17)

set_target_properties(ltm_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(ltm_inference PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(ltm_inference PUBLIC
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

# Inference executable
add_executable(ltm_infer
    main.cpp
)

target_link_libraries(ltm_infer PRIVATE
    ltm_inference
    ltm
)

# Server dependencies
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(OpenSSL REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)

# Generate gRPC/Protobuf files
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/inference.proto
)
protobuf_generate_grpc_cpp(GRPC_SRCS GRPC_HDRS
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/inference.proto
)

# Server executable
add_executable(ltm_server
    server.cpp
    http_handler.cpp
    websocket_handler.cpp
    grpc_service.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
    ${GRPC_SRCS}
    ${GRPC_HDRS}
)

target_include_directories(ltm_server PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf files
    ${Boost_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)

target_link_libraries(ltm_server PRIVATE
    ltm_inference
    ltm
    Boost::system
    Boost::thread
    OpenSSL::SSL
    OpenSSL::Crypto
    gRPC::grpc++
    gRPC::grpc++_reflection
    protobuf::libprotobuf
)

# Install targets
install(TARGETS ltm_inference
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS ltm_infer ltm_server
    RUNTIME DESTINATION bin
)

# Install configuration files
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/config/inference_config.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/config/server_config.yaml
    DESTINATION etc/ltm/inference
)

# Install protocol files
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/inference.proto
    DESTINATION share/ltm/proto
)

# Install API documentation
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/api.md
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/server.md
    DESTINATION share/doc/ltm/inference
)

# Create version file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/version.h
)

# Add version information to server
target_include_directories(ltm_server PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Deployment scripts
install(PROGRAMS
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/deploy_server.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/monitor_server.sh
    DESTINATION bin
)

# Docker support
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Dockerfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Dockerfile
    @ONLY
)

# Add custom target for building Docker image
add_custom_target(docker_image
    COMMAND docker build -t ltm-server:${LTM_VERSION} .
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ltm_server
    COMMENT "Building Docker image for LTM server"
)
