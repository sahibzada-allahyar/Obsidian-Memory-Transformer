# Python bindings configuration
pybind11_add_module(ltm_python
    src/main.cpp
    src/model.cpp
    src/trainer.cpp
    src/inference.cpp
)

target_compile_features(ltm_python PRIVATE cxx_std_17)

target_include_directories(ltm_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${pybind11_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
)

# Link against our main library and its dependencies
target_link_libraries(ltm_python PRIVATE
    ltm
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${MPI_CXX_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${CUTLASS_LIBRARIES}
)

# Set output directory for the Python module
set_target_properties(ltm_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ltm
    OUTPUT_NAME "_ltm"  # The module will be imported as ltm._ltm
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Install Python package
install(TARGETS ltm_python
    LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES}/ltm
    ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}/ltm
)

install(FILES
    ltm/__init__.py
    ltm/model.py
    ltm/trainer.py
    ltm/inference.py
    DESTINATION ${PYTHON_SITE_PACKAGES}/ltm
)

# Copy Python files to build directory
add_custom_command(TARGET ltm_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/ltm/__init__.py
        ${CMAKE_CURRENT_SOURCE_DIR}/ltm/model.py
        ${CMAKE_CURRENT_SOURCE_DIR}/ltm/trainer.py
        ${CMAKE_CURRENT_SOURCE_DIR}/ltm/inference.py
        ${CMAKE_CURRENT_BINARY_DIR}/ltm/
)

# Setup.py configuration
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py
    @ONLY
)

# Build wheel package
add_custom_target(wheel
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ltm_python
)

# Development mode installation
add_custom_target(develop
    COMMAND ${PYTHON_EXECUTABLE} setup.py develop
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ltm_python
)

# Clean Python build artifacts
add_custom_target(clean_python
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/build
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/dist
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/ltm.egg-info
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/ltm/__pycache__
)
