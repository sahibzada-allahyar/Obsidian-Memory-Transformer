# Contributing to LTM Transformer

Thank you for your interest in contributing to LTM Transformer! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to allahyar@singularityresearch.org.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ltm-transformer.git
   cd ltm-transformer
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/singularityresearch/ltm-transformer.git
   ```
4. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- CMake (>= 3.15)
- CUDA Toolkit (>= 11.0)
- C++ Compiler with C++17 support
- Python (>= 3.7)
- PyTorch (>= 1.9.0)

### Building from Source

1. Install dependencies:
   ```bash
   # Install Python dependencies
   pip install -r requirements-dev.txt
   
   # Install system dependencies (Ubuntu/Debian)
   sudo apt-get install build-essential cmake cuda-toolkit-11-0
   ```

2. Build the project:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

3. Run tests:
   ```bash
   ctest --output-on-failure
   ```

## Making Changes

1. Make sure your changes are made on a new branch based on the latest main:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our [Style Guide](#style-guide)

3. Write or update tests as needed

4. Run the test suite to ensure everything works

5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```
   Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Testing

- Write unit tests for new functionality
- Update existing tests when modifying code
- Ensure all tests pass before submitting a PR
- Include performance benchmarks for performance-critical code

### Running Tests

```bash
# Run all tests
ctest --output-on-failure

# Run specific test suite
./tests/ltm_tests

# Run with sanitizers
./tests/ltm_tests_asan
./tests/ltm_tests_tsan
./tests/ltm_tests_ubsan

# Run performance benchmarks
./tests/ltm_perf_tests
```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation with any new features or APIs
3. Ensure all tests pass and CI checks are green
4. Get at least one code review from a maintainer
5. Once approved, a maintainer will merge your PR

### PR Title Format

Follow the Conventional Commits specification:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc)
- refactor: Code refactoring
- perf: Performance improvements
- test: Adding or updating tests
- chore: Maintenance tasks

## Style Guide

### C++

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use clang-format with the provided .clang-format file
- Use meaningful variable and function names
- Document public APIs using Doxygen-style comments
- Keep functions focused and reasonably sized
- Use const correctness
- Handle errors appropriately
- Use smart pointers for memory management

### CUDA

- Follow CUDA best practices for performance
- Use appropriate memory access patterns
- Handle errors using CUDA_CHECK macro
- Document kernel launch parameters
- Consider occupancy when setting block sizes
- Profile kernels to ensure optimal performance

### Python

- Follow PEP 8 style guide
- Use type hints
- Document functions and classes using docstrings
- Use black for code formatting
- Use isort for import sorting
- Use pylint for linting

## Documentation

- Document all public APIs
- Keep documentation up to date with code changes
- Include examples for complex features
- Document performance characteristics
- Update architecture docs for significant changes

### Documentation Structure

- API Reference: Detailed documentation of all public APIs
- Tutorials: Step-by-step guides for common tasks
- Examples: Sample code demonstrating features
- Architecture: High-level design documentation
- Performance: Benchmarks and optimization guidelines

## Questions or Problems?

Feel free to:
- Open an issue for bugs or feature requests
- Join our discussions for general questions
- Contact maintainers for security issues

Thank you for contributing to LTM Transformer!
