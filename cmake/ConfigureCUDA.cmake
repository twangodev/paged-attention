include_guard(GLOBAL)

if(NOT CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "")
    find_program(CMAKE_CUDA_COMPILER NAMES nvcc PATHS /usr/local/cuda/bin)
endif()