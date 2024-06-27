cmake_minimum_required(VERSION 3.10)
project(AVX512Example)

set(CMAKE_C_STANDARD 99)

# Configurar el compilador para usar AVX-512
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")

add_executable(avx512_example main.c)
