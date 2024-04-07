cmake_minimum_required(VERSION 3.20)
project(hlk_ld2450 C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)

add_library(hlk STATIC src/hlk.c)
target_include_directories(hlk PUBLIC src)

add_executable(test test.cpp)
target_link_libraries(test hlk)
