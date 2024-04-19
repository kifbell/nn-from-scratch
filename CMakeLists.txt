cmake_minimum_required(VERSION 3.25)
project(leetcode)

set(CMAKE_CXX_STANDARD 23)

# Add this line to include Eigen library
include_directories(${CMAKE_SOURCE_DIR}/eigen)
include_directories(${CMAKE_SOURCE_DIR}/EigenRand)

#enable_language(CUDA)  # Enable CUDA support in CMake

add_executable(leetcode main.cpp)
