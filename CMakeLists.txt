cmake_minimum_required(VERSION 3.25)
project(nn_from_scratch)

set(CMAKE_CXX_STANDARD 23)

# Add this line to include Eigen library
include_directories(${CMAKE_SOURCE_DIR}/eigen)
include_directories(${CMAKE_SOURCE_DIR}/EigenRand)

add_executable(nn_from_scratch main.cpp
        Optimizer.cpp
        Layer.cpp
        DataHandler.cpp
        Loss.cpp
        NeuralNetwork.cpp

)
#add_executable(nn_from_scratch Optimizer.cpp)
