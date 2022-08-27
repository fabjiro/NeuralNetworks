cmake_minimum_required(VERSION 3.23)
project(NeuronalNetwork)

set(CMAKE_CXX_STANDARD 20)

add_executable(NeuronalNetwork main.cpp Matrix/matrix.cpp Matrix/matrix.h NeuronalNetwork/neuronalnetwork.cpp NeuronalNetwork/neuronalnetwork.h)
