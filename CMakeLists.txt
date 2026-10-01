cmake_minimum_required(VERSION 4.0)
project(ba)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package (Eigen3 REQUIRED NO_MODULE)

add_executable (RotationExample RotationExample.cpp)
target_link_libraries (RotationExample Eigen3::Eigen)

add_executable (MACGridExample MACGridExample.cpp)
target_link_libraries (MACGridExample Eigen3::Eigen)
